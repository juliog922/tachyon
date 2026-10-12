//! AES-128-GCM (NIST SP 800-38D) on AES-NI and PCLMULQDQ. Ported from caudal.
//!
//! Eight counter blocks go through the AES rounds together so the pipelined `aesenc` units stay busy, and their
//! GHASH is folded with one reduction: `X·H⁸ ⊕ C₁·H⁸ ⊕ C₂·H⁷ ⊕ … ⊕ C₈·H`, with the powers of `H` computed once per
//! key. GHASH works on byte-reversed blocks (Intel's reflected method). [`Gcm::seal`] reads plaintext from any
//! number of slices and writes the ciphertext straight into the spare capacity of the output buffer;
//! [`Gcm::open`] decrypts into the output buffer and exposes the plaintext only once the tag checks.

use super::crypto::{ct_eq, wipe};
use core::arch::x86_64::{
    __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_aeskeygenassist_si128, _mm_clmulepi64_si128, _mm_insert_epi32, _mm_loadu_si128, _mm_or_si128,
    _mm_set_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_slli_epi32, _mm_slli_si128, _mm_srli_epi32, _mm_srli_si128, _mm_storeu_si128,
    _mm_xor_si128,
};

/// Bytes of a tag.
pub const TAG: usize = 16;
/// Longest message per nonce: the 32-bit block counter must not reach the tag's block again.
const MAX_MESSAGE: u64 = (1 << 36) - 32;

/// Whether this CPU has the instructions [`Gcm`] needs.
pub fn supported() -> bool {
    is_x86_feature_detected!("aes") && is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1")
}

/// An expanded AES-128 key and the GHASH key powers `H¹ … H⁸`, wiped on drop: five cache lines, the last one
/// completed by a zeroed slot so that no byte is padding.
#[repr(C, align(64))]
pub struct Gcm {
    round_keys: [__m128i; 11],
    h_powers: [__m128i; 8],
    spare: __m128i,
}

const _: () = assert!(size_of::<Gcm>() == 20 * 16, "Gcm has no padding");

impl Drop for Gcm {
    fn drop(&mut self) {
        // SAFETY: `Gcm` is plain data with no padding; its bytes are valid to overwrite.
        wipe(unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(self).cast::<u8>(), size_of::<Gcm>()) });
    }
}

impl Gcm {
    /// The cipher for `key`.
    ///
    /// # Panics
    ///
    /// Without AES-NI, PCLMULQDQ and SSE4.1 ([`supported`]).
    pub fn new(key: &[u8; 16]) -> Gcm {
        assert!(supported(), "AES-GCM needs AES-NI, PCLMULQDQ and SSE4.1");
        // SAFETY: the CPU has the instructions, checked above.
        unsafe { expand(key) }
    }

    /// Appends to `out` the encryption of the concatenated `parts` and its tag. A (key, nonce) pair must never
    /// seal two different messages.
    ///
    /// # Panics
    ///
    /// If the message is longer than 2^36 − 32 bytes, the most one nonce may seal.
    pub fn seal(&self, nonce: &[u8; 12], aad: &[u8], parts: &[&[u8]], out: &mut Vec<u8>) {
        let len: usize = parts.iter().map(|p| p.len()).sum();
        assert!(len as u64 <= MAX_MESSAGE, "AES-GCM seals at most 2^36 − 32 bytes per nonce");
        out.reserve(len + TAG);
        // SAFETY: the instructions were checked when the key was expanded; every write lands in the capacity
        // reserved above, which `set_len` then covers.
        unsafe {
            let mut at = out.as_mut_ptr().add(out.len());
            let mut stream = Stream::begin(self, nonce, aad);
            for part in parts {
                stream.apply(part.as_ptr(), at, part.len(), false);
                at = at.add(part.len());
            }
            let tag = stream.finish(aad.len(), len);
            std::ptr::copy_nonoverlapping(tag.as_ptr(), at, TAG);
            out.set_len(out.len() + len + TAG);
        }
    }

    /// Appends to `out` the plaintext of `sealed` (ciphertext and tag) when it is authentic; otherwise leaves
    /// `out` as it was and returns false.
    pub fn open(&self, nonce: &[u8; 12], aad: &[u8], sealed: &[u8], out: &mut Vec<u8>) -> bool {
        let Some(len) = sealed.len().checked_sub(TAG).filter(|&n| n as u64 <= MAX_MESSAGE) else { return false };
        out.reserve(len);
        // SAFETY: as in `seal`; the plaintext is written into reserved capacity and exposed only if the tag checks.
        unsafe {
            let start = out.as_mut_ptr().add(out.len());
            let mut stream = Stream::begin(self, nonce, aad);
            stream.apply(sealed.as_ptr(), start, len, true);
            let tag = stream.finish(aad.len(), len);
            if !ct_eq(&tag, &sealed[len..]) {
                wipe(std::slice::from_raw_parts_mut(start, len));
                return false;
            }
            out.set_len(out.len() + len);
        }
        true
    }
}

/// Marks functions as needing AES-NI, PCLMULQDQ and SSE4.1, which [`Gcm::new`] checks.
macro_rules! simd {
    ($($item:item)*) => { $(#[target_feature(enable = "aes,pclmulqdq,sse4.1")] $item)* };
}

simd! {
    fn expand(key: &[u8; 16]) -> Gcm {
        let mut rk = [_mm_setzero_si128(); 11];
        // SAFETY: `key` holds 16 bytes; an unaligned load.
        rk[0] = unsafe { _mm_loadu_si128(key.as_ptr().cast()) };
        macro_rules! round {
            ($($i:literal $rcon:literal),*) => { $(rk[$i] = schedule(rk[$i - 1], _mm_aeskeygenassist_si128::<$rcon>(rk[$i - 1]));)* };
        }
        round!(1 0x01, 2 0x02, 3 0x04, 4 0x08, 5 0x10, 6 0x20, 7 0x40, 8 0x80, 9 0x1b, 10 0x36);
        let h = reverse(encrypt(&rk, _mm_setzero_si128()));
        let mut h_powers = [h; 8];
        for i in 1..8 {
            h_powers[i] = multiply(h_powers[i - 1], h);
        }
        Gcm { round_keys: rk, h_powers, spare: _mm_setzero_si128() }
    }

    fn schedule(key: __m128i, assist: __m128i) -> __m128i {
        let mut k = key;
        k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
        k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
        k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
        _mm_xor_si128(k, _mm_shuffle_epi32::<0xff>(assist))
    }

    fn encrypt(rk: &[__m128i; 11], block: __m128i) -> __m128i {
        let mut b = _mm_xor_si128(block, rk[0]);
        for k in &rk[1..10] {
            b = _mm_aesenc_si128(b, *k);
        }
        _mm_aesenclast_si128(b, rk[10])
    }

    fn reverse(x: __m128i) -> __m128i {
        _mm_shuffle_epi8(x, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
    }

    /// The unreduced 256-bit carry-less product, as low, middle and high terms.
    fn product(a: __m128i, b: __m128i) -> [__m128i; 3] {
        let mid = _mm_xor_si128(_mm_clmulepi64_si128::<0x10>(a, b), _mm_clmulepi64_si128::<0x01>(a, b));
        [_mm_clmulepi64_si128::<0x00>(a, b), mid, _mm_clmulepi64_si128::<0x11>(a, b)]
    }

    /// Folds a sum of products into one field element: shifts the 256-bit value left by one (the reflected
    /// representation) and reduces it modulo x¹²⁸ + x⁷ + x² + x + 1.
    fn reduce([lo, mid, hi]: [__m128i; 3]) -> __m128i {
        let lo = _mm_xor_si128(lo, _mm_slli_si128::<8>(mid));
        let hi = _mm_xor_si128(hi, _mm_srli_si128::<8>(mid));
        let lo_carry = _mm_srli_epi32::<31>(lo);
        let hi_carry = _mm_srli_epi32::<31>(hi);
        let lo = _mm_or_si128(_mm_slli_epi32::<1>(lo), _mm_slli_si128::<4>(lo_carry));
        let hi = _mm_or_si128(_mm_or_si128(_mm_slli_epi32::<1>(hi), _mm_slli_si128::<4>(hi_carry)), _mm_srli_si128::<12>(lo_carry));
        let fold = _mm_xor_si128(_mm_xor_si128(_mm_slli_epi32::<31>(lo), _mm_slli_epi32::<30>(lo)), _mm_slli_epi32::<25>(lo));
        let lo = _mm_xor_si128(lo, _mm_slli_si128::<12>(fold));
        let back = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi32::<1>(lo), _mm_srli_epi32::<2>(lo)), _mm_srli_epi32::<7>(lo));
        _mm_xor_si128(hi, _mm_xor_si128(lo, _mm_xor_si128(back, _mm_srli_si128::<4>(fold))))
    }

    fn multiply(a: __m128i, b: __m128i) -> __m128i {
        reduce(product(a, b))
    }

    fn xor3(a: [__m128i; 3], b: [__m128i; 3]) -> [__m128i; 3] {
        [_mm_xor_si128(a[0], b[0]), _mm_xor_si128(a[1], b[1]), _mm_xor_si128(a[2], b[2])]
    }

    /// 16 bytes at `p`, which must be readable.
    unsafe fn load(p: *const u8) -> __m128i {
        // SAFETY: the caller's contract.
        unsafe { _mm_loadu_si128(p.cast()) }
    }
}

/// One message in flight: the counter, the GHASH accumulator, and the keystream and ciphertext of a block left
/// partly used by the previous slice.
struct Stream<'a> {
    gcm: &'a Gcm,
    nonce: __m128i,
    counter: u32,
    hash: __m128i,
    keystream: [u8; 16],
    cipher: [u8; 16],
    used: usize,
}

impl<'a> Stream<'a> {
    simd! {
        fn begin(gcm: &'a Gcm, nonce: &[u8; 12], aad: &[u8]) -> Stream<'a> {
            let mut block = [0u8; 16];
            block[..12].copy_from_slice(nonce);
            // SAFETY: `block` holds 16 bytes.
            let nonce = unsafe { load(block.as_ptr()) };
            let mut s = Stream { gcm, nonce, counter: 1, hash: _mm_setzero_si128(), keystream: [0; 16], cipher: [0; 16], used: 16 };
            for chunk in aad.chunks(16) {
                let mut b = [0u8; 16];
                b[..chunk.len()].copy_from_slice(chunk);
                // SAFETY: `b` holds 16 bytes.
                s.absorb(unsafe { load(b.as_ptr()) });
            }
            s
        }

        fn absorb(&mut self, cipher: __m128i) {
            self.hash = multiply(_mm_xor_si128(self.hash, reverse(cipher)), self.gcm.h_powers[0]);
        }

        fn counter_block(&mut self) -> __m128i {
            self.counter = self.counter.wrapping_add(1);
            _mm_insert_epi32::<3>(self.nonce, self.counter.to_be() as i32)
        }

        /// Encrypts (or decrypts) `len` bytes from `src` to `dst`, continuing where the last call stopped. Both
        /// must hold `len` bytes.
        unsafe fn apply(&mut self, src: *const u8, dst: *mut u8, len: usize, decrypt: bool) {
            // SAFETY: the caller's contract covers every access below, all within `len` bytes.
            unsafe {
                let mut i = self.bytes(src, dst, 0, len, decrypt);
                while len - i >= 128 {
                    self.eight(src.add(i), dst.add(i), decrypt);
                    i += 128;
                }
                let rk = &self.gcm.round_keys;
                while len - i >= 16 {
                    let input = load(src.add(i));
                    let output = _mm_xor_si128(encrypt(rk, self.counter_block()), input);
                    _mm_storeu_si128(dst.add(i).cast(), output);
                    self.absorb(if decrypt { input } else { output });
                    i += 16;
                }
                if i < len {
                    let k = encrypt(rk, self.counter_block());
                    _mm_storeu_si128(self.keystream.as_mut_ptr().cast(), k);
                    self.used = 0;
                    self.bytes(src, dst, i, len, decrypt);
                }
            }
        }

        /// Eight blocks from `src` to `dst` (128 bytes each), their GHASH folded with one reduction.
        unsafe fn eight(&mut self, src: *const u8, dst: *mut u8, decrypt: bool) {
            let rk = &self.gcm.round_keys;
            let mut b = [_mm_setzero_si128(); 8];
            for x in &mut b {
                *x = _mm_xor_si128(self.counter_block(), rk[0]);
            }
            for k in &rk[1..10] {
                for x in &mut b {
                    *x = _mm_aesenc_si128(*x, *k);
                }
            }
            let mut sum = [_mm_setzero_si128(); 3];
            for (j, x) in b.iter().enumerate() {
                // SAFETY: the caller's contract: 128 bytes at each pointer.
                let input = unsafe { load(src.add(16 * j)) };
                let output = _mm_xor_si128(_mm_aesenclast_si128(*x, rk[10]), input);
                // SAFETY: as above.
                unsafe { _mm_storeu_si128(dst.add(16 * j).cast(), output) };
                let mut c = reverse(if decrypt { input } else { output });
                if j == 0 {
                    c = _mm_xor_si128(c, self.hash);
                }
                sum = xor3(sum, product(c, self.gcm.h_powers[7 - j]));
            }
            self.hash = reduce(sum);
        }

        /// Uses up the keystream of a partial block, one byte at a time; returns the position reached.
        unsafe fn bytes(&mut self, src: *const u8, dst: *mut u8, mut i: usize, len: usize, decrypt: bool) -> usize {
            while self.used < 16 && i < len {
                // SAFETY: `i < len`, within the caller's buffers.
                let input = unsafe { *src.add(i) };
                let output = input ^ self.keystream[self.used];
                // SAFETY: as above.
                unsafe { *dst.add(i) = output };
                self.cipher[self.used] = if decrypt { input } else { output };
                self.used += 1;
                i += 1;
                if self.used == 16 {
                    // SAFETY: `cipher` holds 16 bytes.
                    self.absorb(unsafe { load(self.cipher.as_ptr()) });
                }
            }
            i
        }

        fn finish(mut self, aad_len: usize, len: usize) -> [u8; 16] {
            if self.used < 16 {
                self.cipher[self.used..].fill(0);
                // SAFETY: `cipher` holds 16 bytes.
                self.absorb(unsafe { load(self.cipher.as_ptr()) });
            }
            let mut lengths = [0u8; 16];
            lengths[..8].copy_from_slice(&(aad_len as u64 * 8).to_be_bytes());
            lengths[8..].copy_from_slice(&(len as u64 * 8).to_be_bytes());
            // SAFETY: `lengths` holds 16 bytes.
            self.absorb(unsafe { load(lengths.as_ptr()) });
            let first = _mm_insert_epi32::<3>(self.nonce, 1u32.to_be() as i32);
            let mut tag = [0u8; 16];
            // SAFETY: `tag` holds 16 bytes.
            unsafe { _mm_storeu_si128(tag.as_mut_ptr().cast(), _mm_xor_si128(reverse(self.hash), encrypt(&self.gcm.round_keys, first))) };
            wipe(&mut self.keystream);
            tag
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::crypto::tests::hex;

    /// NIST GCM test case 4 (AES-128, with associated data), and a long message through the eight-block path.
    #[test]
    fn gcm_matches_nist_vectors_and_round_trips() {
        let key: [u8; 16] = hex("feffe9928665731c6d6a8f9467308308").try_into().unwrap();
        let nonce: [u8; 12] = hex("cafebabefacedbaddecaf888").try_into().unwrap();
        let plain = hex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39");
        let cipher = "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091";
        let gcm = Gcm::new(&key);
        let mut out = Vec::new();
        let aad = hex("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        gcm.seal(&nonce, &aad, &[&plain[..20], &plain[20..]], &mut out);
        assert_eq!(out, hex(&format!("{cipher}5bc94fbc3221a5db94fae95ae7121a47")));
        let mut back = Vec::new();
        assert!(gcm.open(&nonce, &aad, &out, &mut back));
        assert_eq!(back, plain);
        out[3] ^= 1;
        assert!(!gcm.open(&nonce, &aad, &out, &mut back) && back == plain, "a changed byte is refused");
        let long: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let (mut sealed, mut opened) = (Vec::new(), Vec::new());
        gcm.seal(&nonce, b"x", &[&long[..333], &long[333..]], &mut sealed);
        // The digest of Python `cryptography`'s AESGCM output for the same message.
        let want = hex("24e1bb2637bb81dce09b26c74af87c534e6b5308fa2b5144187b7a98d1fa7fdd");
        assert_eq!(crate::net::crypto::Sha256::digest(&sealed).to_vec(), want);
        assert!(gcm.open(&nonce, b"x", &sealed, &mut opened) && opened == long);
    }
}
