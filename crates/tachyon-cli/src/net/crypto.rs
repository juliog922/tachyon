//! Hashing and key derivation for downloads and TLS: SHA-256 (with the SHA-NI instructions when the CPU has them),
//! HMAC (RFC 2104), HKDF (RFC 5869), and what every primitive shares: random bytes, constant-time comparison, and
//! wiping secrets before memory is reused. Ported from caudal.

// Round constants are written as the standard prints them; the round function keeps its one-letter names.
#![allow(clippy::unreadable_literal, clippy::many_single_char_names)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32,
    _mm_sha256rnds2_epu32, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
};
use std::io::Read as _;

/// Fills `out` from the kernel's random number generator.
///
/// # Panics
///
/// If `/dev/urandom` cannot be read: nothing secure can go on without it.
pub fn random(out: &mut [u8]) {
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(out)).expect("the kernel's random number generator");
}

/// Overwrites `b` with zeros in a way the compiler cannot elide.
pub fn wipe(b: &mut [u8]) {
    for x in b.iter_mut() {
        // SAFETY: `x` is a valid, aligned reference into `b`.
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Equality in time that depends only on the lengths.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74,
    0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d,
    0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e,
    0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (FIPS 180-4). Cloning snapshots a running hash, as TLS transcripts need.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Sha256 {
        Sha256 { state: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], block: [0; 64], filled: 0, total: 0 }
    }
}

impl Drop for Sha256 {
    fn drop(&mut self) {
        wipe(&mut self.block);
        for w in &mut self.state {
            // SAFETY: `w` is a valid, aligned reference into the state.
            unsafe { std::ptr::write_volatile(w, 0) };
        }
    }
}

impl Sha256 {
    /// The digest of `data`.
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Sha256::default();
        h.update(data);
        h.finish()
    }

    /// Hashes `data` after what came before.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            let block = self.block;
            compress(&mut self.state, &block);
            self.filled = 0;
        }
        let whole = data.len() / 64 * 64;
        compress(&mut self.state, &data[..whole]);
        self.block[..data.len() - whole].copy_from_slice(&data[whole..]);
        self.filled = data.len() - whole;
    }

    /// The digest of everything so far; the running state is left as it was.
    pub fn snapshot(&self) -> [u8; 32] {
        self.clone().finish()
    }

    /// The digest of everything hashed.
    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.total * 8;
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        self.update(&pad[..=(119 - self.filled) % 64]);
        self.update(&bits.to_be_bytes());
        let mut out = [0u8; 32];
        for (o, s) in out.chunks_exact_mut(4).zip(self.state) {
            o.copy_from_slice(&s.to_be_bytes());
        }
        out
    }
}

/// Whether this CPU has the SHA-NI instructions [`Sha256`] uses when it can.
pub fn sha_ni() -> bool {
    is_x86_feature_detected!("sha") && is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3")
}

/// Hashes the whole 64-byte blocks of `blocks` into `state`.
fn compress(state: &mut [u32; 8], blocks: &[u8]) {
    static NI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NI.get_or_init(sha_ni) {
        // SAFETY: the CPU has the instructions, checked above.
        unsafe { compress_ni(state, blocks) };
    } else {
        blocks.chunks_exact(64).for_each(|b| compress_scalar(state, b));
    }
}

fn compress_scalar(state: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for (i, c) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let t1 =
            h.wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)).wrapping_add((e & f) ^ (!e & g)).wrapping_add(K[i]).wrapping_add(w[i]);
        let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)).wrapping_add((a & b) ^ (a & c) ^ (b & c));
        (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

/// [`compress`] on the SHA-NI instructions: four rounds per `sha256rnds2` pair, the message schedule by
/// `sha256msg1`/`sha256msg2`, the state kept as the instructions want it (ABEF and CDGH).
#[target_feature(enable = "sha,sse4.1,ssse3")]
fn compress_ni(state: &mut [u32; 8], blocks: &[u8]) {
    let swap = _mm_set_epi64x(0x0c0d0e0f08090a0b, 0x0405060700010203);
    // SAFETY: `state` holds 32 bytes; unaligned loads and stores.
    let (lo, hi) = unsafe { (_mm_loadu_si128(state.as_ptr().cast()), _mm_loadu_si128(state.as_ptr().add(4).cast())) };
    let (dcba, hgfe) = (_mm_shuffle_epi32::<0xb1>(lo), _mm_shuffle_epi32::<0x1b>(hi));
    let (mut abef, mut cdgh) = (_mm_alignr_epi8::<8>(dcba, hgfe), _mm_blend_epi16::<0xf0>(hgfe, dcba));
    for block in blocks.chunks_exact(64) {
        let (abef0, cdgh0) = (abef, cdgh);
        // SAFETY: the block holds 64 bytes; unaligned loads.
        let mut m: [__m128i; 4] = std::array::from_fn(|i| _mm_shuffle_epi8(unsafe { _mm_loadu_si128(block.as_ptr().add(16 * i).cast()) }, swap));
        for r in 0..16 {
            // SAFETY: `K` holds 64 constants; four from 4·r.
            let k = unsafe { _mm_loadu_si128(K.as_ptr().add(4 * r).cast()) };
            let mk = _mm_add_epi32(m[r % 4], k);
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, mk);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32::<0x0e>(mk));
            if (3..15).contains(&r) {
                let next = _mm_add_epi32(m[(r + 1) % 4], _mm_alignr_epi8::<4>(m[r % 4], m[(r + 3) % 4]));
                m[(r + 1) % 4] = _mm_sha256msg2_epu32(next, m[r % 4]);
            }
            if (1..13).contains(&r) {
                m[(r + 3) % 4] = _mm_sha256msg1_epu32(m[(r + 3) % 4], m[r % 4]);
            }
        }
        (abef, cdgh) = (_mm_add_epi32(abef, abef0), _mm_add_epi32(cdgh, cdgh0));
    }
    let (feba, dchg) = (_mm_shuffle_epi32::<0x1b>(abef), _mm_shuffle_epi32::<0xb1>(cdgh));
    // SAFETY: `state` holds 32 bytes; unaligned stores.
    unsafe {
        _mm_storeu_si128(state.as_mut_ptr().cast(), _mm_blend_epi16::<0xf0>(feba, dchg));
        _mm_storeu_si128(state.as_mut_ptr().add(4).cast(), _mm_alignr_epi8::<8>(dchg, feba));
    }
}

/// HMAC-SHA-256 with the key's pads computed once.
#[derive(Clone)]
pub struct Hmac {
    inner: Sha256,
    outer: Sha256,
}

impl Hmac {
    /// A MAC under `key`.
    pub fn new(key: &[u8]) -> Hmac {
        let mut k = [0u8; 64];
        if key.len() > 64 {
            k[..32].copy_from_slice(&Sha256::digest(key));
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let (mut inner, mut outer) = (Sha256::default(), Sha256::default());
        let mut pad = k.map(|x| x ^ 0x36);
        inner.update(&pad);
        pad = k.map(|x| x ^ 0x5c);
        outer.update(&pad);
        wipe(&mut pad);
        wipe(&mut k);
        Hmac { inner, outer }
    }

    /// MACs `data` after what came before.
    pub fn update(&mut self, data: &[u8]) -> &mut Hmac {
        self.inner.update(data);
        self
    }

    /// The MAC of everything.
    pub fn finish(self) -> [u8; 32] {
        let Hmac { inner, mut outer } = self;
        let mut d = inner.finish();
        outer.update(&d);
        wipe(&mut d);
        outer.clone().finish()
    }

    /// The MAC of `data` under `key`.
    pub fn mac(key: &[u8], data: &[u8]) -> [u8; 32] {
        let mut h = Hmac::new(key);
        h.update(data);
        h.finish()
    }
}

/// HKDF-Extract: a pseudorandom key from input keying material.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    Hmac::mac(salt, ikm)
}

/// HKDF-Expand of `prk` with `info` into `out`.
///
/// # Panics
///
/// If `out` is longer than 255 × 32 bytes, the most HKDF-SHA256 expands to.
pub fn hkdf_expand(prk: &[u8], info: &[&[u8]], out: &mut [u8]) {
    assert!(out.len() <= 255 * 32);
    let mut t = [0u8; 32];
    for (i, chunk) in out.chunks_mut(32).enumerate() {
        let mut h = Hmac::new(prk);
        if i > 0 {
            h.update(&t);
        }
        for part in info {
            h.update(part);
        }
        h.update(&[i as u8 + 1]);
        t = h.finish();
        chunk.copy_from_slice(&t[..chunk.len()]);
    }
    wipe(&mut t);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn sha256_matches_fips_180_on_both_paths() {
        let million = vec![b'a'; 1_000_000];
        let cases: [(&[u8], &str); 3] = [
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
            (&million, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
        ];
        for (data, want) in cases {
            assert_eq!(Sha256::digest(data).to_vec(), hex(want));
            let mut state = Sha256::default().state;
            let mut scalar = state;
            let block = &million[..640];
            compress(&mut state, block);
            block.chunks_exact(64).for_each(|b| compress_scalar(&mut scalar, b));
            assert_eq!(state, scalar, "SHA-NI and scalar agree");
        }
        // Every split of a message gives the digest of the whole.
        let whole = Sha256::digest(&million[..300]);
        for cut in [0, 1, 63, 64, 65, 200] {
            let mut h = Sha256::default();
            h.update(&million[..cut]);
            h.update(&million[cut..300]);
            assert_eq!(h.finish(), whole);
        }
    }

    #[test]
    fn hmac_and_hkdf_match_rfc_4231_and_5869() {
        let mac = Hmac::mac(&[0x0b; 20], b"Hi There");
        assert_eq!(mac.to_vec(), hex("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"));
        let prk = hkdf_extract(&hex("000102030405060708090a0b0c"), &[0x0b; 22]);
        assert_eq!(prk.to_vec(), hex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"));
        let mut okm = [0u8; 42];
        hkdf_expand(&prk, &[&hex("f0f1f2f3f4f5f6f7f8f9")], &mut okm);
        assert_eq!(okm.to_vec(), hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"));
    }
}
