//! RSA signature checks (RFC 8017) with SHA-256: PKCS #1 v1.5, which certificate chains use, and PSS, which TLS 1.3
//! servers sign their handshake with. Keys of 1024 to 8192 bits. Inputs are public, so nothing here needs to run in
//! constant time: `sᵉ mod n` by square-and-multiply on Montgomery products over 64-bit limbs.

// The arithmetic keeps the one-letter names of RFC 8017 and of Montgomery's method.
#![allow(clippy::many_single_char_names)]

use super::crypto::Sha256;

/// The `DigestInfo` that precedes a SHA-256 digest in a PKCS #1 v1.5 signature.
const SHA256_INFO: [u8; 19] = [0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20];

/// An RSA public key: the modulus as little-endian limbs, its length in bytes, and the public exponent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    n: Vec<u64>,
    len: usize,
    e: u64,
}

impl PublicKey {
    /// The key of big-endian modulus `n` and exponent `e`; `None` for an even, short or long modulus, or an
    /// exponent that is not odd or does not fit 64 bits.
    pub fn new(n: &[u8], e: &[u8]) -> Option<PublicKey> {
        let n = &n[n.iter().position(|&b| b != 0)?..];
        let e = &e[e.iter().position(|&b| b != 0)?..];
        if !(128..=1024).contains(&n.len()) || e.len() > 8 || n[n.len() - 1] & 1 == 0 || e[e.len() - 1] & 1 == 0 {
            return None;
        }
        Some(PublicKey { n: limbs(n, n.len().div_ceil(8)), len: n.len(), e: e.iter().fold(0, |v, &b| v << 8 | u64::from(b)) })
    }

    /// Whether `sig` is a PKCS #1 v1.5 signature of the SHA-256 `digest`.
    pub fn verify_pkcs1(&self, digest: &[u8; 32], sig: &[u8]) -> bool {
        let Some(em) = self.open(sig) else { return false };
        let pad = self.len - 3 - SHA256_INFO.len() - 32;
        let mut want = vec![0, 1];
        want.extend(std::iter::repeat_n(0xff, pad));
        want.push(0);
        want.extend_from_slice(&SHA256_INFO);
        want.extend_from_slice(digest);
        em == want
    }

    /// Whether `sig` is a PSS signature of the SHA-256 `digest` with MGF1-SHA-256 and a 32-byte salt, as TLS 1.3's
    /// `rsa_pss_rsae_sha256` requires.
    pub fn verify_pss(&self, digest: &[u8; 32], sig: &[u8]) -> bool {
        let Some(full) = self.open(sig) else { return false };
        let bits = self.bits() - 1;
        let (lead, em) = full.split_at(self.len - bits.div_ceil(8));
        let top = 0xff >> (8 * em.len() - bits);
        if lead.iter().any(|&b| b != 0) || em.len() < 66 || em[em.len() - 1] != 0xbc || em[0] & !top != 0 {
            return false;
        }
        let (masked, h) = em[..em.len() - 1].split_at(em.len() - 33);
        let mut db = mgf1(h, masked.len());
        db.iter_mut().zip(masked).for_each(|(d, m)| *d ^= m);
        db[0] &= top;
        let (ps, rest) = db.split_at(db.len() - 33);
        let mut m = Sha256::default();
        m.update(&[0; 8]);
        m.update(digest);
        m.update(&rest[1..]);
        ps.iter().all(|&b| b == 0) && rest[0] == 1 && m.finish() == h
    }

    /// Bits of the modulus.
    fn bits(&self) -> usize {
        64 * self.n.len() - self.n[self.n.len() - 1].leading_zeros() as usize
    }

    /// `sigᵉ mod n` as `len` big-endian bytes; `None` unless the signature is `len` bytes below `n`.
    fn open(&self, sig: &[u8]) -> Option<Vec<u8>> {
        let l = self.n.len();
        let s = limbs(sig, l);
        if sig.len() != self.len || !less(&s, &self.n) {
            return None;
        }
        let inv = (0..6).fold(1u64, |x, _| x.wrapping_mul(2u64.wrapping_sub(self.n[0].wrapping_mul(x)))).wrapping_neg();
        let mul = |a: &[u64], b: &[u64]| montgomery(a, b, &self.n, inv);
        // R² mod n by doubling 1 a further 2·64·l times.
        let r2 = (0..128 * l).fold(limbs(&[1], l), |x, _| double(&x, &self.n));
        let base = mul(&s, &r2);
        let mut acc = base.clone();
        for bit in (0..self.e.ilog2()).rev() {
            acc = mul(&acc, &acc);
            if self.e >> bit & 1 == 1 {
                acc = mul(&acc, &base);
            }
        }
        let out = mul(&acc, &limbs(&[1], l));
        Some(out.iter().rev().flat_map(|w| w.to_be_bytes()).skip(8 * l - self.len).collect())
    }
}

/// The big-endian `bytes` as `l` little-endian limbs.
fn limbs(bytes: &[u8], l: usize) -> Vec<u64> {
    let mut out = vec![0; l.max(bytes.len().div_ceil(8))];
    for (i, &b) in bytes.iter().rev().enumerate() {
        out[i / 8] |= u64::from(b) << (8 * (i % 8));
    }
    out.truncate(l);
    out
}

/// Whether `a < b`, both of the same length.
fn less(a: &[u64], b: &[u64]) -> bool {
    a.iter().rev().cmp(b.iter().rev()) == std::cmp::Ordering::Less
}

/// `a − b` in place, returning the borrow.
fn subtract(a: &mut [u64], b: &[u64]) -> bool {
    let mut borrow = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (d, b1) = x.overflowing_sub(y);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        (*x, borrow) = (d, b1 || b2);
    }
    borrow
}

/// `2x mod n` for `x < n`.
fn double(x: &[u64], n: &[u64]) -> Vec<u64> {
    let mut out: Vec<u64> = x.iter().scan(0, |carry, &w| Some(std::mem::replace(carry, w >> 63) | w << 1)).collect();
    let top = x[x.len() - 1] >> 63 == 1;
    if top || !less(&out, n) {
        subtract(&mut out, n);
    }
    out
}

/// `a·b·R⁻¹ mod n` (CIOS), `inv = −n⁻¹ mod 2⁶⁴`.
fn montgomery(a: &[u64], b: &[u64], n: &[u64], inv: u64) -> Vec<u64> {
    let l = n.len();
    let mut t = vec![0u64; l + 2];
    for &bi in b {
        let mut c = 0u128;
        for j in 0..l {
            c += u128::from(t[j]) + u128::from(a[j]) * u128::from(bi);
            (t[j], c) = (c as u64, c >> 64);
        }
        c += u128::from(t[l]);
        (t[l], t[l + 1]) = (c as u64, (c >> 64) as u64);
        let q = t[0].wrapping_mul(inv);
        let mut c = (u128::from(t[0]) + u128::from(q) * u128::from(n[0])) >> 64;
        for j in 1..l {
            c += u128::from(t[j]) + u128::from(q) * u128::from(n[j]);
            (t[j - 1], c) = (c as u64, c >> 64);
        }
        c += u128::from(t[l]);
        (t[l - 1], t[l]) = (c as u64, t[l + 1] + (c >> 64) as u64);
    }
    let carry = t[l] != 0;
    t.truncate(l);
    if carry || !less(&t, n) {
        subtract(&mut t, n);
    }
    t
}

/// MGF1 with SHA-256: `len` bytes from `seed`.
fn mgf1(seed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 32);
    for counter in 0u32.. {
        if out.len() >= len {
            break;
        }
        let mut h = Sha256::default();
        h.update(seed);
        h.update(&counter.to_be_bytes());
        out.extend_from_slice(&h.finish());
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::crypto::tests::hex;

    /// Signatures made by OpenSSL with a 2048-bit key (`openssl dgst -sha256 -sign`, and with
    /// `-sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:32`) check, each only in its own scheme, and fail
    /// once the message or the signature changes.
    #[test]
    fn rsa_checks_openssl_signatures() {
        let key = PublicKey::new(&hex(RSA_N), &[1, 0, 1]).unwrap();
        let (pkcs1, pss, digest) = (hex(RSA_PKCS1), hex(RSA_PSS), Sha256::digest(b"tachyon"));
        assert!(key.verify_pkcs1(&digest, &pkcs1) && key.verify_pss(&digest, &pss));
        assert!(!key.verify_pkcs1(&digest, &pss) && !key.verify_pss(&digest, &pkcs1), "each scheme refuses the other");
        let other = Sha256::digest(b"tachyoN");
        assert!(!key.verify_pkcs1(&other, &pkcs1) && !key.verify_pss(&other, &pss));
        let mut bad = pss.clone();
        bad[100] ^= 4;
        assert!(!key.verify_pss(&digest, &bad));
        assert!(PublicKey::new(&hex(RSA_N), &[2]).is_none(), "an even exponent is refused");
    }

    const RSA_N: &str = "d5cfa08842fc6fc3a44b3051ee5aa36b318a2488666637d4c859f0a5c54c0e1e2fca3cb9d87a7a803baca47ea6dd22a8b41a821a0af8fce7d7eb75060640a1252ab5b02f3640a38d21f84d800302ceb3d163145e4a9b8394c23b71aa4c6adfa01acd74ef4f4dbe63caa8d8aed36aef1e3bffaf864f628e87de4ea96f0e7f2d21ae834de096ed54ffd6bba632e4601358d6740bf59a01034dbb1da7d41c868931a21c6ce6bea8a2e8f2c9220dfb829f43adbbd3f0b5966ce210f33541ca5de876c6c206430b18765929b8b666886252456efae9a42432855936b7e04aa03a31915eee005e03bb9d03f6c8969e593e131f7c78dfe1f168db82ce95fe951166d15d";
    const RSA_PKCS1: &str = "3b7a906d37705447a8a076dfce0fa31ef3855674357e01e26324398f8f96c03bd8dcbcdc69b498e709e141ab534ae38888d6fbe940b6a576f7e467ce0db8261a25dce80e316af37dff17743cb45add02cb43b84f64b1fdb08f2b430c33d2e5001d1854c906e87c70e2773d1437e8559db3d0ac7076ca6c5ca00f7d2b83cac81d2642db2327b929a90d42905654d220391c130e5d1abcf3871c761264ede1ab18e994d3b2dbfe2e4cb1e3ff3a2eb9d554925025c54149fde2232636d753b9c9c85b3e4d7c0474cec834cb9c5060639844d74b288868da84cfd649f23d049e9080c603b2765267ba9f38832b973d8b860e0ffabd70fae7bd367622b6c9e5513a2e";
    const RSA_PSS: &str = "a6287dd44cc2c116a2b95dd6d074e350949ca7c366e9932ffb93d7e617df4e3e0d48932e29d48ea772de29249401c1390d236ba9bb6d850431ccf6a7109c5cff7f29d706344062350dee43e2ee5eb0456c65da2fbd3e292fe72ed050c302befcdc7e7efb352e55fd2c9a774aae1aa4967d25231d73ce73ba3e3453a8b23d2e81253d0dd8262869abf44c73960e9c320a0acd7dd213369ca84aefff4571e3cfaceddaa9e5e43b6f4cb610d31ad82cbfce7100f99ae691c415d97092ab8bf7a4977d82d8e7e06dc2b066316c93e3d7b5650883daad262878f086cd2ef718c9da6c5fd18f8aaeba1549c47347d487f8fa0f2c5e174a850abee3847a92742afb6e5f";
}
