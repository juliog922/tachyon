//! ECDSA signature checks over NIST P-256 with SHA-256 (FIPS 186-5): what a TLS client verifies. Ported from
//! caudal, without its signing half.
//!
//! Field and scalar arithmetic share one Montgomery multiplier over four 64-bit limbs. Points are projective and
//! added with the complete formulas of Renes, Costello and Batina (2016, algorithm 4, a = −3), so doubling and the
//! point at infinity need no special cases. Inputs are public, so nothing here needs to run in constant time.

// Curve constants are written as the standard prints them; the point formulas keep the paper's names.
#![allow(clippy::unreadable_literal, clippy::many_single_char_names, clippy::similar_names)]

type Limbs = [u64; 4];

struct Modulus {
    m: Limbs,
    /// −m⁻¹ mod 2⁶⁴.
    inv: u64,
    /// R² mod m, R = 2²⁵⁶.
    r2: Limbs,
}

const P: Modulus = Modulus {
    m: [0xffffffffffffffff, 0x00000000ffffffff, 0x0000000000000000, 0xffffffff00000001],
    inv: 1,
    r2: [0x0000000000000003, 0xfffffffbffffffff, 0xfffffffffffffffe, 0x00000004fffffffd],
};
const N: Modulus = Modulus {
    m: [0xf3b9cac2fc632551, 0xbce6faada7179e84, 0xffffffffffffffff, 0xffffffff00000000],
    inv: 0xccd1c8aaee00bc4f,
    r2: [0x83244c95be79eea2, 0x4699799c49bd6fa6, 0x2845b2392b6bec59, 0x66e12d94f3d95620],
};
/// R mod p, b·R mod p, and the base point, in Montgomery form.
const ONE: Limbs = [0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe];
const B: Limbs = [0xd89cdf6229c4bddf, 0xacf005cd78843090, 0xe5a220abf7212ed6, 0xdc30061d04874834];
const G: Point = Point {
    x: [0x79e730d418a9143c, 0x75ba95fc5fedb601, 0x79fb732b77622510, 0x18905f76a53755c6],
    y: [0xddf25357ce95560a, 0x8b4ab8e4ba19e45c, 0xd2e88688dd21f325, 0x8571ff1825885d85],
    z: ONE,
};

fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = u128::from(a) + u128::from(b) + u128::from(carry);
    (t as u64, (t >> 64) as u64)
}

fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let t = u128::from(a).wrapping_sub(u128::from(b) + u128::from(borrow));
    (t as u64, (t >> 127) as u64)
}

/// `a − b` and the final borrow.
fn sub_raw(a: &Limbs, b: &Limbs) -> (Limbs, u64) {
    let mut out = [0; 4];
    let mut borrow = 0;
    for i in 0..4 {
        (out[i], borrow) = sbb(a[i], b[i], borrow);
    }
    (out, borrow)
}

/// `a` where `mask` is all ones, else `b`.
fn select(mask: u64, a: &Limbs, b: &Limbs) -> Limbs {
    std::array::from_fn(|i| (a[i] & mask) | (b[i] & !mask))
}

fn is_zero(a: &Limbs) -> u64 {
    let any = a.iter().fold(0, |acc, x| acc | x);
    ((any | any.wrapping_neg()) >> 63) ^ 1
}

impl Modulus {
    /// `a·b·R⁻¹ mod m` for `a, b < m` (CIOS).
    fn mul(&self, a: &Limbs, b: &Limbs) -> Limbs {
        debug_assert!(self.holds(a) && self.holds(b), "Montgomery inputs are reduced");
        let mut t = [0u64; 6];
        for &bi in b {
            let mut c = 0;
            for j in 0..4 {
                let v = u128::from(t[j]) + u128::from(a[j]) * u128::from(bi) + u128::from(c);
                (t[j], c) = (v as u64, (v >> 64) as u64);
            }
            (t[4], t[5]) = adc(t[4], c, 0);
            let q = t[0].wrapping_mul(self.inv);
            let v = u128::from(t[0]) + u128::from(q) * u128::from(self.m[0]);
            let mut c = (v >> 64) as u64;
            for j in 1..4 {
                let v = u128::from(t[j]) + u128::from(q) * u128::from(self.m[j]) + u128::from(c);
                (t[j - 1], c) = (v as u64, (v >> 64) as u64);
            }
            let (s, carry) = adc(t[4], c, 0);
            (t[3], t[4]) = (s, t[5] + carry);
        }
        self.reduce(&[t[0], t[1], t[2], t[3]], t[4])
    }

    /// `value + carry·2²⁵⁶` reduced once by m.
    fn reduce(&self, value: &Limbs, carry: u64) -> Limbs {
        let (less, borrow) = sub_raw(value, &self.m);
        let keep = 0u64.wrapping_sub(borrow & (carry ^ 1));
        select(keep, value, &less)
    }

    fn add(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let mut sum = [0; 4];
        let mut carry = 0;
        for i in 0..4 {
            (sum[i], carry) = adc(a[i], b[i], carry);
        }
        self.reduce(&sum, carry)
    }

    fn sub(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let (d, borrow) = sub_raw(a, b);
        let back = select(0u64.wrapping_sub(borrow), &self.m, &[0; 4]);
        let mut out = [0; 4];
        let mut carry = 0;
        for i in 0..4 {
            (out[i], carry) = adc(d[i], back[i], carry);
        }
        out
    }

    /// `a·R mod m`: into Montgomery form.
    fn encode(&self, a: &Limbs) -> Limbs {
        self.mul(a, &self.r2)
    }

    /// `a·R⁻¹ mod m`: out of Montgomery form.
    fn decode(&self, a: &Limbs) -> Limbs {
        self.mul(a, &[1, 0, 0, 0])
    }

    /// `a⁻¹` in Montgomery form, as `a^(m−2)`: the exponent is public, so the
    /// branches on its bits leak nothing about `a`.
    fn invert(&self, a: &Limbs) -> Limbs {
        let (e, _) = sub_raw(&self.m, &[2, 0, 0, 0]);
        let mut r = self.encode(&[1, 0, 0, 0]);
        for bit in (0..256).rev() {
            r = self.mul(&r, &r);
            if e[bit / 64] >> (bit % 64) & 1 == 1 {
                r = self.mul(&r, a);
            }
        }
        r
    }

    /// Whether `a < m`.
    fn holds(&self, a: &Limbs) -> bool {
        sub_raw(a, &self.m).1 == 1
    }
}

fn limbs(b: &[u8]) -> Limbs {
    std::array::from_fn(|i| b[24 - 8 * i..32 - 8 * i].iter().fold(0, |v, &x| v << 8 | u64::from(x)))
}

#[derive(Clone, Copy)]
struct Point {
    x: Limbs,
    y: Limbs,
    z: Limbs,
}

const INFINITY: Point = Point { x: [0; 4], y: ONE, z: [0; 4] };

fn add(p: &Point, q: &Point) -> Point {
    let (mul, add, sub) = (|a: &Limbs, b: &Limbs| P.mul(a, b), |a: &Limbs, b: &Limbs| P.add(a, b), |a: &Limbs, b: &Limbs| P.sub(a, b));
    let t0 = mul(&p.x, &q.x);
    let t1 = mul(&p.y, &q.y);
    let t2 = mul(&p.z, &q.z);
    let t3 = mul(&add(&p.x, &p.y), &add(&q.x, &q.y));
    let t3 = sub(&t3, &add(&t0, &t1));
    let t4 = mul(&add(&p.y, &p.z), &add(&q.y, &q.z));
    let t4 = sub(&t4, &add(&t1, &t2));
    let x3 = mul(&add(&p.x, &p.z), &add(&q.x, &q.z));
    let y3 = sub(&x3, &add(&t0, &t2));
    let x3 = sub(&y3, &mul(&B, &t2));
    let x3 = add(&x3, &add(&x3, &x3));
    let z3 = sub(&t1, &x3);
    let x3 = add(&t1, &x3);
    let y3 = mul(&B, &y3);
    let t2 = add(&t2, &add(&t2, &t2));
    let y3 = sub(&sub(&y3, &t2), &t0);
    let y3 = add(&y3, &add(&y3, &y3));
    let t0 = sub(&add(&t0, &add(&t0, &t0)), &t2);
    let t1 = mul(&t4, &y3);
    let t2 = mul(&t0, &y3);
    let y3 = add(&mul(&x3, &z3), &t2);
    let x3 = sub(&mul(&t3, &x3), &t1);
    let z3 = add(&mul(&t4, &z3), &mul(&t3, &t0));
    Point { x: x3, y: y3, z: z3 }
}

/// The 4-bit digit of `k` at `window` (0 is the lowest).
fn digit(k: &Limbs, window: usize) -> u64 {
    k[window / 16] >> (window % 16 * 4) & 15
}

/// `k·p` for a scalar below n: four doublings and one addition per 4-bit digit.
fn multiply(k: &Limbs, p: &Point) -> Point {
    let mut table = [INFINITY; 16];
    for i in 1..16 {
        table[i] = add(&table[i - 1], p);
    }
    let mut acc = INFINITY;
    for window in (0..64).rev() {
        for _ in 0..4 {
            acc = add(&acc, &acc);
        }
        acc = add(&acc, &table[digit(k, window) as usize]);
    }
    acc
}

/// Affine coordinates out of Montgomery form; `None` at infinity.
fn affine(p: &Point) -> Option<(Limbs, Limbs)> {
    if is_zero(&p.z) == 1 {
        return None;
    }
    let zi = P.invert(&p.z);
    Some((P.decode(&P.mul(&p.x, &zi)), P.decode(&P.mul(&p.y, &zi))))
}

/// A public key: a point on the curve other than infinity.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PublicKey {
    x: Limbs,
    y: Limbs,
}

impl PublicKey {
    /// A key from its uncompressed SEC 1 encoding (`04 ‖ x ‖ y`), checked to lie on the curve.
    pub fn from_sec1(b: &[u8]) -> Option<PublicKey> {
        let [4, rest @ ..] = b else { return None };
        if rest.len() != 64 {
            return None;
        }
        let (x, y) = (limbs(&rest[..32]), limbs(&rest[32..]));
        if !P.holds(&x) || !P.holds(&y) {
            return None;
        }
        let (xm, ym) = (P.encode(&x), P.encode(&y));
        let three_x = P.add(&xm, &P.add(&xm, &xm));
        let rhs = P.add(&P.sub(&P.mul(&P.mul(&xm, &xm), &xm), &three_x), &B);
        (P.mul(&ym, &ym) == rhs).then_some(PublicKey { x, y })
    }

    /// Whether `sig` (r ‖ s) signs the SHA-256 `digest`. Inputs are public, so this need not be constant time.
    pub fn verify(&self, digest: &[u8; 32], sig: &[u8; 64]) -> bool {
        let (r, s) = (limbs(&sig[..32]), limbs(&sig[32..]));
        if !N.holds(&r) || !N.holds(&s) || is_zero(&r) == 1 || is_zero(&s) == 1 {
            return false;
        }
        let z = N.encode(&N.reduce(&limbs(digest), 0));
        let w = N.invert(&N.encode(&s));
        let u1 = N.decode(&N.mul(&z, &w));
        let u2 = N.decode(&N.mul(&N.encode(&r), &w));
        let q = Point { x: P.encode(&self.x), y: P.encode(&self.y), z: ONE };
        match affine(&add(&multiply(&u1, &G), &multiply(&u2, &q))) {
            Some((x, _)) => N.reduce(&x, 0) == r,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::crypto::{Sha256, tests::hex};

    /// A signature made by OpenSSL (`openssl dgst -sha256 -sign` with a P-256 key) checks, and fails once the
    /// message or the signature changes.
    #[test]
    fn p256_checks_an_openssl_signature() {
        let key = PublicKey::from_sec1(&hex(P256_KEY)).expect("a point on the curve");
        let sig: [u8; 64] = hex(P256_SIG).try_into().unwrap();
        assert!(key.verify(&Sha256::digest(b"tachyon"), &sig));
        assert!(!key.verify(&Sha256::digest(b"tachyoN"), &sig));
        let mut bad = sig;
        bad[40] ^= 1;
        assert!(!key.verify(&Sha256::digest(b"tachyon"), &bad));
        let mut off = hex(P256_KEY);
        off[64] ^= 1;
        assert!(PublicKey::from_sec1(&off).is_none(), "a point off the curve is refused");
    }

    const P256_KEY: &str = "040404a0d826266ebfd63049699541d9f639d6c170bfeef7ba3fe288027c011f6008bc2d6a2c42fc260e13ff447e21a18fcea2cd8b6f2365d39f3915e4d9e62481";
    const P256_SIG: &str = "e12a225f471f78d3b1e775e282f524ce697c2c17d080670addf227a2e3bca9b803f17f094bffdff25a67ee9f4c02d4af615ac8d8e556793f819361b04b6edf2c";
}
