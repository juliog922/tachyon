//! X25519 Diffie-Hellman (RFC 7748). Ported from caudal.
//!
//! Field elements of GF(2²⁵⁵ − 19) are five 51-bit limbs multiplied through
//! `u128`. The Montgomery ladder runs the same instructions for every scalar
//! bit and swaps with masks, so its timing does not depend on the secret.

// Field constants are written as the reference implementation prints them; the ladder keeps the RFC's names.
#![allow(clippy::unreadable_literal, clippy::many_single_char_names, clippy::similar_names)]

use super::crypto::wipe;

type Fe = [u64; 5];

const MASK: u64 = (1 << 51) - 1;
/// The u-coordinate of the base point.
const BASE: [u8; 32] = {
    let mut b = [0; 32];
    b[0] = 9;
    b
};

fn decode(b: &[u8; 32]) -> Fe {
    let word = |i: usize| u64::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7]]);
    [word(0) & MASK, (word(6) >> 3) & MASK, (word(12) >> 6) & MASK, (word(19) >> 1) & MASK, (word(24) >> 12) & MASK]
}

fn carry(mut h: Fe) -> Fe {
    for i in 0..4 {
        h[i + 1] += h[i] >> 51;
        h[i] &= MASK;
    }
    h[0] += 19 * (h[4] >> 51);
    h[4] &= MASK;
    h
}

fn encode(f: &Fe) -> [u8; 32] {
    let mut h = carry(carry(*f));
    let mut q = (h[0] + 19) >> 51;
    for limb in &h[1..] {
        q = (limb + q) >> 51;
    }
    h[0] += 19 * q;
    for i in 0..4 {
        h[i + 1] += h[i] >> 51;
        h[i] &= MASK;
    }
    h[4] &= MASK;
    let words = [h[0] | h[1] << 51, h[1] >> 13 | h[2] << 38, h[2] >> 26 | h[3] << 25, h[3] >> 39 | h[4] << 12];
    let mut out = [0u8; 32];
    for (o, w) in out.chunks_exact_mut(8).zip(words) {
        o.copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn add(a: &Fe, b: &Fe) -> Fe {
    std::array::from_fn(|i| a[i] + b[i])
}

/// `a − b` for `b` with limbs below 2⁵², kept positive by adding 2p.
fn sub(a: &Fe, b: &Fe) -> Fe {
    const TWO_P: Fe = [0xfffffffffffda, 0xffffffffffffe, 0xffffffffffffe, 0xffffffffffffe, 0xffffffffffffe];
    carry(std::array::from_fn(|i| a[i] + TWO_P[i] - b[i]))
}

fn mul(a: &Fe, b: &Fe) -> Fe {
    let m = |x: u64, y: u64| u128::from(x) * u128::from(y);
    let [b1, b2, b3, b4] = [b[1] * 19, b[2] * 19, b[3] * 19, b[4] * 19];
    let r0 = m(a[0], b[0]) + m(a[1], b4) + m(a[2], b3) + m(a[3], b2) + m(a[4], b1);
    let r1 = m(a[0], b[1]) + m(a[1], b[0]) + m(a[2], b4) + m(a[3], b3) + m(a[4], b2);
    let r2 = m(a[0], b[2]) + m(a[1], b[1]) + m(a[2], b[0]) + m(a[3], b4) + m(a[4], b3);
    let r3 = m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0]) + m(a[4], b4);
    let r4 = m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0]);
    let r1 = r1 + (r0 >> 51);
    let r2 = r2 + (r1 >> 51);
    let r3 = r3 + (r2 >> 51);
    let r4 = r4 + (r3 >> 51);
    let h0 = u128::from(r0 as u64 & MASK) + (r4 >> 51) * 19;
    [h0 as u64 & MASK, (r1 as u64 & MASK) + (h0 >> 51) as u64, r2 as u64 & MASK, r3 as u64 & MASK, r4 as u64 & MASK]
}

fn square(a: &Fe, times: usize) -> Fe {
    (0..times).fold(*a, |x, _| mul(&x, &x))
}

/// `z^(p−2)`, the inverse, by the addition chain of the reference implementation.
fn invert(z: &Fe) -> Fe {
    let z2 = square(z, 1);
    let z9 = mul(&square(&z2, 2), z);
    let z11 = mul(&z9, &z2);
    let z_5 = mul(&square(&z11, 1), &z9);
    let z_10 = mul(&square(&z_5, 5), &z_5);
    let z_20 = mul(&square(&z_10, 10), &z_10);
    let z_40 = mul(&square(&z_20, 20), &z_20);
    let z_50 = mul(&square(&z_40, 10), &z_10);
    let z_100 = mul(&square(&z_50, 50), &z_50);
    let z_200 = mul(&square(&z_100, 100), &z_100);
    let z_250 = mul(&square(&z_200, 50), &z_50);
    mul(&square(&z_250, 5), &z11)
}

fn swap(choice: u64, a: &mut Fe, b: &mut Fe) {
    let mask = 0u64.wrapping_sub(choice);
    for i in 0..5 {
        let t = mask & (a[i] ^ b[i]);
        a[i] ^= t;
        b[i] ^= t;
    }
}

/// The shared secret of `scalar` and the peer's `point`; all zeros when the
/// point has small order, which callers must reject.
pub fn x25519(scalar: &[u8; 32], point: &[u8; 32]) -> [u8; 32] {
    let mut k = *scalar;
    k[0] &= 0xf8;
    k[31] = (k[31] & 127) | 64;
    let u = decode(point);
    let (mut x2, mut z2, mut x3, mut z3) = ([1, 0, 0, 0, 0], [0; 5], u, [1, 0, 0, 0, 0]);
    let mut swapped = 0;
    for t in (0..255).rev() {
        let bit = u64::from(k[t / 8] >> (t % 8)) & 1;
        swapped ^= bit;
        swap(swapped, &mut x2, &mut x3);
        swap(swapped, &mut z2, &mut z3);
        swapped = bit;
        let a = add(&x2, &z2);
        let aa = mul(&a, &a);
        let b = sub(&x2, &z2);
        let bb = mul(&b, &b);
        let e = sub(&aa, &bb);
        let c = add(&x3, &z3);
        let d = sub(&x3, &z3);
        let da = mul(&d, &a);
        let cb = mul(&c, &b);
        let sum = add(&da, &cb);
        x3 = mul(&sum, &sum);
        let diff = sub(&da, &cb);
        z3 = mul(&u, &mul(&diff, &diff));
        x2 = mul(&aa, &bb);
        z2 = mul(&e, &add(&aa, &mul(&[121665, 0, 0, 0, 0], &e)));
    }
    swap(swapped, &mut x2, &mut x3);
    swap(swapped, &mut z2, &mut z3);
    wipe(&mut k);
    let shared = encode(&mul(&x2, &invert(&z2)));
    for fe in [&mut x2, &mut z2, &mut x3, &mut z3] {
        // SAFETY: five `u64`s are 40 initialized bytes.
        wipe(unsafe { std::slice::from_raw_parts_mut(fe.as_mut_ptr().cast::<u8>(), 40) });
    }
    shared
}

/// The public key of `secret`.
pub fn public(secret: &[u8; 32]) -> [u8; 32] {
    x25519(secret, &BASE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::crypto::tests::hex;

    /// RFC 7748 §5.2 and §6.1.
    #[test]
    fn x25519_matches_rfc_7748() {
        let arr = |s: &str| <[u8; 32]>::try_from(hex(s)).unwrap();
        let scalar = arr("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        let point = arr("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        assert_eq!(x25519(&scalar, &point), arr("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"));
        let alice = arr("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let bob = arr("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        assert_eq!(public(&alice), arr("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"));
        let shared = arr("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
        assert_eq!((x25519(&alice, &public(&bob)), x25519(&bob, &public(&alice))), (shared, shared));
    }
}
