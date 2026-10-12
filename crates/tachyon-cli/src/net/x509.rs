//! X.509 for a TLS client: certificates in DER and PEM, the trust anchors of the system's bundle, and the checks a
//! client makes on a server's chain. Ported from caudal, reduced to verification, and extended to RSA.
//!
//! Keys are ECDSA P-256 or RSA; signatures ECDSA with SHA-256 or RSA PKCS #1 v1.5 with SHA-256. Certificates with
//! other keys or signatures parse, so a bundle loads whole, but cannot be used or verified.

use super::crypto::Sha256;
use super::{p256, rsa};

const OID_EC_PUBLIC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const OID_P256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
const OID_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
const OID_ECDSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
const OID_RSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
const OID_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
const OID_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];
const OID_BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
const OID_EXT_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x25];
const OID_SERVER_AUTH: &[u8] = &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01];
/// Extensions a client may meet marked critical and safely leave to the issuing CA: name constraints and
/// certificate policies, policy mappings and constraints, and the subject and authority key identifiers.
const UNDERSTOOD: [&[u8]; 6] = [&[0x55, 0x1d, 0x1e], &[0x55, 0x1d, 0x20], &[0x55, 0x1d, 0x21], &[0x55, 0x1d, 0x24], &[0x55, 0x1d, 0x0e], &[0x55, 0x1d, 0x23]];

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Every `CERTIFICATE` block of a PEM text, as DER; blocks that are not valid base64 are skipped.
pub fn pem_certificates(text: &str) -> Vec<Vec<u8>> {
    let (mut out, mut rest) = (Vec::new(), text);
    while let Some(start) = rest.find("-----BEGIN CERTIFICATE-----") {
        let body = &rest[start + 27..];
        let Some(end) = body.find("-----END CERTIFICATE-----") else { break };
        let (mut der, mut bits, mut n, mut good) = (Vec::with_capacity(end * 3 / 4), 0u32, 0u32, true);
        for c in body[..end].bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
            let Some(sextet) = ALPHABET.iter().position(|&a| a == c) else {
                good = false;
                break;
            };
            (n, bits) = (n << 6 | sextet as u32, bits + 6);
            if bits >= 8 {
                bits -= 8;
                der.push((n >> bits) as u8);
            }
        }
        if good {
            out.push(der);
        }
        rest = &body[end..];
    }
    out
}

/// A reader over consecutive DER elements.
struct Der<'a>(&'a [u8]);

impl<'a> Der<'a> {
    /// The next element as (tag, content, whole encoding).
    fn next(&mut self) -> Option<(u8, &'a [u8], &'a [u8])> {
        let b = self.0;
        let (&tag, &first) = (b.first()?, b.get(1)?);
        let (len, head) = match first {
            0..=0x7f => (first as usize, 2),
            0x81..=0x83 => {
                let n = (first & 0x7f) as usize;
                let len = b.get(2..2 + n)?.iter().fold(0usize, |l, &x| l << 8 | x as usize);
                if len < 0x80 || (n > 1 && b[2] == 0) {
                    return None;
                }
                (len, 2 + n)
            }
            _ => return None,
        };
        let whole = b.get(..head + len)?;
        self.0 = &b[head + len..];
        Some((tag, &whole[head..], whole))
    }

    fn expect(&mut self, tag: u8) -> Option<&'a [u8]> {
        self.next().filter(|e| e.0 == tag).map(|e| e.1)
    }

    /// The content of the next element if it has `tag`; otherwise nothing is consumed.
    fn optional(&mut self, tag: u8) -> Option<&'a [u8]> {
        (self.0.first() == Some(&tag)).then(|| self.expect(tag)).flatten()
    }

    fn done(&self) -> bool {
        self.0.is_empty()
    }
}

/// The r ‖ s of a DER `Ecdsa-Sig-Value`.
pub fn raw_signature(der: &[u8]) -> Option<[u8; 64]> {
    let mut outer = Der(der);
    let mut d = Der(outer.expect(0x30)?);
    let mut out = [0u8; 64];
    for half in out.chunks_exact_mut(32) {
        let int = unsigned(d.expect(0x02)?)?;
        half.get_mut(32usize.checked_sub(int.len())?..)?.copy_from_slice(int);
    }
    (d.done() && outer.done()).then_some(out)
}

/// The magnitude of a minimally encoded, non-negative DER INTEGER.
fn unsigned(int: &[u8]) -> Option<&[u8]> {
    match int {
        [0, rest @ ..] if rest.first().is_some_and(|b| b & 0x80 != 0) => Some(rest),
        [b, ..] if b & 0x80 == 0 && (int.len() == 1 || *b != 0) => Some(int),
        _ => None,
    }
}

/// The day number of a civil date (days since 1970-01-01).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// Seconds since the Unix epoch of a `UTCTime` (`0x17`) or `GeneralizedTime` (`0x18`).
fn decode_time(tag: u8, text: &[u8]) -> Option<i64> {
    let text = std::str::from_utf8(text).ok()?;
    let (year, rest) = match (tag, text.len()) {
        (0x17, 13) => {
            let yy: i64 = text.get(..2)?.parse().ok()?;
            (if yy < 50 { 2000 + yy } else { 1900 + yy }, &text[2..])
        }
        (0x18, 15) => (text.get(..4)?.parse().ok()?, &text[4..]),
        _ => return None,
    };
    if !rest.ends_with('Z') || !rest[..10].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let f: Vec<i64> = (0..5).map(|i| rest[2 * i..2 * i + 2].parse().unwrap_or(99)).collect();
    let valid = (1..=12).contains(&f[0]) && (1..=31).contains(&f[1]) && f[2] < 24 && f[3] < 60 && f[4] < 60;
    valid.then(|| days_from_civil(year, f[0], f[1]) * 86_400 + f[2] * 3600 + f[3] * 60 + f[4])
}

/// Seconds since the Unix epoch, from the system clock.
pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// A subject's public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    /// ECDSA on P-256.
    P256(p256::PublicKey),
    /// RSA.
    Rsa(rsa::PublicKey),
}

impl Key {
    /// Whether `sig` (an issuer's, as the certificate carries it) signs the SHA-256 `digest`.
    fn verifies(&self, digest: &[u8; 32], sig: &Signature) -> bool {
        match (self, sig) {
            (Key::P256(k), Signature::Ecdsa(s)) => k.verify(digest, s),
            (Key::Rsa(k), Signature::Rsa(s)) => k.verify_pkcs1(digest, s),
            _ => false,
        }
    }
}

/// An issuer's signature over a certificate: ECDSA (r ‖ s) or RSA PKCS #1 v1.5, both with SHA-256.
#[derive(Debug, Clone)]
enum Signature {
    Ecdsa([u8; 64]),
    Rsa(Vec<u8>),
}

/// A name a certificate holds: a DNS name (lowercase) or an IP address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Name {
    /// A DNS name, possibly a `*.` wildcard.
    Dns(String),
    /// The 4 or 16 bytes of an IP address.
    Ip(Vec<u8>),
}

/// Why a certificate chain is not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// Not valid DER, or not a certificate.
    Malformed,
    /// A key, signature or critical extension this client does not understand.
    Unsupported,
    /// Outside its validity period.
    Expired,
    /// No trust anchor issued it.
    UnknownIssuer,
    /// It does not name the host.
    WrongHost,
    /// A signature does not check.
    BadSignature,
}

/// What a certificate's extensions say: a CA, allowed to sign certificates, allowed for TLS servers, its names.
#[derive(Debug, Clone)]
struct Extensions {
    ca: bool,
    signs_certificates: bool,
    serves_tls: bool,
    names: Vec<Name>,
}

/// A parsed certificate. Ranges index into `der`.
#[derive(Debug, Clone)]
pub struct Certificate {
    /// The certificate's encoding.
    pub der: Vec<u8>,
    tbs: std::ops::Range<usize>,
    issuer: std::ops::Range<usize>,
    subject: std::ops::Range<usize>,
    not_before: i64,
    not_after: i64,
    /// The subject's key, when this client understands it.
    pub key: Option<Key>,
    signature: Option<Signature>,
    ext: Extensions,
}

impl Certificate {
    /// Parses `der`.
    pub fn parse(der: Vec<u8>) -> Result<Certificate, Rejected> {
        let cert = Certificate::read(&der).ok_or(Rejected::Malformed)??;
        Ok(Certificate { der, ..cert })
    }

    /// The certificate's fields, `der` left empty for [`Certificate::parse`] to fill; `Some(Err)` for a critical
    /// extension this client does not understand.
    fn read(der: &[u8]) -> Option<Result<Certificate, Rejected>> {
        let at = |part: &[u8]| {
            let start = part.as_ptr() as usize - der.as_ptr() as usize;
            start..start + part.len()
        };
        let (whole, content, signature) = envelope(der)?;
        let mut tbs = Der(content);
        let (issuer, [not_before, not_after], subject, key) = body(&mut tbs)?;
        tbs.optional(0x81);
        tbs.optional(0x82);
        let ext = tbs.optional(0xa3).map_or(Some(Ok(Extensions::default())), extensions)?;
        let (tbs_range, issuer, subject) = (at(whole), at(issuer), at(subject));
        tbs.done().then(|| ext.map(|ext| Certificate { der: Vec::new(), tbs: tbs_range, issuer, subject, not_before, not_after, key, signature, ext }))
    }

    fn subject(&self) -> &[u8] {
        &self.der[self.subject.clone()]
    }

    /// Whether it may issue certificates: a CA whose key usage allows signing them.
    fn issues(&self) -> bool {
        self.ext.ca && self.ext.signs_certificates
    }

    fn issued_by(&self, issuer: &Certificate) -> bool {
        let digest = Sha256::digest(&self.der[self.tbs.clone()]);
        self.der[self.issuer.clone()] == *issuer.subject()
            && match (&issuer.key, &self.signature) {
                (Some(key), Some(sig)) => key.verifies(&digest, sig),
                _ => false,
            }
    }

    /// Whether the certificate names `host`: an IP address against its IP names, a DNS name against its DNS names
    /// (a leading `*.` matching one label, never a whole registrable name such as `*.com`).
    pub fn names_host(&self, host: &str) -> bool {
        let host = host.strip_suffix('.').unwrap_or(host);
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            let octets = match ip {
                std::net::IpAddr::V4(v4) => v4.octets().to_vec(),
                std::net::IpAddr::V6(v6) => v6.octets().to_vec(),
            };
            return self.ext.names.contains(&Name::Ip(octets));
        }
        let host = host.to_ascii_lowercase();
        let wildcard = |base: &str| base.contains('.') && host.split_once('.').is_some_and(|(label, rest)| !label.is_empty() && rest == base);
        self.ext.names.iter().any(|n| matches!(n, Name::Dns(d) if *d == host || d.strip_prefix("*.").is_some_and(wildcard)))
    }

    /// Whether it is valid at `now`.
    fn current(&self, now: i64) -> bool {
        (self.not_before..=self.not_after).contains(&now)
    }
}

impl Default for Extensions {
    /// What a certificate without extensions allows: everything but issuing.
    fn default() -> Extensions {
        Extensions { ca: false, signs_certificates: true, serves_tls: true, names: Vec::new() }
    }
}

/// The certificate (`tbsCertificate`), its content, and the issuer's signature when this client understands it.
fn envelope(der: &[u8]) -> Option<(&[u8], &[u8], Option<Signature>)> {
    let mut outer = Der(der);
    let mut cert = Der(outer.expect(0x30)?);
    let (tag, content, whole) = cert.next()?;
    let (alg, bits) = (cert.expect(0x30)?, cert.expect(0x03)?);
    let signature = match (Der(alg).expect(0x06), bits) {
        (Some(OID_ECDSA_SHA256), [0, sig @ ..]) => Some(Signature::Ecdsa(raw_signature(sig)?)),
        (Some(OID_RSA_SHA256), [0, sig @ ..]) => Some(Signature::Rsa(sig.to_vec())),
        _ => None,
    };
    (tag == 0x30 && cert.done() && outer.done()).then_some((whole, content, signature))
}

/// The issuer, validity, subject and key of a certificate's content, read up to its optional fields.
/// What [`body`] reads: issuer, validity, subject, key.
type Body<'a> = (&'a [u8], [i64; 2], &'a [u8], Option<Key>);

fn body<'a>(tbs: &mut Der<'a>) -> Option<Body<'a>> {
    tbs.optional(0xa0);
    tbs.expect(0x02)?;
    tbs.expect(0x30)?;
    let (_, _, issuer) = tbs.next().filter(|e| e.0 == 0x30)?;
    let validity = validity(tbs.expect(0x30)?)?;
    let (_, _, subject) = tbs.next().filter(|e| e.0 == 0x30)?;
    Some((issuer, validity, subject, public_key(tbs.expect(0x30)?)?))
}

/// The first and last second of a `Validity`.
fn validity(der: &[u8]) -> Option<[i64; 2]> {
    let mut d = Der(der);
    let mut time = || d.next().and_then(|(t, v, _)| decode_time(t, v));
    Some([time()?, time()?])
}

/// The key of a `SubjectPublicKeyInfo`: `Some(None)` for a kind this client does not understand.
#[allow(clippy::option_option)] // Malformed (refused) is not unknown (kept, unusable).
fn public_key(spki: &[u8]) -> Option<Option<Key>> {
    let mut spki = Der(spki);
    let mut alg = Der(spki.expect(0x30)?);
    let (kind, bits) = (alg.expect(0x06)?, spki.expect(0x03)?);
    let [0, bits @ ..] = bits else { return None };
    match kind {
        OID_EC_PUBLIC_KEY if alg.expect(0x06) == Some(OID_P256) => p256::PublicKey::from_sec1(bits).map(|k| Some(Key::P256(k))),
        OID_RSA => rsa_key(bits).map(Some),
        _ => Some(None),
    }
}

/// An `RSAPublicKey`: modulus and exponent.
fn rsa_key(der: &[u8]) -> Option<Key> {
    let mut seq = Der(Der(der).expect(0x30)?);
    rsa::PublicKey::new(seq.expect(0x02)?, seq.expect(0x02)?).map(Key::Rsa)
}

/// The extensions of a certificate; `Some(Err)` for a critical one this client does not understand.
fn extensions(block: &[u8]) -> Option<Result<Extensions, Rejected>> {
    let mut ext = Extensions::default();
    let mut list = Der(Der(block).expect(0x30)?);
    while !list.done() {
        let mut one = Der(list.expect(0x30)?);
        let oid = one.expect(0x06)?;
        let critical = one.optional(0x01).is_some_and(|v| v == [0xff]);
        if !ext.apply(oid, one.expect(0x04)?)? && critical && !UNDERSTOOD.contains(&oid) {
            return Some(Err(Rejected::Unsupported));
        }
    }
    Some(Ok(ext))
}

impl Extensions {
    /// Takes in extension `oid` with `value`; whether it is one this client reads, `None` if it is malformed.
    fn apply(&mut self, oid: &[u8], value: &[u8]) -> Option<bool> {
        match oid {
            OID_BASIC_CONSTRAINTS => self.ca = Der(Der(value).expect(0x30)?).optional(0x01).is_some_and(|v| v == [0xff]),
            OID_ALT_NAME => self.names = names(value)?,
            OID_KEY_USAGE => self.signs_certificates = Der(value).expect(0x03).and_then(|bits| bits.get(1)).is_some_and(|b| b & 0x04 != 0),
            OID_EXT_KEY_USAGE => {
                let mut purposes = Der(Der(value).expect(0x30)?);
                self.serves_tls = false;
                while let Some(oid) = purposes.expect(0x06) {
                    self.serves_tls |= oid == OID_SERVER_AUTH;
                }
            }
            _ => return Some(false),
        }
        Some(true)
    }
}

/// The DNS names and IP addresses of a `subjectAltName` value.
fn names(value: &[u8]) -> Option<Vec<Name>> {
    let (mut list, mut out) = (Der(Der(value).expect(0x30)?), Vec::new());
    while let Some((tag, name, _)) = list.next() {
        match tag {
            0x82 => out.push(Name::Dns(std::str::from_utf8(name).ok()?.to_ascii_lowercase())),
            0x87 => out.push(Name::Ip(name.to_vec())),
            _ => {}
        }
    }
    Some(out)
}

/// The certificates of a PEM bundle that parse (a system bundle holds some this client cannot use).
pub fn anchors(bundle: &str) -> Vec<Certificate> {
    pem_certificates(bundle).into_iter().filter_map(|der| Certificate::parse(der).ok()).collect()
}

/// The system's trust anchors, from the first bundle found at the usual paths (Debian and Ubuntu, Fedora and
/// RHEL, openSUSE, Alpine).
pub fn system_anchors() -> Vec<Certificate> {
    const BUNDLES: [&str; 3] = ["/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/ca-bundle.pem"];
    BUNDLES.iter().find_map(|p| std::fs::read_to_string(p).ok()).map_or_else(Vec::new, |text| anchors(&text))
}

/// Checks that `chain` (server certificate first) names `host`, is valid at `now`, and leads by valid signatures
/// to one of the `anchors`.
pub fn verify_chain(chain: &[Certificate], anchors: &[Certificate], host: &str, now: i64) -> Result<(), Rejected> {
    let leaf = chain.first().ok_or(Rejected::Malformed)?;
    if leaf.key.is_none() || !leaf.ext.serves_tls {
        return Err(Rejected::Unsupported);
    }
    if !leaf.names_host(host) {
        return Err(Rejected::WrongHost);
    }
    for (i, cert) in chain.iter().enumerate() {
        if let Some(verdict) = link(cert, chain.get(i + 1), anchors, now) {
            return verdict;
        }
    }
    Err(Rejected::UnknownIssuer)
}

/// The verdict on a chain at `cert` when it is a trust anchor or one issued it.
fn anchored(cert: &Certificate, anchors: &[Certificate], now: i64) -> Option<Result<(), Rejected>> {
    if anchors.iter().any(|a| a.der == cert.der) {
        return Some(Ok(()));
    }
    let anchor = anchors.iter().find(|a| a.issues() && cert.issued_by(a))?;
    Some(if anchor.current(now) { Ok(()) } else { Err(Rejected::Expired) })
}

/// The verdict on a chain at `cert`, whose sender put `next` after it; `None` when `next` issued it and the
/// check goes on there.
fn link(cert: &Certificate, next: Option<&Certificate>, anchors: &[Certificate], now: i64) -> Option<Result<(), Rejected>> {
    if !cert.current(now) {
        return Some(Err(Rejected::Expired));
    }
    if let Some(verdict) = anchored(cert, anchors, now) {
        return Some(verdict);
    }
    match next {
        Some(next) if next.issues() && cert.issued_by(next) => None,
        Some(next) if next.issues() && cert.der[cert.issuer.clone()] == *next.subject() => Some(Err(Rejected::BadSignature)),
        _ => Some(Err(Rejected::UnknownIssuer)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2027-01-01, inside the fixtures' validity.
    const NOW: i64 = 1_798_761_600;

    fn load(name: &str) -> Certificate {
        let text = std::fs::read_to_string(format!("{}/tests/fixtures/tls/{name}.pem", env!("CARGO_MANIFEST_DIR"))).unwrap();
        Certificate::parse(pem_certificates(&text).remove(0)).unwrap()
    }

    /// The fixtures of `scripts/tls-fixtures.sh`: a P-256 server under a P-256 intermediate under an RSA root, and
    /// an RSA server under the root.
    #[test]
    fn chains_are_checked_to_the_anchors() {
        let (root, inter, hub, rsa) = (load("root"), load("inter"), load("hub"), load("rsa"));
        assert!(matches!(hub.key, Some(Key::P256(_))) && matches!(rsa.key, Some(Key::Rsa(_))));
        let anchors = [root];
        let chain = [hub.clone(), inter.clone()];
        assert_eq!(verify_chain(&chain, &anchors, "hub.test", NOW), Ok(()));
        assert_eq!(verify_chain(&chain, &anchors, "x.cdn.test", NOW), Ok(()), "a wildcard covers one label");
        assert_eq!(verify_chain(&chain, &anchors, "a.b.cdn.test", NOW), Err(Rejected::WrongHost));
        assert_eq!(verify_chain(&chain, &anchors, "other.test", NOW), Err(Rejected::WrongHost));
        assert_eq!(verify_chain(std::slice::from_ref(&rsa), &anchors, "rsa.test", NOW), Ok(()));
        assert_eq!(verify_chain(&chain, &anchors, "hub.test", NOW + 200 * 365 * 86_400), Err(Rejected::Expired));
        assert_eq!(verify_chain(&chain, &[], "hub.test", NOW), Err(Rejected::UnknownIssuer));
        assert_eq!(verify_chain(std::slice::from_ref(&hub), &anchors, "hub.test", NOW), Err(Rejected::UnknownIssuer), "the intermediate is missing");
        let mut forged = hub.der.clone();
        let last = forged.len() - 5;
        forged[last] ^= 1;
        let forged = Certificate::parse(forged).unwrap();
        assert_eq!(verify_chain(&[forged, inter], &anchors, "hub.test", NOW), Err(Rejected::BadSignature));
    }

    /// The system bundle loads, minus what this client cannot use; skipped where there is none.
    #[test]
    fn the_system_bundle_loads() {
        if std::path::Path::new("/etc/ssl/certs/ca-certificates.crt").exists() {
            let all = pem_certificates(&std::fs::read_to_string("/etc/ssl/certs/ca-certificates.crt").unwrap()).len();
            let usable = system_anchors().len();
            assert!(usable > all / 2 && usable <= all, "{usable} of {all} anchors parse");
        }
    }
}
