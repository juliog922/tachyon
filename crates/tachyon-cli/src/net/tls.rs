//! A TLS 1.3 client (RFC 8446). Ported from caudal and reduced to its client half, with RSA servers added.
//!
//! [`Session`] is the protocol without I/O: it turns received bytes into plaintext and handshake replies, and
//! plaintext into records. [`Tls`] runs it over a TCP socket as a [`Read`] and [`Write`] stream. One configuration
//! of the protocol, the one every current server accepts: `TLS_AES_128_GCM_SHA256`, X25519 key exchange, ALPN
//! `http/1.1`, a server certificate with an ECDSA P-256 key or an RSA one (signing the handshake with RSA-PSS). No
//! `HelloRetryRequest`, no resumption, no early data, no client certificates. Middlebox compatibility mode is kept.

use super::aes_gcm::{Gcm, TAG};
use super::crypto::{Hmac, Sha256, ct_eq, hkdf_expand, hkdf_extract, wipe};
use super::x509::{self, Certificate, Key, Rejected};
use super::x25519;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

/// Largest plaintext of a record, and largest record on the wire.
const MAX_PLAIN: usize = 1 << 14;
const MAX_RECORD: usize = MAX_PLAIN + 256;
const HEADER: usize = 5;
const MAX_HANDSHAKE: usize = 64 << 10;
/// Records sealed under one key before it is updated, well inside AES-GCM's limit (RFC 8446 §5.5).
const REKEY_AFTER: u64 = 1 << 24;

const CHANGE_CIPHER_SPEC: u8 = 20;
const ALERT: u8 = 21;
const HANDSHAKE: u8 = 22;
const APPLICATION_DATA: u8 = 23;

const CLIENT_HELLO: u8 = 1;
const SERVER_HELLO: u8 = 2;
const NEW_SESSION_TICKET: u8 = 4;
const ENCRYPTED_EXTENSIONS: u8 = 8;
const CERTIFICATE: u8 = 11;
const CERTIFICATE_VERIFY: u8 = 15;
const FINISHED: u8 = 20;
const KEY_UPDATE: u8 = 24;

const SERVER_NAME: u16 = 0;
const SUPPORTED_GROUPS: u16 = 10;
const SIGNATURE_ALGORITHMS: u16 = 13;
const ALPN: u16 = 16;
const SUPPORTED_VERSIONS: u16 = 43;
const KEY_SHARE: u16 = 51;

const AES_128_GCM_SHA256: [u8; 2] = [0x13, 0x01];
const X25519: u16 = 0x001d;
const ECDSA_P256_SHA256: [u8; 2] = [0x04, 0x03];
const RSA_PSS_SHA256: [u8; 2] = [0x08, 0x04];
/// Signatures this client checks: on the handshake ECDSA P-256 and RSA-PSS, on certificates also RSA PKCS #1 v1.5.
const SIGNATURES: [u8; 6] = [0x04, 0x03, 0x08, 0x04, 0x04, 0x01];
const TLS13: [u8; 2] = [0x03, 0x04];
/// The ALPN extension data naming only `http/1.1`.
const ALPN_HTTP11: &[u8] = b"\x00\x09\x08http/1.1";
/// The `ServerHello` random that marks a `HelloRetryRequest`.
const RETRY_RANDOM: [u8; 32] = [
    0xcf, 0x21, 0xad, 0x74, 0xe5, 0x9a, 0x61, 0x11, 0xbe, 0x1d, 0x8c, 0x02, 0x1e, 0x65, 0xb8, 0x91, 0xc2, 0xa2, 0x11, 0x16, 0x7a, 0xbb, 0x8c, 0x5e, 0x07, 0x9e,
    0x09, 0xe2, 0xc8, 0xa8, 0x33, 0x9c,
];
const SERVER_VERIFY_CONTEXT: &[u8] = b"TLS 1.3, server CertificateVerify\0";

/// Alerts this client sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Alert {
    /// The end of the session.
    CloseNotify = 0,
    /// A message out of order.
    UnexpectedMessage = 10,
    /// A record that does not decrypt.
    BadRecordMac = 20,
    /// A record too long.
    RecordOverflow = 22,
    /// No acceptable parameters (or a `HelloRetryRequest`).
    HandshakeFailure = 40,
    /// A certificate malformed, for another host, or badly signed.
    BadCertificate = 42,
    /// A certificate with a key, signature or extension this client does not understand.
    UnsupportedCertificate = 43,
    /// A certificate outside its validity period.
    CertificateExpired = 45,
    /// A field with a value the protocol forbids.
    IllegalParameter = 47,
    /// A chain no trust anchor issued.
    UnknownCa = 48,
    /// A message that does not parse.
    DecodeError = 50,
    /// A signature or `Finished` that does not check.
    DecryptError = 51,
    /// Not TLS 1.3.
    ProtocolVersion = 70,
    /// A required extension missing.
    MissingExtension = 109,
    /// An extension the client did not offer.
    UnsupportedExtension = 110,
}

/// Why a session ended: an alert this client sent (and appended to the output), or one the server sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// This client's alert.
    Local(Alert),
    /// The server's alert code.
    Peer(u8),
}

use Alert::{
    BadCertificate, BadRecordMac, CertificateExpired, DecodeError, DecryptError, HandshakeFailure, IllegalParameter, MissingExtension, ProtocolVersion,
    RecordOverflow, UnexpectedMessage, UnknownCa, UnsupportedCertificate, UnsupportedExtension,
};

fn fail<T>(alert: Alert) -> Result<T, Error> {
    Err(Error::Local(alert))
}

/// What a client trusts, its entropy and its clock, replaceable in tests.
pub struct ClientConfig {
    /// Trust anchors.
    pub anchors: Vec<Certificate>,
    /// Fills a buffer with random bytes.
    pub random: fn(&mut [u8]),
    /// Seconds since the Unix epoch, for certificate validity.
    pub now: fn() -> i64,
}

impl ClientConfig {
    /// The system's trust anchors, the kernel's randomness and the system clock.
    pub fn system() -> ClientConfig {
        ClientConfig { anchors: x509::system_anchors(), random: super::crypto::random, now: x509::now }
    }
}

/// Keys and record counter of one direction, wiped on drop. Boxed in a session, and with no implicit padding, so
/// no byte of its heap block is left holding whatever the stack had there before.
#[repr(C)]
struct Direction {
    gcm: Gcm,
    secret: [u8; 32],
    seq: u64,
    iv: [u8; 12],
    spare: [u8; 12],
}

const _: () = assert!(size_of::<Direction>() == size_of::<Gcm>() + 64, "Direction has no padding");

impl Direction {
    fn new(secret: &[u8; 32]) -> Box<Direction> {
        let (mut key, mut iv) = ([0u8; 16], [0u8; 12]);
        expand_label(secret, "key", &[], &mut key);
        expand_label(secret, "iv", &[], &mut iv);
        let d = Box::new(Direction { gcm: Gcm::new(&key), secret: *secret, seq: 0, iv, spare: [0; 12] });
        wipe(&mut key);
        d
    }

    fn nonce(&mut self) -> [u8; 12] {
        let mut n = self.iv;
        n[4..].iter_mut().zip(self.seq.to_be_bytes()).for_each(|(b, s)| *b ^= s);
        self.seq = self.seq.checked_add(1).expect("a key never seals 2^64 records");
        n
    }

    /// The next generation of keys (`KeyUpdate`).
    fn update(&mut self) {
        let mut next = [0u8; 32];
        expand_label(&self.secret, "traffic upd", &[], &mut next);
        *self = *Direction::new(&next);
        wipe(&mut next);
    }
}

impl Drop for Direction {
    fn drop(&mut self) {
        wipe(&mut self.secret);
        wipe(&mut self.iv);
    }
}

fn expand_label(secret: &[u8], label: &str, context: &[u8], out: &mut [u8]) {
    let (len, label_len) = ((out.len() as u16).to_be_bytes(), [6 + label.len() as u8]);
    hkdf_expand(secret, &[&len, &label_len, b"tls13 ", label.as_bytes(), &[context.len() as u8], context], out);
}

fn derive(secret: &[u8; 32], label: &str, transcript: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    expand_label(secret, label, transcript, &mut out);
    out
}

fn finished(base: &[u8; 32], transcript: &[u8; 32]) -> [u8; 32] {
    let mut key = [0u8; 32];
    expand_label(base, "finished", &[], &mut key);
    let mac = Hmac::mac(&key, transcript);
    wipe(&mut key);
    mac
}

/// Reads the length-prefixed structures of handshake messages.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let (head, rest) = self.0.split_at_checked(n).ok_or(Error::Local(DecodeError))?;
        self.0 = rest;
        Ok(head)
    }

    fn int(&mut self, n: usize) -> Result<usize, Error> {
        Ok(self.bytes(n)?.iter().fold(0, |v, &b| v << 8 | b as usize))
    }

    fn vec(&mut self, prefix: usize) -> Result<&'a [u8], Error> {
        let n = self.int(prefix)?;
        self.bytes(n)
    }

    fn end(&self) -> Result<(), Error> {
        if self.0.is_empty() { Ok(()) } else { fail(DecodeError) }
    }

    /// Calls `each` with the type and data of every extension in a u16-prefixed block. A repeated type is refused.
    fn extensions(&mut self, mut each: impl FnMut(u16, Reader<'a>) -> Result<(), Error>) -> Result<(), Error> {
        let mut r = Reader(self.vec(2)?);
        let mut seen = 0u64;
        while !r.0.is_empty() {
            let (kind, data) = (r.int(2)? as u16, r.vec(2)?);
            if kind < 64 && seen & 1 << kind != 0 {
                return fail(IllegalParameter);
            }
            seen |= 1u64.checked_shl(u32::from(kind)).unwrap_or(0);
            each(kind, Reader(data))?;
        }
        Ok(())
    }
}

/// Appends what `body` writes after its big-endian length in `width` bytes.
fn prefixed(out: &mut Vec<u8>, width: usize, body: impl FnOnce(&mut Vec<u8>)) {
    let at = out.len();
    out.extend_from_slice(&[0; 3][..width]);
    body(out);
    let len = out.len() - at - width;
    out[at..at + width].copy_from_slice(&len.to_be_bytes()[8 - width..]);
}

fn extension(out: &mut Vec<u8>, kind: u16, body: impl FnOnce(&mut Vec<u8>)) {
    out.extend_from_slice(&kind.to_be_bytes());
    prefixed(out, 2, body);
}

/// Appends an unencrypted record; returns where its content starts.
fn plain_record(out: &mut Vec<u8>, kind: u8, legacy: u8, body: impl FnOnce(&mut Vec<u8>)) -> usize {
    out.extend_from_slice(&[kind, 3, legacy]);
    let at = out.len() + 2;
    prefixed(out, 2, body);
    at
}

/// The digest a `CertificateVerify` signs: 64 spaces, the context, the transcript hash.
fn verify_digest(transcript: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::default();
    h.update(&[0x20; 64]);
    h.update(SERVER_VERIFY_CONTEXT);
    h.update(transcript);
    h.finish()
}

fn rejected(r: Rejected) -> Error {
    Error::Local(match r {
        Rejected::Malformed | Rejected::WrongHost | Rejected::BadSignature => BadCertificate,
        Rejected::Unsupported => UnsupportedCertificate,
        Rejected::Expired => CertificateExpired,
        Rejected::UnknownIssuer => UnknownCa,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    ServerHello,
    EncryptedExtensions,
    Certificate,
    CertificateVerify,
    Finished,
    Connected,
    Closed,
}

/// One TLS client connection, without I/O.
pub struct Session {
    config: Arc<ClientConfig>,
    host: String,
    hello_id: [u8; 32],
    ephemeral: [u8; 32],
    /// The server certificate's key, once its chain checks.
    peer: Option<Key>,
    state: State,
    transcript: Sha256,
    read: Option<Box<Direction>>,
    write: Option<Box<Direction>>,
    /// Handshake bytes received but not yet a whole message.
    handshake: Vec<u8>,
    /// The handshake traffic secrets, kept until the server's `Finished` checks; then the handshake secret.
    client_secret: [u8; 32],
    server_secret: [u8; 32],
    pending: [u8; 32],
    change_cipher_spec: bool,
}

impl Drop for Session {
    fn drop(&mut self) {
        wipe(&mut self.client_secret);
        wipe(&mut self.server_secret);
        wipe(&mut self.pending);
        wipe(&mut self.ephemeral);
    }
}

impl Session {
    /// A session with the server `host`; its `ClientHello` is appended to `out`.
    pub fn client(config: Arc<ClientConfig>, host: &str, out: &mut Vec<u8>) -> Session {
        let (mut random, mut session_id, mut ephemeral) = ([0u8; 32], [0u8; 32], [0u8; 32]);
        (config.random)(&mut random);
        (config.random)(&mut session_id);
        (config.random)(&mut ephemeral);
        let share = x25519::public(&ephemeral);
        let at = plain_record(out, HANDSHAKE, 1, |o| {
            o.push(CLIENT_HELLO);
            prefixed(o, 3, |o| {
                o.extend_from_slice(&[3, 3]);
                o.extend_from_slice(&random);
                prefixed(o, 1, |o| o.extend_from_slice(&session_id));
                prefixed(o, 2, |o| o.extend_from_slice(&AES_128_GCM_SHA256));
                o.extend_from_slice(&[1, 0]);
                prefixed(o, 2, |o| hello_extensions(o, host, &share));
            });
        });
        let mut s = Session {
            config,
            host: host.into(),
            hello_id: session_id,
            ephemeral,
            peer: None,
            state: State::ServerHello,
            transcript: Sha256::default(),
            read: None,
            write: None,
            handshake: Vec::new(),
            client_secret: [0; 32],
            server_secret: [0; 32],
            pending: [0; 32],
            change_cipher_spec: false,
        };
        s.transcript.update(&out[at..]);
        wipe(&mut ephemeral);
        s
    }

    /// Whether the handshake is done, so application data can flow.
    pub fn established(&self) -> bool {
        self.state == State::Connected
    }

    /// Whether the session is over: the server closed it with `close_notify`, or an alert ended it.
    pub fn closed(&self) -> bool {
        self.state == State::Closed
    }

    /// Consumes the whole records at the start of `wire` and returns how many bytes they took. Application data is
    /// appended to `plain`; handshake replies, and on failure an alert, to `out`.
    pub fn receive(&mut self, wire: &[u8], plain: &mut Vec<u8>, out: &mut Vec<u8>) -> Result<usize, Error> {
        let mut used = 0;
        while self.state != State::Closed && wire.len() - used >= HEADER {
            let rest = &wire[used..];
            let len = u16::from_be_bytes([rest[3], rest[4]]) as usize;
            let limit = if self.read.is_some() { MAX_RECORD } else { MAX_PLAIN };
            if rest[1] != 3 || len > limit {
                return Err(self.abort(Error::Local(if rest[1] == 3 { RecordOverflow } else { DecodeError }), out));
            }
            if rest.len() < HEADER + len {
                break;
            }
            let (header, body) = rest[..HEADER + len].split_at(HEADER);
            if let Err(e) = self.record(header, body, plain, out) {
                return Err(self.abort(e, out));
            }
            used += HEADER + len;
        }
        Ok(used)
    }

    fn abort(&mut self, e: Error, out: &mut Vec<u8>) -> Error {
        if let Error::Local(alert) = e {
            self.alert(alert, out);
        }
        self.state = State::Closed;
        e
    }

    /// Appends an alert record; after a fatal one the session is closed.
    pub fn alert(&mut self, alert: Alert, out: &mut Vec<u8>) {
        let body = [if alert == Alert::CloseNotify { 1 } else { 2 }, alert as u8];
        if self.write.is_some() {
            self.seal(ALERT, &[&body], out);
        } else {
            plain_record(out, ALERT, 3, |o| o.extend_from_slice(&body));
        }
        self.state = State::Closed;
    }

    fn record(&mut self, header: &[u8], body: &[u8], plain: &mut Vec<u8>, out: &mut Vec<u8>) -> Result<(), Error> {
        let kind = header[0];
        if kind == CHANGE_CIPHER_SPEC {
            let handshaking = self.state != State::Connected;
            let first = !std::mem::replace(&mut self.change_cipher_spec, true);
            return if handshaking && body == [1] && first { Ok(()) } else { fail(UnexpectedMessage) };
        }
        if self.read.is_some() {
            return self.protected(header, body, plain, out);
        }
        match kind {
            HANDSHAKE if !body.is_empty() => {
                self.handshake.extend_from_slice(body);
                self.handshakes(out)
            }
            ALERT => self.peer_alert(body),
            _ => fail(UnexpectedMessage),
        }
    }

    /// An encrypted record: application data appended to `plain`, or a handshake message or alert handled.
    fn protected(&mut self, header: &[u8], body: &[u8], plain: &mut Vec<u8>, out: &mut Vec<u8>) -> Result<(), Error> {
        let (read, start) = (self.read.as_mut().expect("keys are in place"), plain.len());
        let nonce = read.nonce();
        if header[0] != APPLICATION_DATA || !read.gcm.open(&nonce, header, body, plain) {
            return fail(if header[0] == APPLICATION_DATA { BadRecordMac } else { UnexpectedMessage });
        }
        let Some(end) = plain[start..].iter().rposition(|&b| b != 0).filter(|&end| end <= MAX_PLAIN) else {
            plain.truncate(start);
            return fail(UnexpectedMessage);
        };
        let inner = plain[start + end];
        plain.truncate(start + end);
        if inner == APPLICATION_DATA && self.state == State::Connected {
            return Ok(());
        }
        let content = plain.split_off(start);
        match inner {
            HANDSHAKE if !content.is_empty() => {
                self.handshake.extend_from_slice(&content);
                self.handshakes(out)
            }
            ALERT => self.peer_alert(&content),
            _ => fail(UnexpectedMessage),
        }
    }

    fn peer_alert(&mut self, body: &[u8]) -> Result<(), Error> {
        match body {
            [_, 0] => {
                self.state = State::Closed;
                Ok(())
            }
            [_, 90] => Ok(()),
            [_, code] => Err(Error::Peer(*code)),
            _ => fail(DecodeError),
        }
    }

    /// Handles every whole message in the handshake buffer.
    fn handshakes(&mut self, out: &mut Vec<u8>) -> Result<(), Error> {
        let buffer = std::mem::take(&mut self.handshake);
        let mut at = 0;
        while buffer.len() - at >= 4 {
            let len = Reader(&buffer[at + 1..at + 4]).int(3)?;
            if len > MAX_HANDSHAKE {
                return fail(UnexpectedMessage);
            }
            if buffer.len() - at < 4 + len {
                break;
            }
            let whole = &buffer[at..at + 4 + len];
            at += 4 + len;
            self.message(whole, at == buffer.len(), out)?;
        }
        if buffer.len() - at > MAX_HANDSHAKE {
            return fail(UnexpectedMessage);
        }
        self.handshake = buffer[at..].to_vec();
        Ok(())
    }

    /// One handshake message, whole. `last`: nothing follows it in the buffer, as messages that change keys require.
    fn message(&mut self, whole: &[u8], last: bool, out: &mut Vec<u8>) -> Result<(), Error> {
        let (kind, body) = (whole[0], &whole[4..]);
        let rekeys = matches!((self.state, kind), (State::Finished, FINISHED) | (State::ServerHello, SERVER_HELLO) | (State::Connected, KEY_UPDATE));
        if rekeys && !last {
            return fail(UnexpectedMessage);
        }
        match (self.state, kind) {
            (State::ServerHello, SERVER_HELLO) => self.server_hello(body, whole),
            (State::EncryptedExtensions, ENCRYPTED_EXTENSIONS) => self.encrypted_extensions(body, whole),
            (State::Certificate, CERTIFICATE) => self.certificate(body, whole),
            (State::CertificateVerify, CERTIFICATE_VERIFY) => self.certificate_verify(body, whole),
            (State::Finished, FINISHED) => self.finished(body, whole, out),
            (State::Connected, NEW_SESSION_TICKET) => Ok(()),
            (State::Connected, KEY_UPDATE) => self.key_update(body, out),
            _ => fail(UnexpectedMessage),
        }
    }

    fn server_hello(&mut self, body: &[u8], whole: &[u8]) -> Result<(), Error> {
        let mut r = Reader(body);
        hello_fields(&mut r, &self.hello_id)?;
        let peer = server_share(&mut r)?;
        let mut shared = x25519::x25519(&self.ephemeral, &peer);
        wipe(&mut self.ephemeral);
        if shared == [0; 32] {
            return fail(IllegalParameter);
        }
        self.transcript.update(whole);
        self.handshake_secrets(&mut shared);
        self.read = Some(Direction::new(&self.server_secret));
        self.write = Some(Direction::new(&self.client_secret));
        self.state = State::EncryptedExtensions;
        Ok(())
    }

    fn encrypted_extensions(&mut self, body: &[u8], whole: &[u8]) -> Result<(), Error> {
        let mut r = Reader(body);
        r.extensions(|kind, data| if kind == ALPN && data.0 != ALPN_HTTP11 { fail(IllegalParameter) } else { Ok(()) })?;
        r.end()?;
        self.transcript.update(whole);
        self.state = State::Certificate;
        Ok(())
    }

    fn certificate(&mut self, body: &[u8], whole: &[u8]) -> Result<(), Error> {
        let mut r = Reader(body);
        if !r.vec(1)?.is_empty() {
            return fail(IllegalParameter);
        }
        let mut list = Reader(r.vec(3)?);
        r.end()?;
        let mut chain = Vec::new();
        while !list.0.is_empty() {
            let der = list.vec(3)?;
            list.vec(2)?;
            chain.push(Certificate::parse(der.to_vec()).map_err(rejected)?);
        }
        x509::verify_chain(&chain, &self.config.anchors, &self.host, (self.config.now)()).map_err(rejected)?;
        self.peer = chain.swap_remove(0).key;
        self.transcript.update(whole);
        self.state = State::CertificateVerify;
        Ok(())
    }

    fn certificate_verify(&mut self, body: &[u8], whole: &[u8]) -> Result<(), Error> {
        let mut r = Reader(body);
        let (scheme, sig) = (r.bytes(2)?, r.vec(2)?);
        r.end()?;
        let digest = verify_digest(&self.transcript.snapshot());
        let good = match (&self.peer, <[u8; 2]>::try_from(scheme)) {
            (Some(Key::P256(key)), Ok(ECDSA_P256_SHA256)) => key.verify(&digest, &x509::raw_signature(sig).ok_or(Error::Local(DecodeError))?),
            (Some(Key::Rsa(key)), Ok(RSA_PSS_SHA256)) => key.verify_pss(&digest, sig),
            _ => return fail(IllegalParameter),
        };
        if !good {
            return fail(DecryptError);
        }
        self.transcript.update(whole);
        self.state = State::Finished;
        Ok(())
    }

    /// The server's `Finished`: checked, then the client's sent and the application keys installed.
    fn finished(&mut self, body: &[u8], whole: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        if !ct_eq(body, &finished(&self.server_secret, &self.transcript.snapshot())) {
            return fail(DecryptError);
        }
        self.transcript.update(whole);
        let (mut client, mut server) = self.application_secrets();
        let mut fin = [0u8; 36];
        fin[..4].copy_from_slice(&[FINISHED, 0, 0, 32]);
        fin[4..].copy_from_slice(&finished(&self.client_secret, &self.transcript.snapshot()));
        plain_record(out, CHANGE_CIPHER_SPEC, 3, |o| o.push(1));
        self.seal(HANDSHAKE, &[&fin], out);
        self.write = Some(Direction::new(&client));
        self.read = Some(Direction::new(&server));
        for secret in [&mut client, &mut server, &mut self.client_secret, &mut self.server_secret] {
            wipe(secret);
        }
        self.state = State::Connected;
        Ok(())
    }

    fn key_update(&mut self, body: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        let request = match body {
            [0] => false,
            [1] => true,
            _ => return fail(IllegalParameter),
        };
        self.read.as_mut().expect("connected").update();
        if request {
            self.seal(HANDSHAKE, &[&[KEY_UPDATE, 0, 0, 1, 0]], out);
            self.write.as_mut().expect("connected").update();
        }
        Ok(())
    }

    /// Derives the handshake traffic secrets from the ECDHE secret, which is wiped, and keeps the handshake secret
    /// in `pending` for the application secrets.
    fn handshake_secrets(&mut self, shared: &mut [u8; 32]) {
        let empty = Sha256::digest(&[]);
        let early = hkdf_extract(&[0; 32], &[0; 32]);
        let mut secret = hkdf_extract(&derive(&early, "derived", &empty), shared);
        wipe(shared);
        let hash = self.transcript.snapshot();
        self.client_secret = derive(&secret, "c hs traffic", &hash);
        self.server_secret = derive(&secret, "s hs traffic", &hash);
        self.pending = derive(&secret, "derived", &empty);
        wipe(&mut secret);
    }

    /// The (client, server) application traffic secrets, from the transcript through the server's `Finished`.
    fn application_secrets(&mut self) -> ([u8; 32], [u8; 32]) {
        let mut master = hkdf_extract(&self.pending, &[0; 32]);
        wipe(&mut self.pending);
        let hash = self.transcript.snapshot();
        let secrets = (derive(&master, "c ap traffic", &hash), derive(&master, "s ap traffic", &hash));
        wipe(&mut master);
        secrets
    }

    /// Appends the records carrying `data` as application data.
    pub fn send(&mut self, data: &[u8], out: &mut Vec<u8>) {
        self.seal(APPLICATION_DATA, &[data], out);
    }

    /// Encrypts the concatenated `parts` into records of at most [`MAX_PLAIN`] bytes each. After [`REKEY_AFTER`]
    /// records under one key, application data first moves to the next key with a `KeyUpdate` (RFC 8446 §5.5).
    fn seal(&mut self, kind: u8, parts: &[&[u8]], out: &mut Vec<u8>) {
        let write = self.write.as_mut().expect("keys are in place before anything is sealed");
        let data = parts.concat();
        for piece in data.chunks(MAX_PLAIN).chain(data.is_empty().then_some(&[][..])) {
            if kind == APPLICATION_DATA && write.seq >= REKEY_AFTER {
                let header = [APPLICATION_DATA, 3, 3, 0, 6 + TAG as u8];
                out.extend_from_slice(&header);
                let nonce = write.nonce();
                write.gcm.seal(&nonce, &header, &[&[KEY_UPDATE, 0, 0, 1, 0, HANDSHAKE]], out);
                write.update();
            }
            let len = piece.len() + 1 + TAG;
            let header = [APPLICATION_DATA, 3, 3, (len >> 8) as u8, len as u8];
            out.extend_from_slice(&header);
            let nonce = write.nonce();
            write.gcm.seal(&nonce, &header, &[piece, &[kind]], out);
        }
    }
}

/// The fixed fields of a `ServerHello`: TLS 1.2 as the legacy version, not a `HelloRetryRequest`, our session ID
/// echoed, our cipher suite, no compression.
fn hello_fields(r: &mut Reader, session_id: &[u8]) -> Result<(), Error> {
    if r.bytes(2)? != [3, 3] {
        return fail(ProtocolVersion);
    }
    if r.bytes(32)? == RETRY_RANDOM {
        return fail(HandshakeFailure);
    }
    let (id, suite, compression) = (r.vec(1)?, r.bytes(2)?, r.bytes(1)?);
    if (id, suite, compression) == (session_id, &AES_128_GCM_SHA256[..], &[0][..]) { Ok(()) } else { fail(IllegalParameter) }
}

/// The server's X25519 share, from the extensions that end a `ServerHello`, which must also choose TLS 1.3.
fn server_share(r: &mut Reader) -> Result<[u8; 32], Error> {
    let (mut tls13, mut share) = (false, None);
    r.extensions(|kind, d| hello_extension(kind, d, &mut tls13, &mut share))?;
    r.end()?;
    match (tls13, share) {
        (false, _) => fail(ProtocolVersion),
        (true, None) => fail(MissingExtension),
        (true, Some(share)) => Ok(share),
    }
}

/// One `ServerHello` extension: the version chosen, or the key share.
fn hello_extension(kind: u16, mut d: Reader, tls13: &mut bool, share: &mut Option<[u8; 32]>) -> Result<(), Error> {
    match kind {
        SUPPORTED_VERSIONS => *tls13 = d.bytes(2)? == TLS13,
        KEY_SHARE if d.int(2)? as u16 == X25519 => *share = Some(<[u8; 32]>::try_from(d.vec(2)?).map_err(|_| Error::Local(IllegalParameter))?),
        KEY_SHARE => return fail(IllegalParameter),
        _ => return fail(UnsupportedExtension),
    }
    d.end()
}

/// The `ClientHello` extensions for `host`: its name (unless it is an IP address), X25519, the signatures this
/// client checks, TLS 1.3, the key share, and ALPN `http/1.1`.
fn hello_extensions(o: &mut Vec<u8>, host: &str, share: &[u8; 32]) {
    if host.parse::<std::net::IpAddr>().is_err() {
        extension(o, SERVER_NAME, |o| {
            prefixed(o, 2, |o| {
                o.push(0);
                prefixed(o, 2, |o| o.extend_from_slice(host.as_bytes()));
            });
        });
    }
    extension(o, SUPPORTED_GROUPS, |o| prefixed(o, 2, |o| o.extend_from_slice(&X25519.to_be_bytes())));
    extension(o, SIGNATURE_ALGORITHMS, |o| prefixed(o, 2, |o| o.extend_from_slice(&SIGNATURES)));
    extension(o, SUPPORTED_VERSIONS, |o| prefixed(o, 1, |o| o.extend_from_slice(&TLS13)));
    extension(o, KEY_SHARE, |o| {
        prefixed(o, 2, |o| {
            o.extend_from_slice(&X25519.to_be_bytes());
            prefixed(o, 2, |o| o.extend_from_slice(share));
        });
    });
    extension(o, ALPN, |o| o.extend_from_slice(ALPN_HTTP11));
}

/// A TLS session over a TCP connection, as a byte stream.
pub struct Tls {
    sock: TcpStream,
    session: Session,
    /// Received bytes not yet whole records; plaintext received, and how much of it was read; records to send.
    raw: Vec<u8>,
    plain: Vec<u8>,
    taken: usize,
    out: Vec<u8>,
}

impl Tls {
    /// Runs the handshake with `host` over the connected socket `sock`.
    pub fn connect(sock: TcpStream, host: &str, config: &Arc<ClientConfig>) -> io::Result<Tls> {
        let mut out = Vec::with_capacity(MAX_RECORD);
        let session = Session::client(config.clone(), host, &mut out);
        let mut tls = Tls { sock, session, raw: Vec::with_capacity(4 * MAX_RECORD), plain: Vec::new(), taken: 0, out };
        tls.flush()?;
        while !tls.session.established() {
            if !tls.pump()? {
                return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "the server closed the connection during the TLS handshake"));
            }
        }
        Ok(tls)
    }

    /// Reads from the socket and decrypts what it can; false once the connection is over.
    fn pump(&mut self) -> io::Result<bool> {
        if self.session.closed() {
            return Ok(false);
        }
        let len = self.raw.len();
        self.raw.resize(len + 4 * MAX_RECORD, 0);
        let n = self.sock.read(&mut self.raw[len..]);
        self.raw.truncate(len + *n.as_ref().unwrap_or(&0));
        if n? == 0 {
            return Ok(false);
        }
        let used = self.session.receive(&self.raw, &mut self.plain, &mut self.out);
        self.flush()?;
        let used = used.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("TLS: {e:?}")))?;
        self.raw.drain(..used);
        Ok(true)
    }
}

impl Read for Tls {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.taken == self.plain.len() {
            self.taken = 0;
            self.plain.clear();
            if !self.pump()? {
                return Ok(0);
            }
        }
        let n = buf.len().min(self.plain.len() - self.taken);
        buf[..n].copy_from_slice(&self.plain[self.taken..self.taken + n]);
        self.taken += n;
        Ok(n)
    }
}

impl Write for Tls {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.session.send(buf, &mut self.out);
        self.flush()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sock.write_all(&self.out)?;
        self.out.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};

    /// 2027-01-01, inside the fixtures' validity.
    fn now() -> i64 {
        1_798_761_600
    }

    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/tls/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    /// `openssl s_server` with the fixture `cert`, answering `GET` with its status page; `None` without OpenSSL.
    fn server(cert: &str, chain: bool) -> Option<(Child, u16)> {
        let port = std::net::TcpListener::bind("127.0.0.1:0").ok()?.local_addr().ok()?.port();
        let (pem, key) = (fixture(&format!("{cert}.pem")), fixture(&format!("{cert}.key")));
        let mut cmd = Command::new("openssl");
        cmd.args(["s_server", "-accept", &format!("127.0.0.1:{port}"), "-cert", &pem, "-key", &key, "-tls1_3", "-www", "-quiet", "-naccept", "4"]);
        if chain {
            cmd.args(["-cert_chain", &fixture("inter.pem")]);
        }
        let child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()?;
        let up = (0..100).any(|_| {
            let up = TcpStream::connect(("127.0.0.1", port)).is_ok();
            std::thread::sleep(std::time::Duration::from_millis(if up { 0 } else { 50 }));
            up
        });
        up.then_some((child, port))
    }

    fn get(port: u16, host: &str) -> io::Result<String> {
        let anchors = x509::pem_certificates(&std::fs::read_to_string(fixture("root.pem")).unwrap());
        let config = Arc::new(ClientConfig { anchors: vec![Certificate::parse(anchors[0].clone()).unwrap()], random: super::super::crypto::random, now });
        let mut tls = Tls::connect(TcpStream::connect(("127.0.0.1", port))?, host, &config)?;
        tls.write_all(b"GET / HTTP/1.0\r\n\r\n")?;
        let mut text = String::new();
        tls.read_to_string(&mut text)?;
        Ok(text)
    }

    /// The client against OpenSSL's server: an ECDSA P-256 certificate under an intermediate, and an RSA one
    /// (`CertificateVerify` in RSA-PSS); a certificate for another host is refused. Skipped without `openssl`.
    #[test]
    fn talks_to_openssl_with_ecdsa_and_rsa_certificates() {
        for (cert, chain, host) in [("hub", true, "hub.test"), ("rsa", false, "rsa.test")] {
            let Some((mut child, port)) = server(cert, chain) else {
                assert!(Command::new("openssl").arg("version").output().is_err(), "OpenSSL is installed but its server did not start");
                return;
            };
            let answer = get(port, host);
            let wrong = get(port, "other.test");
            child.kill().unwrap();
            let answer = answer.unwrap();
            assert!(answer.starts_with("HTTP/1.0 200 ok"), "{cert}: {answer:.80}");
            assert!(answer.contains("TLS_AES_128_GCM_SHA256"), "{cert}: the status page names the cipher");
            let wrong = wrong.unwrap_err().to_string();
            assert!(wrong.contains("BadCertificate"), "{cert}: {wrong}");
        }
    }
}
