//! Downloads over HTTPS with nothing but the standard library: TLS 1.3 and the certificate checks it needs, ported
//! from caudal and reduced to a client, an HTTP/1.1 client, and the Hugging Face Hub calls `tachyon pull` makes.

pub mod aes_gcm;
pub mod crypto;
pub mod http;
pub mod hub;
pub mod p256;
pub mod rsa;
pub mod tls;
pub mod x25519;
pub mod x509;
