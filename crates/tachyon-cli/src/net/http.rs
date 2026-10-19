//! An HTTP/1.1 client: `GET` over a kept-alive connection, bodies by `Content-Length`, chunked or to
//! the end of the connection, redirects across hosts, and byte ranges. HTTPS through [`super::tls`]; plain HTTP
//! for local servers and tests. The parser follows caudal's server.

use super::tls::{ClientConfig, Tls};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

/// Longest response head, and most redirects followed.
const MAX_HEAD: usize = 64 << 10;
const MAX_REDIRECTS: usize = 8;
/// How long a read or write may wait before the connection is given up.
const TIMEOUT: Duration = Duration::from_secs(60);

/// A URL this client fetches: `http` or `https`, a host, a port, and the path with its query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    /// Whether the connection uses TLS.
    pub https: bool,
    /// Host name or address.
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Path and query, from the first `/`.
    pub path: String,
}

impl Url {
    /// Parses an absolute `http://` or `https://` URL.
    pub fn parse(url: &str) -> Option<Url> {
        let (https, rest) = match url.split_once("://")? {
            ("https", rest) => (true, rest),
            ("http", rest) => (false, rest),
            _ => return None,
        };
        let (authority, path) = rest.find(['/', '?']).map_or((rest, "/"), |i| rest.split_at(i));
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) => (host, port.parse().ok()?),
            None => (authority, if https { 443 } else { 80 }),
        };
        let path = if path.starts_with('?') { format!("/{path}") } else { path.to_string() };
        (!host.is_empty()).then(|| Url { https, host: host.to_string(), port, path })
    }

    /// `location` resolved against this URL: absolute, or a path on the same host.
    fn join(&self, location: &str) -> Option<Url> {
        if location.starts_with('/') {
            return Some(Url { path: location.to_string(), ..self.clone() });
        }
        Url::parse(location)
    }
}

/// A response's status and headers (names lowercase).
#[derive(Debug, Clone)]
pub struct Response {
    /// The status code.
    pub status: u16,
    /// Every header, name lowercase, in order.
    pub headers: Vec<(String, String)>,
}

impl Response {
    /// The first header `name` (lowercase).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn length(&self) -> Option<u64> {
        self.header("content-length").and_then(|v| v.trim().parse().ok())
    }
}

/// A byte stream a connection runs on: TLS, or plain TCP.
trait Duplex: Read + Write + Send {}
impl<T: Read + Write + Send> Duplex for T {}

/// A kept-alive connection to one host.
pub struct Connection {
    url: Url,
    stream: Box<dyn Duplex>,
    /// Bytes received after the last response's end.
    buf: Vec<u8>,
    /// The server will close the connection after this response.
    closing: bool,
}

impl Connection {
    /// Connects to the host and port of `url`, with TLS for `https`.
    pub fn open(url: &Url, config: &Arc<ClientConfig>) -> io::Result<Connection> {
        let sock = TcpStream::connect((url.host.as_str(), url.port))?;
        sock.set_read_timeout(Some(TIMEOUT))?;
        sock.set_write_timeout(Some(TIMEOUT))?;
        sock.set_nodelay(true)?;
        let stream: Box<dyn Duplex> = if url.https { Box::new(Tls::connect(sock, &url.host, config)?) } else { Box::new(sock) };
        Ok(Connection { url: url.clone(), stream, buf: Vec::with_capacity(MAX_HEAD), closing: false })
    }

    /// Whether this connection reaches the host of `url` and can take another request.
    pub fn serves(&self, url: &Url) -> bool {
        !self.closing && (self.url.https, &self.url.host, self.url.port) == (url.https, &url.host, url.port)
    }

    /// Sends `GET` for `path` with `headers` and returns the response, its body handed to `sink` in pieces (a
    /// redirect's body is dropped).
    pub fn get(&mut self, path: &str, headers: &[(&str, &str)], sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<Response> {
        let mut head =
            format!("GET {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: tachyon/{}\r\nAccept-Encoding: identity\r\n", self.url.host, env!("CARGO_PKG_VERSION"));
        for (k, v) in headers {
            head = head + k + ": " + v + "\r\n";
        }
        head.push_str("\r\n");
        self.stream.write_all(head.as_bytes())?;
        let response = self.head()?;
        self.closing = response.header("connection").is_some_and(|c| c.eq_ignore_ascii_case("close"));
        if !matches!(response.status, 100..=199 | 204 | 304) {
            let mut drop_body = |_: &[u8]| Ok(());
            self.body(&response, if is_redirect(response.status) { &mut drop_body } else { sink })?;
        }
        Ok(response)
    }

    /// Hands the body of `response` to `sink`: chunked, by length, or up to the end of the connection.
    fn body(&mut self, response: &Response, sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        match (response.header("transfer-encoding"), response.length()) {
            (Some(te), _) if te.eq_ignore_ascii_case("chunked") => self.chunked(sink),
            (_, Some(len)) => self.exact(len, sink),
            _ => self.until_closed(sink),
        }
    }

    /// Reads and parses a response head, leaving what follows it in `buf`.
    fn head(&mut self) -> io::Result<Response> {
        let end = loop {
            if let Some(end) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break end;
            }
            if self.buf.len() > MAX_HEAD || self.fill()? == 0 {
                return Err(invalid("the server sent no complete response head"));
            }
        };
        let text = String::from_utf8_lossy(&self.buf[..end]).into_owned();
        self.buf.drain(..end + 4);
        let mut lines = text.split("\r\n");
        let status = lines.next().and_then(|l| l.split(' ').nth(1)).and_then(|s| s.parse().ok()).ok_or_else(|| invalid("a malformed status line"))?;
        let headers = lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string())).collect();
        Ok(Response { status, headers })
    }

    /// Reads more bytes into `buf`; how many (0 at the end of the connection).
    fn fill(&mut self) -> io::Result<usize> {
        let len = self.buf.len();
        self.buf.resize(len + (64 << 10), 0);
        let n = self.stream.read(&mut self.buf[len..]);
        self.buf.truncate(len + *n.as_ref().unwrap_or(&0));
        n
    }

    /// Hands `len` body bytes to `sink`.
    fn exact(&mut self, mut len: u64, sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        while len > 0 {
            if self.buf.is_empty() && self.fill()? == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the connection ended inside a body"));
            }
            let take = self.buf.len().min(usize::try_from(len).unwrap_or(usize::MAX));
            sink(&self.buf[..take])?;
            self.buf.drain(..take);
            len -= take as u64;
        }
        Ok(())
    }

    /// Hands a chunked body to `sink`.
    fn chunked(&mut self, sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        loop {
            let line = self.line()?;
            let size = u64::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16).map_err(|_| invalid("a malformed chunk size"))?;
            if size == 0 {
                while !self.line()?.is_empty() {}
                return Ok(());
            }
            self.exact(size, sink)?;
            self.line()?;
        }
    }

    /// The next line of `buf`, without its CRLF.
    fn line(&mut self) -> io::Result<String> {
        loop {
            if let Some(end) = self.buf.windows(2).position(|w| w == b"\r\n") {
                let line = String::from_utf8_lossy(&self.buf[..end]).into_owned();
                self.buf.drain(..end + 2);
                return Ok(line);
            }
            if self.buf.len() > MAX_HEAD || self.fill()? == 0 {
                return Err(invalid("a chunked body ended early"));
            }
        }
    }

    /// Hands everything up to the end of the connection to `sink`.
    fn until_closed(&mut self, sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        self.closing = true;
        loop {
            if !self.buf.is_empty() {
                sink(&self.buf)?;
                self.buf.clear();
            }
            if self.fill()? == 0 {
                return Ok(());
            }
        }
    }
}

fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("HTTP: {why}"))
}

/// Fetches `url` with `GET` and `headers`, following redirects, on `conn` when it serves the final host (a new
/// connection replaces it otherwise); the body goes to `sink`. Returns the final response and its URL. An
/// `Authorization` header goes to `url`'s host only: a redirect elsewhere (the Hub's CDN) does not carry it.
pub fn fetch(
    conn: &mut Option<Connection>,
    url: &Url,
    headers: &[(&str, &str)],
    config: &Arc<ClientConfig>,
    sink: &mut dyn FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<(Response, Url)> {
    let (mut url, origin) = (url.clone(), url.host.clone());
    for _ in 0..=MAX_REDIRECTS {
        let c = match conn {
            Some(c) if c.serves(&url) => c,
            _ => conn.insert(Connection::open(&url, config)?),
        };
        let sent: Vec<(&str, &str)> = headers.iter().copied().filter(|(k, _)| url.host == origin || !k.eq_ignore_ascii_case("authorization")).collect();
        let response = c.get(&url.path, &sent, sink)?;
        if !is_redirect(response.status) {
            return Ok((response, url));
        }
        url = response.header("location").and_then(|l| url.join(l)).ok_or_else(|| invalid("a redirect without a usable Location"))?;
    }
    Err(invalid("too many redirects"))
}

/// The body of `url`, fetched with `GET` and `headers`, following redirects; an error unless the status is 200.
pub fn get(url: &str, headers: &[(&str, &str)], config: &Arc<ClientConfig>) -> io::Result<Vec<u8>> {
    let url = Url::parse(url).ok_or_else(|| invalid("not an http or https URL"))?;
    let mut body = Vec::new();
    let (response, _) = fetch(&mut None, &url, headers, config, &mut |b| {
        body.extend_from_slice(b);
        Ok(())
    })?;
    match response.status {
        200 => Ok(body),
        status => Err(io::Error::other(format!("{} answered {status} for {}", url.host, url.path))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A server on a free port that answers each connection's requests in turn with `replies`.
    fn serve(replies: Vec<&'static str>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut replies = replies.into_iter();
            for sock in listener.incoming() {
                let mut sock = sock.unwrap();
                let mut seen = Vec::new();
                for reply in replies.by_ref() {
                    while seen.windows(4).filter(|w| *w == b"\r\n\r\n").count() == 0 {
                        let mut b = [0; 1024];
                        let n = sock.read(&mut b).unwrap();
                        seen.extend_from_slice(&b[..n]);
                    }
                    let end = seen.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                    seen.drain(..end + 4);
                    sock.write_all(reply.as_bytes()).unwrap();
                    if reply.contains("Connection: close") {
                        break;
                    }
                }
            }
        });
        port
    }

    fn config() -> Arc<ClientConfig> {
        Arc::new(ClientConfig { anchors: Vec::new(), random: super::super::crypto::random, now: || 0 })
    }

    #[test]
    fn urls_parse_and_join() {
        let u = Url::parse("https://huggingface.co/api/models/x?a=1").unwrap();
        assert_eq!((u.https, u.host.as_str(), u.port, u.path.as_str()), (true, "huggingface.co", 443, "/api/models/x?a=1"));
        assert_eq!(Url::parse("http://127.0.0.1:8080").unwrap().path, "/");
        assert_eq!(u.join("/other").unwrap().host, "huggingface.co");
        assert_eq!(u.join("https://cdn.test/f").unwrap().host, "cdn.test");
        assert!(Url::parse("ftp://x").is_none() && Url::parse("https://").is_none());
    }

    /// Bodies by length and chunked on one kept-alive connection, a redirect followed, and a body that runs to
    /// the end of the connection.
    #[test]
    fn bodies_redirects_and_keep_alive() {
        let port = serve(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=1\r\nde\r\n0\r\n\r\n",
            "HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 3\r\n\r\nxyz",
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 2\r\nContent-Range: bytes 4-5/10\r\n\r\n45",
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nto the end",
        ]);
        let (config, mut conn) = (config(), None);
        let url = Url::parse(&format!("http://127.0.0.1:{port}/a")).unwrap();
        let mut got = Vec::new();
        let mut take = |b: &[u8]| {
            got.extend_from_slice(b);
            Ok(())
        };
        for _ in 0..2 {
            fetch(&mut conn, &url, &[], &config, &mut take).unwrap();
        }
        let (response, last) = fetch(&mut conn, &url, &[("Range", "bytes=4-5")], &config, &mut take).unwrap();
        assert_eq!((response.status, last.path.as_str()), (206, "/final"));
        let (response, _) = fetch(&mut conn, &url, &[], &config, &mut take).unwrap();
        assert_eq!((response.status, String::from_utf8(got).unwrap()), (200, "helloabcde45to the end".to_string()));
        assert!(!conn.unwrap().serves(&url), "a closing connection is not reused");
    }
}
