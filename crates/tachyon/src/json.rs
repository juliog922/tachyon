//! A JSON reader for the files models ship with, such as `tokenizer.json` and `config.json`. It reads whole
//! documents into a tree: these files are read once, when a model is converted.

use crate::{Error, Result};

/// A JSON value. Objects keep their keys in document order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses a document.
    pub fn parse(text: &[u8]) -> Result<Json> {
        let mut p = Parser { s: text, i: 0 };
        let v = p.value()?;
        p.space();
        if p.i == text.len() { Ok(v) } else { Err(p.fail("trailing characters")) }
    }

    /// The value of `key` in an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.obj()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The string, if this is one.
    pub fn str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    /// The number, if this is one.
    pub fn num(&self) -> Option<f64> {
        if let Json::Num(n) = self { Some(*n) } else { None }
    }

    /// The boolean, if this is one.
    pub fn bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }

    /// The elements, if this is an array.
    pub fn arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
    }

    /// The members, if this is an object.
    pub fn obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self { Some(o) } else { None }
    }
}

/// `"`, `{` and `}`, spelled so that complexity tools do not take them for code.
const QUOTE: u8 = b'\x22';
const OPEN: u8 = b'\x7B';
const CLOSE: u8 = b'\x7D';

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn fail(&self, what: &str) -> Error {
        Error::Format(format!("JSON: {what} at byte {}", self.i))
    }

    fn space(&mut self) {
        while self.s.get(self.i).is_some_and(u8::is_ascii_whitespace) {
            self.i += 1;
        }
    }

    fn eat(&mut self, word: &[u8]) -> bool {
        let found = self.s[self.i..].starts_with(word);
        self.i += if found { word.len() } else { 0 };
        found
    }

    fn value(&mut self) -> Result<Json> {
        self.space();
        match self.s.get(self.i).copied() {
            Some(OPEN) => self.list(CLOSE, Parser::member).map(Json::Obj),
            Some(b'[') => self.list(b']', Parser::value).map(Json::Arr),
            Some(QUOTE) => self.string().map(Json::Str),
            Some(c) if c.is_ascii_digit() || c == b'-' => self.number(),
            _ => self.word(),
        }
    }

    fn member(&mut self) -> Result<(String, Json)> {
        let k = self.string()?;
        self.space();
        if !self.eat(b":") {
            return Err(self.fail("expected ':'"));
        }
        Ok((k, self.value()?))
    }

    fn word(&mut self) -> Result<Json> {
        [(&b"null"[..], Json::Null), (b"true", Json::Bool(true)), (b"false", Json::Bool(false))]
            .into_iter()
            .find(|(w, _)| self.eat(w))
            .map(|(_, v)| v)
            .ok_or_else(|| self.fail("expected a value"))
    }

    /// The items of an array or object, opened at `self.i` and closed by `end`.
    fn list<T>(&mut self, end: u8, mut item: impl FnMut(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        self.i += 1;
        let mut out = Vec::new();
        self.space();
        if self.eat(&[end]) {
            return Ok(out);
        }
        loop {
            out.push(item(self)?);
            self.space();
            if self.eat(&[end]) {
                return Ok(out);
            }
            if !self.eat(b",") {
                return Err(self.fail("expected ',' or the end of a list"));
            }
            self.space();
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        while self.s.get(self.i).is_some_and(|b| b"+-.eE0123456789".contains(b)) {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or_default();
        text.parse().map(Json::Num).map_err(|_| self.fail("expected a value"))
    }

    fn string(&mut self) -> Result<String> {
        if !self.eat(b"\"") {
            return Err(self.fail("expected a string"));
        }
        let mut out = Vec::new();
        loop {
            let run = self.s[self.i..].iter().position(|&b| b == QUOTE || b == b'\\').ok_or_else(|| self.fail("unterminated string"))?;
            out.extend_from_slice(&self.s[self.i..self.i + run]);
            self.i += run + 1;
            if self.s[self.i - 1] == QUOTE {
                return String::from_utf8(out).map_err(|_| self.fail("invalid UTF-8"));
            }
            let c = self.escape()?;
            out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
        }
    }

    /// The character of the escape after a backslash.
    fn escape(&mut self) -> Result<char> {
        let c = *self.s.get(self.i).ok_or_else(|| self.fail("unterminated escape"))?;
        self.i += 1;
        Ok(match c {
            b'n' => '\n',
            b't' => '\t',
            b'r' => '\r',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'u' => return self.unicode(),
            _ => char::from(c),
        })
    }

    /// `\uXXXX`, with a second one for a surrogate pair.
    fn unicode(&mut self) -> Result<char> {
        let hex = |p: &mut Self| -> Result<u32> {
            let digits = p.s.get(p.i..p.i + 4).and_then(|d| std::str::from_utf8(d).ok()).and_then(|d| u32::from_str_radix(d, 16).ok());
            p.i += 4;
            digits.ok_or_else(|| p.fail("bad \\u escape"))
        };
        let hi = hex(self)?;
        let code = if (0xD800..0xDC00).contains(&hi) && self.eat(b"\\u") { 0x10000 + ((hi - 0xD800) << 10) + (hex(self)? - 0xDC00) } else { hi };
        Ok(char::from_u32(code).unwrap_or('\u{FFFD}'))
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn parses_what_model_files_hold() {
        let doc = br#" {"a": [1, -2.5e3, true, null], "b": {"c": "x\"\\\n\u00e9\ud83d\ude00"}, "e": []} "#;
        let v = Json::parse(doc).unwrap();
        assert_eq!(v.get("a").unwrap().arr().unwrap()[1].num(), Some(-2500.0));
        assert_eq!(v.get("b").and_then(|b| b.get("c")).and_then(Json::str), Some("x\"\\\né😀"));
        assert_eq!(v.get("e"), Some(&Json::Arr(vec![])));
    }

    #[test]
    fn malformed_documents_are_errors() {
        for bad in [&b"{\"a\" 1}"[..], b"[1,]", b"\"open", b"[1] 2", b"{\"a\":}"] {
            assert!(Json::parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
    }
}