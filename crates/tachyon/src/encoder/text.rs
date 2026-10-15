//! Text in, answers out: Laya's decisions about a state, Granite's embeddings of texts.

use super::Encoder;
use super::laya::{Budget, Builder, Frame, Question, probabilities};
use crate::token::Tokenizer;
use crate::{Error, Result};

impl Encoder {
    /// Laya's answer to each of `questions` about `state`: the probability of each option, in order (for a
    /// [`super::laya::Kind::Noul`], false then true). A `conversation`, newest last, keeps its end when it does not
    /// fit; other states keep their start. Each question is one sequence, as the `laya` package builds it.
    pub fn decide(&mut self, state: &str, conversation: bool, questions: &[Question]) -> Result<Vec<Vec<f32>>> {
        let head = self.head().ok_or_else(|| Error::Unsupported("decisions with a model that has no Laya head".into()))?.clone();
        let seqs = self.sequences(state, conversation, questions, head.budget)?;
        let items: Vec<_> = seqs.iter().zip(questions).map(|((ids, marks), q)| (ids.as_slice(), marks.as_slice(), q.kind)).collect();
        let logits = self.logits(&items)?;
        Ok(logits.iter().zip(questions).map(|(l, q)| probabilities(l, head.temperature(q.kind, l.len()))).collect())
    }

    /// Each question's sequence and its markers' positions, within `budget`.
    fn sequences(&self, state: &str, conversation: bool, questions: &[Question], budget: Budget) -> Result<Vec<(Vec<u32>, Vec<usize>)>> {
        let tok = self.tokenizer().ok_or_else(|| Error::Format("the model directory has no tokenizer.tok".into()))?;
        let frame = Frame { cls: special(tok, &["<bos>", "[CLS]"])?, sep: special(tok, &["<eos>", "[SEP]"])?, mask: special(tok, &["<mask>", "[MASK]"])? };
        let builder = Builder { tok, frame, budget };
        let state = builder.tokens(state);
        questions
            .iter()
            .map(|q| {
                let mut ids = Vec::new();
                builder.sequence(q, &state, conversation, &mut ids).map(|marks| (ids, marks))
            })
            .collect()
    }

    /// Granite's embedding of each of `texts` (its first token's last state, to unit length), cut to its first
    /// `dims` values and back to unit length if asked (Matryoshka: Granite 311M is trained for 768, 512, 384, 256 and
    /// 128). Each text is framed as its tokenizer's post-processor frames it (`<bos>` before, for a tokenizer that
    /// says nothing). A text longer than the encoder's capacity is cut, as sentence-transformers does at its limit,
    /// keeping the token after it.
    pub fn embed(&mut self, texts: &[&str], dims: Option<usize>) -> Result<Vec<Vec<f32>>> {
        let seqs: Vec<Vec<u32>> = {
            let tok = self.tokenizer().ok_or_else(|| Error::Format("the model directory has no tokenizer.tok".into()))?;
            let [before, after] = tok.frame();
            let (cls, cap) = (before.map_or_else(|| special(tok, &["<bos>", "[CLS]"]), Ok)?, self.capacity());
            texts
                .iter()
                .map(|t| {
                    let mut ids = vec![cls];
                    tok.encode(t, &mut ids);
                    ids.truncate(cap - usize::from(after.is_some()));
                    ids.extend(after);
                    ids
                })
                .collect()
        };
        let mut out = self.embeddings(&seqs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        for e in out.iter_mut().filter(|_| dims.is_some()) {
            e.truncate(dims.unwrap_or(usize::MAX));
            let len = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in e.iter_mut() {
                *v /= len;
            }
        }
        Ok(out)
    }
}

/// The first of `names` that is a special token of `tok`.
fn special(tok: &Tokenizer, names: &[&str]) -> Result<u32> {
    names.iter().find_map(|n| tok.special(n)).ok_or_else(|| Error::Format(format!("the tokenizer has none of {names:?}")))
}
