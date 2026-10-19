//! Which Laya checkpoint reads a text best: `laya-english` reads English best (0.820 against 0.710 for the
//! multilingual checkpoint, by the `laya` package's benchmark) and collapses on other scripts; `laya-multilingual`
//! leads on the other 50 languages measured. A small rule in the spirit of the `laya` package's `Router`, not a port of
//! it: a text is not English when its letters are mostly outside the Latin script, when more than 2% of its Latin
//! letters are accented, or when it has at least two function words of one of six other Latin-script languages and
//! more than of English's, or when it runs to eight words or more without one of English's; any other text is, as the `laya` package also sends Latin text without such evidence to
//! the English checkpoint.

/// Function words, lower case: English's first, then Spanish, French, German, Portuguese, Italian and Dutch. Words
/// shared between English and another of them (`a`, `in`, `is`, `me`, `no`, `on`, `do`) are left out of English's.
const WORDS: [&[&str]; 7] = [
    &[
        "the", "and", "are", "was", "were", "to", "of", "for", "with", "that", "this", "it", "you", "have", "has", "not", "but", "at", "be", "from", "will",
        "can", "would", "there", "their", "what", "which", "please", "we", "i", "my", "your", "how", "does", "did", "get", "been", "our", "they", "if", "or",
        "any", "about", "just", "should", "could", "why", "when", "where",
    ],
    &["el", "la", "los", "las", "que", "de", "y", "en", "un", "una", "por", "para", "con", "mi", "es", "pero", "lo", "se", "su", "del", "al", "muy"],
    &["le", "la", "les", "et", "de", "des", "un", "une", "est", "je", "pas", "pour", "dans", "que", "qui", "vous", "mon", "ma", "avec", "sur", "du"],
    &["der", "die", "das", "und", "ist", "ich", "nicht", "ein", "eine", "mit", "zu", "den", "von", "sie", "es", "auf", "mein", "bitte", "dem", "wie"],
    &["o", "os", "as", "de", "e", "que", "um", "uma", "para", "com", "meu", "minha", "nao", "não", "em", "por", "se", "da", "do", "muito"],
    &["il", "lo", "la", "gli", "le", "di", "e", "che", "un", "una", "per", "con", "mio", "non", "sono", "del", "della", "ho", "ma", "si"],
    &["de", "het", "een", "en", "ik", "niet", "van", "met", "voor", "op", "mijn", "dat", "zijn", "maar", "ook", "je", "wij", "graag"],
];

/// Whether `text` reads as English, the `laya-english` checkpoint's language.
pub fn english(text: &str) -> bool {
    let latin = |c: char| c.is_ascii_alphabetic() || matches!(c as u32, 0xC0..=0x24F | 0x1E00..=0x1EFF);
    let letters = text.chars().filter(|c| c.is_alphabetic()).count().max(1);
    let (latin_letters, accented) = text.chars().filter(|&c| latin(c)).fold((0, 0), |(n, a), c| (n + 1, a + usize::from(!c.is_ascii())));
    if latin_letters * 5 < letters * 4 || accented * 50 > latin_letters {
        return false;
    }
    let words: Vec<String> = text.split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
    let hits: Vec<usize> = WORDS.iter().map(|list| words.iter().filter(|w| list.contains(&w.as_str())).count()).collect();
    let (ours, theirs) = (hits[0], hits[1..].iter().copied().max().unwrap_or(0));
    (theirs < 2 || theirs <= ours) && (ours > 0 || words.len() < 8)
}

#[cfg(test)]
mod tests {
    use super::english;

    #[test]
    fn english_goes_to_the_english_checkpoint_and_the_rest_does_not() {
        let yes = [
            "Hi, I was charged twice for my subscription this month. Please refund the second payment.",
            "how do I get a refund",
            "app keeps crashing",
            "URGENT: our production database is down and customers cannot log in. Need help immediately!!!",
            "{\"order\": 17, \"message\": \"The package arrived but two of the three items are missing.\"}",
        ];
        let no = [
            "Hola, no puedo iniciar sesión desde ayer. Me dice que mi contraseña es incorrecta pero la cambié.",
            "Hola, no puedo iniciar sesion desde ayer, me dice que mi contrasena es incorrecta pero la cambie",
            "Guten Tag, die Lieferung kam beschädigt an. Ich möchte eine Rückerstattung.",
            "Bonjour, ma facture contient une erreur : on m'a facturé 49 € au lieu de 29 €.",
            "Goedemorgen, mijn wachtwoord werkt niet meer en ik krijg geen resetmail.",
            "您好，我上个月的账单多扣了一笔费用，请帮我退款。",
            "Здравствуйте, мне дважды списали деньги за заказ.",
        ];
        assert!(yes.iter().all(|t| english(t)), "{:?}", yes.iter().find(|t| !english(t)));
        assert!(!no.iter().any(|t| english(t)), "{:?}", no.iter().find(|t| english(t)));
    }
}
