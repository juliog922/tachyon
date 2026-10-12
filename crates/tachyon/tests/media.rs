//! Images against PIL: every file of `tests/fixtures/media/` (written by `scripts/media.py images`) decoded to the
//! same RGB as PIL's `exif_transpose(open(…)).convert("RGB")`, and resized to the same size and pixels as Gemma 4's
//! image processor gives it for each soft-token budget; the files PIL reads that Tachyon does not, refused.

use tachyon::json::Json;
use tachyon::media::{self, Image};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/media");

/// FNV-1a 64, as the fixture's hashes.
fn fnv(data: &[u8]) -> String {
    format!("{:016x}", data.iter().fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)))
}

fn cases() -> Vec<Json> {
    std::fs::read_to_string(format!("{DIR}/images.ref")).unwrap().lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect()
}

fn file(case: &Json) -> (&str, Vec<u8>) {
    let name = case.get("file").and_then(Json::str).unwrap();
    (name, std::fs::read(format!("{DIR}/{name}")).unwrap())
}

/// What is wrong with Tachyon's decoding of `case`, if anything.
fn mismatch(case: &Json) -> Option<String> {
    let (name, bytes) = file(case);
    let refused = case.get("refused").is_some() || case.get("error").is_some();
    match (media::decode(&bytes), refused) {
        (Err(e), false) => Some(format!("{name}: {e}")),
        (Ok(_), true) => Some(format!("{name}: decoded, but Tachyon refuses it")),
        (Err(_), true) => None,
        (Ok(img), false) => {
            let size = [case.get("width"), case.get("height")].map(|v| v.and_then(Json::num).unwrap() as usize);
            let same = fnv(&img.rgb) == case.get("fnv").and_then(Json::str).unwrap() || case.get("clipped").is_some();
            ([img.width, img.height] != size || !same).then(|| format!("{name}: {}×{} decoded, its pixels differ from PIL's", img.width, img.height))
        }
    }
}

#[test]
fn images_decode_as_pil_decodes_them() {
    let cases = cases();
    let wrong: Vec<String> = cases.iter().filter_map(mismatch).collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert!(cases.iter().filter(|c| c.get("fnv").is_some()).count() >= 45, "too few files");
}

#[test]
fn images_resize_as_gemma_4_processor_does() {
    for case in cases().iter().filter(|c| c.get("fitted").is_some() && c.get("refused").is_none() && c.get("clipped").is_none()) {
        let (name, bytes) = file(case);
        let img = media::decode(&bytes).unwrap();
        for (budget, want) in case.get("fitted").and_then(Json::obj).unwrap() {
            let want = want.arr().unwrap();
            let (w, h) = img.fitted_size(budget.parse().unwrap()).unwrap();
            assert_eq!([w, h], [0, 1].map(|i| want[i].num().unwrap() as usize), "{name} at {budget}: size");
            let out = img.resize(w, h);
            assert_eq!(fnv(&out.rgb), want[2].str().unwrap(), "{name} at {budget}: pixels differ from PIL's bicubic");
        }
    }
}

#[test]
fn orientations_turn_the_image() {
    let img = Image { width: 3, height: 2, rgb: (0..18).collect() };
    let px = |i: &Image, x: usize, y: usize| i.rgb[(y * i.width + x) * 3];
    let turned = img.clone().oriented(6);
    assert_eq!((turned.width, turned.height), (2, 3));
    // Clockwise: the bottom-left pixel comes to the top left.
    assert_eq!(px(&turned, 0, 0), px(&img, 0, 1));
    assert_eq!(img.clone().oriented(8).oriented(6), img);
    assert_eq!(img.clone().oriented(3).oriented(3), img);
}

#[test]
fn damaged_and_unknown_files_are_errors() {
    let (_, png) = file(&cases().into_iter().find(|c| c.get("file").and_then(Json::str) == Some("rgb8.png")).unwrap());
    for cut in [8, 30, png.len() / 2, png.len() - 13] {
        assert!(media::decode(&png[..cut]).is_err(), "a PNG cut at {cut} decoded");
    }
    assert!(media::decode(b"GIF89a").is_err());
    assert!(media::decode(&[]).is_err());
}
