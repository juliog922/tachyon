//! Prefill attention on tensor cores: every token of a chunk against its window of the key-value cache, many
//! queries sharing each key and value they read.
//!
//! [`FLASH_256`] and [`FLASH_512`] take the queries as [`super::QUERY_256`] leaves them (`f16 [tokens][H][D]`,
//! and values already in). A block takes a tile of tokens × the four query heads of block row `ctaid.y` (all of
//! key-value head `ctaid.y / group`): 16 tokens for heads of 256, 8 for heads of 512, one row per token and head.
//! [`FLASH_256_PAIRS`] takes two query heads a block, for models whose key-value heads serve only two (Gemma 4's
//! larger models' sliding layers): 32 tokens a block.
//! It walks the keys the tile can see, from the oldest position its first token sees to its last token, a tile of
//! keys at a time (32 for heads of 256, 16 for 512): with `mma` (`m16n8k16`, `f16` in, `f32` sums) the scores `Q·Kᵀ`, masked to each
//! row's window (`p − window < j ≤ p`; tiles start at multiples of their size, so a token's sums do not depend on
//! the tile it is in), an online softmax in registers (scale 1, as Gemma 4 has it), and the
//! probabilities, rounded to `f16`, times the values (`ldmatrix.trans` reads them as the product needs). Two warps
//! share each 16 rows: both score all the tile's keys, each sums half of the head's width, so a thread holds a
//! quarter of a head's values per row. Keys and values are copied with `cp.async`, the values while the scores are
//! computed and the next keys while the values are summed. Blocks start from the last tile of tokens, whose
//! windows are longest. Out: `f32 [tokens][H·D]`, rows past `tokens` not written.
//!
//! Arguments: `q, k_cache, v_cache: u64`, `pos: *const u32` (the chunk's first position), `cap, window, group,
//! tokens: u32`, `y: u64`. Launch a grid of `(tokens.div_ceil(T), H / P)` blocks of [`Shape::threads`] with
//! [`shared`] bytes of dynamic shared memory, `T` = [`Shape::tokens`], `P` the heads a block takes (4, or 2 for
//! [`FLASH_256_PAIRS`]); the queries' buffer holds whole tiles.

use std::fmt::Write;

/// Name of the prefill attention on tensor cores, heads of 256.
pub const FLASH_256: &str = "flash_256";
/// Name of the prefill attention on tensor cores, heads of 512.
pub const FLASH_512: &str = "flash_512";
/// Name of the prefill attention on tensor cores, heads of 256, two query heads a block.
pub const FLASH_256_PAIRS: &str = "flash_256_2";

/// A block of [`FLASH_256`], [`FLASH_512`] or [`FLASH_256_PAIRS`]: head size, query heads, tiles of 16 rows, keys
/// per step.
#[derive(Clone, Copy)]
pub struct Shape {
    d: usize,
    per: usize,
    tiles: usize,
    keys: usize,
}

impl Shape {
    /// The shape of the kernel for heads of `d`, `per` (4 or 2) query heads a block: four tiles of rows and 32 keys
    /// a step for 256, two and 16 for 512, so the queries, keys and values fit in shared memory and a thread's sums
    /// in registers.
    #[must_use]
    pub const fn of(d: usize, per: u32) -> Shape {
        let per = per as usize;
        if d == 256 { Shape { d, per, tiles: 4, keys: 32 } } else { Shape { d, per, tiles: 2, keys: 16 } }
    }

    /// The kernel's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match (self.d, self.per) {
            (256, 2) => FLASH_256_PAIRS,
            (256, _) => FLASH_256,
            _ => FLASH_512,
        }
    }

    /// Tokens per block: a row per token and each of its heads.
    #[must_use]
    pub const fn tokens(self) -> u32 {
        (16 * self.tiles / self.per) as u32
    }

    /// Threads per block: two warps per tile of rows.
    #[must_use]
    pub const fn threads(self) -> u32 {
        64 * self.tiles as u32
    }

    /// Bytes of a row in shared memory, padded so `ldmatrix` is free of bank conflicts.
    const fn row(self) -> usize {
        2 * (self.d + 8)
    }

    /// Where the keys and the values start in shared memory, and the bytes in all.
    const fn layout(self) -> (usize, usize, usize) {
        let keys = 16 * self.tiles * self.row();
        let values = keys + self.keys * self.row();
        (keys, values, values + self.keys * self.row())
    }
}

/// The bytes of dynamic shared memory a flash kernel takes; 0 for any other kernel.
#[must_use]
pub fn shared(name: &str) -> u32 {
    match name {
        FLASH_256 | FLASH_256_PAIRS => Shape::of(256, 4).layout().2 as u32,
        FLASH_512 => Shape::of(512, 4).layout().2 as u32,
        _ => 0,
    }
}

/// The block, its rows and the range of keys: `%tok0` the tile's first token, `%hg` its heads' group, `%kvh` their
/// key-value head, `%pos0` the chunk's first position, `%lo..=%hi` the positions the tile sees; per thread `%rt` its
/// tile of rows, `%half` its half of a head, `%pr0`, `%pr1` the positions of its rows `g` and `g + 8`.
const HEAD: &str = "
    mov.u32 %th, %tid.x;
    and.b32 %lane, %th, 31;
    shr.u32 %warp, %th, 5;
    shr.u32 %rt, %warp, 1;
    and.b32 %half, %warp, 1;
    shr.u32 %g, %lane, 2;
    and.b32 %t, %lane, 3;
    mov.u32 %x, %nctaid.x;
    mov.u32 %y, %ctaid.x;
    sub.u32 %x, %x, %y;
    sub.u32 %x, %x, 1;
    mul.lo.u32 %tok0, %x, TOKENS;
    mov.u32 %hg, %ctaid.y;
    mov.u32 %nh, %nctaid.y;
    shl.b32 %nh, %nh, PSHIFT;
    ld.param.u32 %x, [pgroup];
    div.u32 %kvh, %hg, %x;
    ld.param.u32 %n, [ptok];
    ld.param.u32 %cap, [pcap];
    ld.param.u32 %win, [pwin];
    ld.param.u64 %a, [ppos];
    cvta.to.global.u64 %a, %a;
    ld.global.u32 %pos0, [%a];
    add.u32 %x, %pos0, %tok0;
    add.u32 %lo, %x, 1;
    sub.u32 %lo, %lo, %win;
    setp.lt.u32 %p, %x, %win;
    @%p mov.u32 %lo, 0;
    and.b32 %lo, %lo, KEYMASK;
    add.u32 %y, %tok0, TOKENS;
    min.u32 %y, %y, %n;
    add.u32 %hi, %pos0, %y;
    sub.u32 %hi, %hi, 1;
    mad.lo.u32 %x, %rt, 16, %g;
    shr.u32 %x, %x, PSHIFT;
    add.u32 %pr0, %pos0, %tok0;
    add.u32 %pr0, %pr0, %x;
    add.u32 %pr1, %pr0, RSTEP;
    ld.param.u64 %qp, [pq];
    cvta.to.global.u64 %qp, %qp;
    ld.param.u64 %kc, [pkc];
    cvta.to.global.u64 %kc, %kc;
    ld.param.u64 %vc, [pvc];
    cvta.to.global.u64 %vc, %vc;
    mov.u32 %sm, fsmem;
    and.b32 %col, %th, COLMASK;
    shr.u32 %r0, %th, COLSHIFT;
";

/// The fragments' addresses in shared memory: `%fq` this warp's rows of queries, `%fk` the keys, `%fv` this warp's
/// half of the values (read transposed).
fn fragments(s: &mut String, sh: Shape) {
    let (keys, values, _) = sh.layout();
    let row = sh.row();
    let _ = writeln!(s, "    and.b32 %x, %lane, 15;\n    mad.lo.u32 %x, %rt, 16, %x;\n    mad.lo.u32 %fq, %x, {row}, %sm;");
    s.push_str("    shr.u32 %x, %lane, 4;\n    mad.lo.u32 %fq, %x, 16, %fq;\n");
    let _ = writeln!(s, "    and.b32 %x, %lane, 7;\n    shr.u32 %y, %lane, 4;\n    mad.lo.u32 %x, %y, 8, %x;\n    mad.lo.u32 %fk, %x, {row}, %sm;");
    let _ = writeln!(s, "    add.u32 %fk, %fk, {keys};\n    shr.u32 %x, %lane, 3;\n    and.b32 %x, %x, 1;\n    mad.lo.u32 %fk, %x, 16, %fk;");
    let _ = writeln!(
        s,
        "    and.b32 %x, %lane, 7;\n    shr.u32 %y, %lane, 3;\n    and.b32 %y, %y, 1;\n    mad.lo.u32 %x, %y, 8, %x;\n    mad.lo.u32 %fv, %x, {row}, %sm;"
    );
    let _ = writeln!(s, "    add.u32 %fv, %fv, {values};\n    mul.lo.u32 %x, %half, {};\n    add.u32 %fv, %fv, %x;", sh.d);
    s.push_str("    shr.u32 %x, %lane, 4;\n    mad.lo.u32 %fv, %x, 16, %fv;\n");
}

/// Queues the copy of this thread's share of the queries: column `%col` (16 bytes) of rows `%r0 + step·i`, row
/// `r` being token `%tok0 + r / P` and head `P·%hg + r % P`, `P` the heads a block takes.
fn queries(s: &mut String, sh: Shape) {
    let (step, rows) = (sh.threads() as usize / (sh.d / 8), 16 * sh.tiles);
    for i in 0..rows / step {
        let (shift, per) = (sh.per.trailing_zeros(), sh.per);
        let _ = writeln!(s, "    add.u32 %x, %r0, {};\n    shr.u32 %y, %x, {shift};\n    add.u32 %y, %y, %tok0;\n    and.b32 %j, %x, {};", step * i, per - 1);
        let _ = writeln!(
            s,
            "    mad.lo.u32 %y, %y, %nh, %j;\n    mad.lo.u32 %y, %hg, {per}, %y;\n    mul.lo.u32 %y, %y, {};\n    mad.lo.u32 %y, %col, 8, %y;",
            sh.d
        );
        let _ =
            writeln!(s, "    mul.wide.u32 %o, %y, 2;\n    add.u64 %a, %qp, %o;\n    mad.lo.u32 %x, %x, {}, %sm;\n    mad.lo.u32 %x, %col, 16, %x;", sh.row());
        s.push_str("    cp.async.cg.shared.global [%x], [%a], 16;\n");
    }
    s.push_str("    cp.async.commit_group;\n");
}

/// Queues the copy of this thread's share of the keys (or values, `cache` `%vc`) at positions `%jb..` into the
/// buffer at `at`: column `%col` of keys `%r0 + step·i`, each from its slot in the ring.
fn tile(s: &mut String, sh: Shape, cache: &str, at: usize) {
    let step = sh.threads() as usize / (sh.d / 8);
    for i in 0..sh.keys / step {
        let _ = writeln!(s, "    add.u32 %x, %r0, {};\n    add.u32 %y, %jb, %x;\n    rem.u32 %y, %y, %cap;\n    mad.lo.u32 %y, %kvh, %cap, %y;", step * i);
        let _ = writeln!(s, "    mul.lo.u32 %y, %y, {};\n    mad.lo.u32 %y, %col, 8, %y;\n    mul.wide.u32 %o, %y, 2;\n    add.u64 %a, {cache}, %o;", sh.d);
        let _ = writeln!(s, "    mad.lo.u32 %x, %x, {}, %sm;\n    mad.lo.u32 %x, %col, 16, %x;\n    cp.async.cg.shared.global [%x+{at}], [%a], 16;", sh.row());
    }
    s.push_str("    cp.async.commit_group;\n");
}

/// The scores of this warp's 16 rows against the tile's keys, into `%s` (n-tile `n` in `%s{4n}..`).
fn scores(s: &mut String, sh: Shape) {
    (0..sh.keys / 2).for_each(|i| _ = writeln!(s, "    mov.f32 %s{i}, 0f00000000;"));
    let mma = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
    for ks in 0..sh.d / 16 {
        let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%qa0, %qa1, %qa2, %qa3}}, [%fq+{}];", 32 * ks);
        for np in 0..sh.keys / 16 {
            let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%kb0, %kb1, %kb2, %kb3}}, [%fk+{}];", 32 * ks + 16 * np * sh.row());
            for (n, b) in [(2 * np, 0), (2 * np + 1, 2)] {
                let c = 4 * n;
                let _ = writeln!(
                    s,
                    "    {mma} {{%s{c}, %s{}, %s{}, %s{}}}, {{%qa0, %qa1, %qa2, %qa3}}, {{%kb{b}, %kb{}}}, {{%s{c}, %s{}, %s{}, %s{}}};",
                    c + 1,
                    c + 2,
                    c + 3,
                    b + 1,
                    c + 1,
                    c + 2,
                    c + 3
                );
            }
        }
    }
}

/// Masks the scores to each row's window, then the online softmax: new maxima `%m`, the factor `%al` the sums so
/// far shrink by, the probabilities as `f16` pairs in `%pa` (the `mma` A fragments of the values' product), and the
/// thread's share of each row's sum in `%l`.
fn softmax(s: &mut String, sh: Shape) {
    s.push_str("    mad.lo.u32 %jt, %t, 2, %jb;\n");
    for n in 0..sh.keys / 8 {
        for e in 0..4 {
            let (row, i) = (e / 2, 4 * n + e);
            let _ = writeln!(
                s,
                "    add.u32 %y, %jt, {};\n    sub.u32 %x, %pr{row}, %y;\n    setp.ge.u32 %p, %x, %win;\n    @%p mov.f32 %s{i}, {NEG_INF};",
                8 * n + e % 2
            );
        }
    }
    for row in 0..2 {
        let _ = writeln!(s, "    mov.f32 %mx, %m{row};");
        (0..sh.keys / 8).flat_map(|n| [4 * n + 2 * row, 4 * n + 2 * row + 1]).for_each(|i| _ = writeln!(s, "    max.f32 %mx, %mx, %s{i};"));
        s.push_str(
            "    shfl.sync.bfly.b32 %f, %mx, 1, 31, -1;\n    max.f32 %mx, %mx, %f;\n    shfl.sync.bfly.b32 %f, %mx, 2, 31, -1;\n    max.f32 %mx, %mx, %f;\n",
        );
        // A row that sees no key yet keeps −∞: its probabilities and factor come out 0 against a maximum of 0. A
        // maximum that did not change keeps the factor at exactly 1, so a tile of keys a row does not see changes
        // nothing in it, and a token's sums do not depend on the chunk it came in.
        let _ = writeln!(s, "    setp.eq.f32 %p, %mx, {NEG_INF};\n    selp.f32 %mu, 0f00000000, %mx, %p;\n    mul.f32 %mu, %mu, {LOG2E_NEG};");
        let _ = writeln!(s, "    fma.rn.f32 %f, %m{row}, {LOG2E}, %mu;\n    ex2.approx.f32 %al{row}, %f;\n    setp.eq.f32 %p, %mx, %m{row};");
        let _ = writeln!(s, "    @%p mov.f32 %al{row}, 0f3F800000;\n    mov.f32 %m{row}, %mx;\n    mul.f32 %l{row}, %l{row}, %al{row};");
        for i in (0..sh.keys / 8).flat_map(|n| [4 * n + 2 * row, 4 * n + 2 * row + 1]) {
            let _ = writeln!(s, "    fma.rn.f32 %f, %s{i}, {LOG2E}, %mu;\n    ex2.approx.f32 %s{i}, %f;\n    add.f32 %l{row}, %l{row}, %s{i};");
        }
    }
    for k in 0..sh.keys / 16 {
        for (r, i) in [(0, 8 * k), (1, 8 * k + 2), (2, 8 * k + 4), (3, 8 * k + 6)] {
            let _ = writeln!(s, "    cvt.rn.f16x2.f32 %pa{}, %s{}, %s{i};", 4 * k + r, i + 1);
        }
    }
}

/// This warp's sums scaled by `%al`, then the probabilities times its half of the values (n-tile `n` in `%o{4n}..`).
fn values(s: &mut String, sh: Shape) {
    for n in 0..sh.d / 16 {
        let _ = writeln!(s, "    mul.f32 %o{0}, %o{0}, %al0;\n    mul.f32 %o{1}, %o{1}, %al0;", 4 * n, 4 * n + 1);
        let _ = writeln!(s, "    mul.f32 %o{0}, %o{0}, %al1;\n    mul.f32 %o{1}, %o{1}, %al1;", 4 * n + 2, 4 * n + 3);
    }
    let mma = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
    for k in 0..sh.keys / 16 {
        let a = 4 * k;
        for np in 0..sh.d / 32 {
            let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%vb0, %vb1, %vb2, %vb3}}, [%fv+{}];", 16 * k * sh.row() + 32 * np);
            for (n, b) in [(2 * np, 0), (2 * np + 1, 2)] {
                let c = 4 * n;
                let _ = writeln!(
                    s,
                    "    {mma} {{%o{c}, %o{}, %o{}, %o{}}}, {{%pa{a}, %pa{}, %pa{}, %pa{}}}, {{%vb{b}, %vb{}}}, {{%o{c}, %o{}, %o{}, %o{}}};",
                    c + 1,
                    c + 2,
                    c + 3,
                    a + 1,
                    a + 2,
                    a + 3,
                    b + 1,
                    c + 1,
                    c + 2,
                    c + 3
                );
            }
        }
    }
}

/// Divides the sums by each row's total and stores this warp's half of rows `g` and `g + 8`, those of tokens below
/// `tokens`.
fn store(s: &mut String, sh: Shape) {
    s.push_str("    ld.param.u64 %a, [py];\n    cvta.to.global.u64 %a, %a;\n");
    for row in 0..2 {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %f, %l{row}, 1, 31, -1;\n    add.f32 %l{row}, %l{row}, %f;");
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %f, %l{row}, 2, 31, -1;\n    add.f32 %l{row}, %l{row}, %f;\n    rcp.rn.f32 %l{row}, %l{row};");
        // Row `r` of the block is token `%tok0 + r / P`, head `P·%hg + r % P`; row `g + 8` is `8 / P` tokens on.
        let (shift, per) = (sh.per.trailing_zeros(), sh.per);
        let _ = writeln!(s, "    mad.lo.u32 %x, %rt, 16, %g;\n    add.u32 %x, %x, {};\n    shr.u32 %y, %x, {shift};\n    add.u32 %y, %y, %tok0;", 8 * row);
        let _ =
            writeln!(s, "    setp.lt.u32 %p, %y, %n;\n    and.b32 %j, %x, {};\n    mad.lo.u32 %y, %y, %nh, %j;\n    mad.lo.u32 %y, %hg, {per}, %y;", per - 1);
        let _ = writeln!(s, "    mul.lo.u32 %y, %y, {};\n    mad.lo.u32 %y, %half, {}, %y;\n    mad.lo.u32 %y, %t, 2, %y;", sh.d, sh.d / 2);
        s.push_str("    mul.wide.u32 %o, %y, 4;\n    add.u64 %b, %a, %o;\n");
        for n in 0..sh.d / 16 {
            let c = 4 * n + 2 * row;
            let _ = writeln!(s, "    mul.f32 %f, %o{c}, %l{row};\n    mul.f32 %mx, %o{}, %l{row};", c + 1);
            let _ = writeln!(s, "    @%p st.global.v2.f32 [%b+{}], {{%f, %mx}};", 32 * n);
        }
    }
}

const LOG2E: &str = "0f3FB8AA3B";
const LOG2E_NEG: &str = "0fBFB8AA3B";
const NEG_INF: &str = "0fFF800000";

/// The dynamic shared memory every flash kernel uses, declared once in the module: `ptxas` refuses a second
/// declaration.
pub(super) const FSMEM: &str = ".extern .shared .align 16 .b8 fsmem[];\n";

/// The flash kernel for heads of `d`, `per` query heads a block; it needs [`FSMEM`] declared before it.
pub(super) fn flash(d: usize, per: u32) -> String {
    let sh = Shape::of(d, per);
    let (k_at, v_at, _) = sh.layout();
    let mut s = format!(
        ".visible .entry {}(.param .u64 pq, .param .u64 pkc, .param .u64 pvc, .param .u64 ppos, .param .u32 pcap, .param .u32 pwin,
    .param .u32 pgroup, .param .u32 ptok, .param .u64 py)
    .maxntid {}, 1, 1
{{
    .reg .pred %p;
    .reg .b32 %th, %lane, %warp, %rt, %half, %g, %t, %x, %y, %j, %jb, %jt, %tok0, %hg, %nh, %kvh, %n, %cap, %win, %pos0, %lo, %hi;
    .reg .b32 %pr0, %pr1, %sm, %col, %r0, %fq, %fk, %fv, %qa<4>, %kb<4>, %vb<4>, %pa<{}>;
    .reg .f32 %s<{}>, %o<{}>, %m<2>, %l<2>, %al<2>, %mx, %mu, %f;
    .reg .b64 %qp, %kc, %vc, %a, %b, %o;",
        sh.name(),
        sh.threads(),
        sh.keys / 4,
        sh.keys / 2,
        sh.d / 4
    );
    let cols = sh.d / 8;
    s += &HEAD
        .replace("TOKENS", &sh.tokens().to_string())
        .replace("PSHIFT", &sh.per.trailing_zeros().to_string())
        .replace("RSTEP", &(8 / sh.per).to_string())
        .replace("COLMASK", &(cols - 1).to_string())
        .replace("COLSHIFT", &cols.trailing_zeros().to_string())
        .replace("KEYMASK", &format!("0x{:08X}", !(sh.keys as u32 - 1)));
    fragments(&mut s, sh);
    for row in 0..2 {
        let _ = writeln!(s, "    mov.f32 %m{row}, {NEG_INF};\n    mov.f32 %l{row}, 0f00000000;");
    }
    (0..sh.d / 4).for_each(|i| _ = writeln!(s, "    mov.f32 %o{i}, 0f00000000;"));
    queries(&mut s, sh);
    s += "    mov.u32 %jb, %lo;\n";
    tile(&mut s, sh, "%kc", k_at);
    s += "KEYS:\n";
    tile(&mut s, sh, "%vc", v_at);
    s += "    cp.async.wait_group 1;\n    bar.sync 0;\n";
    scores(&mut s, sh);
    softmax(&mut s, sh);
    let _ = writeln!(s, "    cp.async.wait_group 0;\n    bar.sync 0;\n    add.u32 %jb, %jb, {};\n    setp.le.u32 %p, %jb, %hi;\n    @%p bra NEXT;", sh.keys);
    s += "    cp.async.commit_group;\n    bra SUM;\nNEXT:\n";
    tile(&mut s, sh, "%kc", k_at);
    s += "SUM:\n";
    values(&mut s, sh);
    s += "    bar.sync 0;\n    setp.le.u32 %p, %jb, %hi;\n    @%p bra KEYS;\n";
    store(&mut s, sh);
    s + "    ret;\n}\n"
}
