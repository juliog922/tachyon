//! [`GEMM_Q4`], the prefill matrix product on tensor cores: Q4 weights times the Q8P activations of many tokens,
//! `f32` out. [`MMA_PEAK`] measures the ceiling it is judged against.
//!
//! Integer `mma.sync` (`m16n8k32`, signed activations × unsigned weight nibbles, `s32` sums) at twice the rate of
//! the `f16` tensor path. The activations are Q8P ([`crate::quant::q8p`]): one scale per 64 values, the span of a
//! Q4 group, so the two `mma` of a group chain their sums in integers and the float work is one step per group.
//! Lane `t` of a quad takes 16 activation bytes at `16t` and weight words `2t, 2t+1` (one 16- and one 8-byte load,
//! the layouts of [`crate::quant`] as they are): the first `mma` reads bytes `0..8` against word `2t`, its low
//! nibbles as `k` slots `4t..4t+3` and its high nibbles as `16+4t..`, the second bytes `8..16` against word `2t+1`.
//! Each chain starts from the integer `0x4B400000`, so its sum `i` reads as the float [`crate::quant::MAGIC`]` + i`;
//! the Q8P second word `z = −8·s·Σq − MAGIC·s` turns it into `s·Σ(n − 8)·q` with one `fma`, and one more applies
//! the weight scale `d`: two per output per group. `z`'s rounding costs about a unit of `i` per group.
//!
//! A block of 4 warps computes [`GEMM_TOKENS`] tokens × [`GEMM_ROWS`] rows, each warp all 128 tokens (8 `mma` row
//! tiles) × 16 rows (2 column tiles), so each weight fragment is unpacked once per block. The operands of one Q4
//! group are staged in shared memory by `cp.async`, two stages deep, so the next group's copy overlaps this
//! group's `mma`; every fragment load is free of bank conflicts. Blocks of consecutive tokens, launched together,
//! share the weights in L2. Arguments: `w, ws` (Q4 `rows × cols`), Q8P in (`tokens × cols`), `y: *mut f32`
//! (`tokens × rows`), `rows, cols, tokens: u32`. `rows` is a multiple of [`GEMM_ROWS`], `cols` of 128; the Q8P
//! buffer holds whole tiles of [`GEMM_TOKENS`] rows (only `y` rows below `tokens` are written). Launch a grid of
//! `(tokens.div_ceil(GEMM_TOKENS), rows / GEMM_ROWS, slices)` blocks of 128 threads: with more than one slice
//! ([`slices`]), block `z` sums its share of the columns into `y`'s slice `z` (`y` then holds `slices` of
//! `tokens × rows`), and [`SUM_SLICES`] adds them up, in order, so the sums do not depend on timing.

// Indices keep their one-letter names from the mma fragments: m, n tiles, e element, j half.
#![allow(clippy::many_single_char_names)]

use std::fmt::Write;

/// Name of the Q4 × Q8P matrix product.
pub const GEMM_Q4: &str = "gemm_q4";
/// Tokens per [`GEMM_Q4`] block.
pub const GEMM_TOKENS: u32 = 128;
/// Rows per [`GEMM_Q4`] block.
pub const GEMM_ROWS: u32 = 64;
/// Name of the kernel that adds the slices of a split [`GEMM_Q4`]: `y[i] = Σ y[z·len + i]` over `z < slices`, four
/// values per thread. Arguments: `y: *mut f32`, `len, slices: u32`; `len` a multiple of 4. Launch
/// `len.div_ceil(1024)` blocks of 256 threads.
pub const SUM_SLICES: &str = "sum_slices";

/// The slices of the columns a [`GEMM_Q4`] of `rows × cols` over `tokens` takes on a GPU of `sms`
/// multiprocessors, with room in `y` for `room` slices: as many as keep its blocks within the three per
/// multiprocessor that run at once, eight at most and 512 columns each at least (a slice of fewer spends more on
/// its store and the sum than it saves), so short and narrow products fill the GPU. They follow the
/// tokens, so a product's sums can differ in their last bits between batches of different lengths.
#[must_use]
pub fn slices(rows: u32, cols: u32, tokens: u32, sms: u32, room: u32) -> u32 {
    let blocks = tokens.div_ceil(GEMM_TOKENS) * (rows / GEMM_ROWS);
    (3 * sms / blocks.max(1)).min(room).min(cols / 512).clamp(1, 8)
}
/// Name of the kernel that runs `iters` rounds of 8 independent integer `mma` per warp and writes one word per
/// thread to `out`. Arguments: `out: *mut u32`, `iters: u32`. Each `mma` is 8,192 operations.
pub const MMA_PEAK: &str = "mma_peak";
/// [`MMA_PEAK`] for `f16` (`m16n8k16`, 4,096 operations each) with `f32` sums, as the encoder's product runs.
pub const MMA_PEAK_F16: &str = "mma_peak_f16";
/// [`MMA_PEAK`] for `f16` with `f16` sums: on consumer Ampere twice the rate of `f32` sums.
pub const MMA_PEAK_F16H: &str = "mma_peak_f16h";
/// Each peak kernel: its name, its `mma` and the accumulator registers of one.
pub(super) const PEAKS: [(&str, &str, usize); 3] = [
    (MMA_PEAK, "m16n8k32.row.col.s32.s8.u8.s32", 4),
    (MMA_PEAK_F16, "m16n8k16.row.col.f32.f16.f16.f32", 4),
    (MMA_PEAK_F16H, "m16n8k16.row.col.f16.f16.f16.f16", 2),
];

/// Where each part of a shared-memory stage starts: 128 tokens × 64 Q8 bytes, their 128 `[s, z]`, 64 weight rows
/// × 32 bytes, and the rows' scales (the 4 bytes holding the group's f16 and its neighbour's).
const SCALES: usize = 128 * 64;
const WEIGHTS: usize = SCALES + 128 * 8;
const WSCALES: usize = WEIGHTS + 64 * 32;
const STAGE: usize = WSCALES + 64 * 4;

/// `%r8` the block's first token, `%r10` the warp's first row. Copies: `%ca{i}` 16 bytes of token `th/4 + 32i`,
/// `%cs` token `th`'s `[s, z]`, `%cw` 16 bytes of row `th/2`, `%cd` row `th`'s scales (threads below 64), into
/// `%da`, `%ds`, `%dw`, `%dd`. Fragments: `%fa`, `%fs`, `%fw`, `%fd`.
const HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shr.u32 %r2, %r0, 5;
    shr.u32 %r3, %r1, 2;
    and.b32 %r4, %r1, 3;
    ld.param.u32 %r5, [pcols];
    ld.param.u32 %r6, [prows];
    ld.param.u32 %r7, [ptok];
    mov.u32 %r8, %ctaid.x;
    shl.b32 %r8, %r8, 7;
    mov.u32 %r9, %ctaid.y;
    shl.b32 %r9, %r9, 6;
    mad.lo.u32 %r10, %r2, 16, %r9;
    setp.lt.u32 %lo, %r0, 64;
    ld.param.u64 %rd0, [pq];
    ld.param.u64 %rd1, [ps];
    ld.param.u64 %rd2, [pw];
    ld.param.u64 %rd3, [pws];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    shr.u32 %r11, %r0, 2;
    add.u32 %r11, %r11, %r8;
    mul.wide.u32 %rd4, %r11, %r5;
    add.u64 %ca0, %rd0, %rd4;
    and.b32 %r12, %r0, 3;
    mul.wide.u32 %rd4, %r12, 16;
    add.u64 %ca0, %ca0, %rd4;
    mul.wide.u32 %rd5, %r5, 32;
    add.u32 %r11, %r8, %r0;
    shr.u32 %r12, %r5, 3;
    mul.wide.u32 %rd4, %r11, %r12;
    add.u64 %cs, %rd1, %rd4;
    shr.u32 %r11, %r0, 1;
    add.u32 %r11, %r11, %r9;
    shr.u32 %r12, %r5, 1;
    mul.wide.u32 %rd4, %r11, %r12;
    add.u64 %cw, %rd2, %rd4;
    and.b32 %r12, %r0, 1;
    mul.wide.u32 %rd4, %r12, 16;
    add.u64 %cw, %cw, %rd4;
    and.b32 %r11, %r0, 63;
    add.u32 %r11, %r11, %r9;
    shr.u32 %r12, %r5, 5;
    mul.wide.u32 %rd4, %r11, %r12;
    add.u64 %cd, %rd3, %rd4;
    mov.u32 %sm, smem;
    mad.lo.u32 %da, %r0, 16, %sm;
    mad.lo.u32 %ds, %r0, 8, %sm;
    add.u32 %ds, %ds, SCALES;
    mad.lo.u32 %dw, %r0, 16, %sm;
    add.u32 %dw, %dw, WEIGHTS;
    and.b32 %r11, %r0, 63;
    mad.lo.u32 %dd, %r11, 4, %sm;
    add.u32 %dd, %dd, WSCALES;
    mad.lo.u32 %fa, %r3, 64, %sm;
    mad.lo.u32 %fa, %r4, 16, %fa;
    mad.lo.u32 %fs, %r3, 8, %sm;
    add.u32 %fs, %fs, SCALES;
    mad.lo.u32 %r11, %r2, 16, %r3;
    mad.lo.u32 %fw, %r11, 32, %sm;
    mad.lo.u32 %fw, %r4, 8, %fw;
    add.u32 %fw, %fw, WEIGHTS;
    mul.lo.u32 %r11, %r2, 16;
    mad.lo.u32 %r11, %r4, 2, %r11;
    mad.lo.u32 %fd, %r11, 4, %sm;
    add.u32 %fd, %fd, WSCALES;
    mov.u32 %z, 0x4B400000;
    shr.u32 %r13, %r5, 7;
    mov.u32 %r11, %ctaid.z;
    mov.u32 %r12, %nctaid.z;
    mul.lo.u32 %r14, %r13, %r11;
    div.u32 %r14, %r14, %r12;
    mad.lo.u32 %r15, %r13, %r11, %r13;
    div.u32 %r15, %r15, %r12;
    sub.u32 %r13, %r15, %r14;
    mul.wide.u32 %rd4, %r14, 128;
    add.u64 %ca0, %ca0, %rd4;
    mul.wide.u32 %rd4, %r14, 16;
    add.u64 %cs, %cs, %rd4;
    mul.wide.u32 %rd4, %r14, 64;
    add.u64 %cw, %cw, %rd4;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %cd, %cd, %rd4;
";

/// Queues the copy of the group `off` groups past the turn's first into stage `buf`, under `guard`; the weight
/// scales under `wguard`.
fn copy(s: &mut String, buf: usize, off: usize, guard: &str, wguard: &str) {
    let base = buf * STAGE;
    for i in 0..4 {
        let _ = writeln!(s, "    {guard}cp.async.cg.shared.global [%da+{}], [%ca{i}+{}], 16;", base + 2048 * i, 64 * off);
    }
    let _ = writeln!(s, "    {guard}cp.async.ca.shared.global [%ds+{base}], [%cs+{}], 8;", 8 * off);
    let _ = writeln!(s, "    {guard}cp.async.cg.shared.global [%dw+{base}], [%cw+{}], 16;", 32 * off);
    let _ = writeln!(s, "    {wguard}cp.async.ca.shared.global [%dd+{base}], [%cd+{}], 4;\n    cp.async.commit_group;", 4 * (off / 2));
}

/// The group in stage `buf`: weight fragments and scales, then per row tile its activations and per column tile
/// a chain of two `mma` and two `fma` per output.
fn group(s: &mut String, buf: usize) {
    let base = buf * STAGE;
    for n in 0..2 {
        let _ = writeln!(s, "    ld.shared.v2.b32 {{%w0, %w1}}, [%fw+{}];", base + 256 * n);
        for (k, w) in ["%w0", "%w1"].iter().enumerate() {
            let b = 4 * n + 2 * k;
            let _ = writeln!(s, "    and.b32 %b{b}, {w}, 0x0F0F0F0F;\n    shr.u32 %b{}, {w}, 4;\n    and.b32 %b{}, %b{}, 0x0F0F0F0F;", b + 1, b + 1, b + 1);
        }
        for e in 0..2 {
            let _ = writeln!(s, "    ld.shared.b16 %hs, [%fd+{}];\n    cvt.f32.f16 %d{}, %hs;", base + 32 * n + 4 * e + 2 * buf, 2 * n + e);
        }
    }
    for m in 0..8 {
        let at = base + 1024 * m;
        let _ = writeln!(s, "    ld.shared.v4.b32 {{%a0, %a1, %a2, %a3}}, [%fa+{at}];\n    ld.shared.v4.b32 {{%a4, %a5, %a6, %a7}}, [%fa+{}];", at + 512);
        let _ =
            writeln!(s, "    ld.shared.v2.f32 {{%sx0, %sz0}}, [%fs+{}];\n    ld.shared.v2.f32 {{%sx1, %sz1}}, [%fs+{}];", base + 128 * m, base + 128 * m + 64);
        for n in 0..2 {
            let mma = "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32";
            let b = 4 * n;
            let _ = writeln!(s, "    {mma} {{%i0, %i1, %i2, %i3}}, {{%a0, %a4, %a1, %a5}}, {{%b{b}, %b{}}}, {{%z, %z, %z, %z}};", b + 1);
            let _ = writeln!(s, "    {mma} {{%i0, %i1, %i2, %i3}}, {{%a2, %a6, %a3, %a7}}, {{%b{}, %b{}}}, {{%i0, %i1, %i2, %i3}};", b + 2, b + 3);
            for e in 0..4 {
                let (j, c) = (e / 2, (2 * m + n) * 4 + e);
                let _ = writeln!(s, "    fma.rn.f32 %f, %i{e}, %sx{j}, %sz{j};\n    fma.rn.f32 %c{c}, %f, %d{}, %c{c};", 2 * n + e % 2);
            }
        }
    }
}

/// Stores the warp's tiles into the block's slice of `y`: rows `R + 8n + 2t, +1` of tokens `T + 16m + g (+8)`,
/// those below `tokens`.
fn store(s: &mut String) {
    s.push_str("    ld.param.u64 %rd0, [py];\n    cvta.to.global.u64 %rd0, %rd0;\n    mad.lo.u32 %r11, %r4, 2, %r10;\n    add.u32 %r9, %r8, %r3;\n");
    s.push_str("    mov.u32 %r14, %ctaid.z;\n    mul.lo.u32 %r15, %r7, %r6;\n    mul.wide.u32 %rd4, %r15, %r14;\n    shl.b64 %rd4, %rd4, 2;\n    add.u64 %rd0, %rd0, %rd4;\n");
    for m in 0..8 {
        for j in 0..2 {
            let _ = writeln!(s, "    add.u32 %r12, %r9, {};\n    setp.lt.u32 %p, %r12, %r7;", 16 * m + 8 * j);
            let _ = writeln!(s, "    mad.lo.u32 %r12, %r12, %r6, %r11;\n    mul.wide.u32 %rd4, %r12, 4;\n    add.u64 %rd4, %rd0, %rd4;");
            for n in 0..2 {
                let c = (2 * m + n) * 4 + 2 * j;
                let _ = writeln!(s, "    @%p st.global.v2.f32 [%rd4+{}], {{%c{c}, %c{}}};", 32 * n, c + 1);
            }
        }
    }
}

/// [`GEMM_Q4`], and [`SUM_SLICES`] after it.
pub(super) fn gemm_q4() -> String {
    let mut s = String::from(
        ".visible .entry gemm_q4(.param .u64 pw, .param .u64 pws, .param .u64 pq, .param .u64 ps, .param .u64 py, .param .u32 prows, .param .u32 pcols, .param .u32 ptok)
    .maxntid 128, 1, 1
    .minnctapersm 3
{
    .reg .pred %p, %q, %lo, %qlo;
    .reg .b16 %hs;
    .reg .b32 %r<16>, %a<8>, %b<8>, %w<2>, %i<4>, %z, %sm, %da, %ds, %dw, %dd, %fa, %fs, %fw, %fd;
    .reg .f32 %c<64>, %sx<2>, %sz<2>, %d<4>, %f;
    .reg .b64 %rd<6>, %ca<4>, %cs, %cw, %cd;
    .shared .align 16 .b8 smem[SIZE];",
    )
    .replace("SIZE", &(2 * STAGE).to_string());
    s += &HEAD.replace("WSCALES", &WSCALES.to_string()).replace("WEIGHTS", &WEIGHTS.to_string()).replace("SCALES", &SCALES.to_string());
    (1..4).for_each(|i| _ = writeln!(s, "    add.u64 %ca{i}, %ca{}, %rd5;", i - 1));
    (0..64).for_each(|c| _ = writeln!(s, "    mov.f32 %c{c}, 0f00000000;"));
    copy(&mut s, 0, 0, "", "@%lo ");
    s += "TURN:\n    setp.gt.u32 %q, %r13, 1;\n    and.pred %qlo, %q, %lo;\n";
    for buf in 0..2 {
        if buf == 0 {
            copy(&mut s, 1, 1, "", "@%lo ");
        } else {
            copy(&mut s, 0, 2, "@%q ", "@%qlo ");
        }
        s += "    cp.async.wait_group 1;\n    bar.sync 0;\n";
        group(&mut s, buf);
        s += "    bar.sync 0;\n";
    }
    (0..4).for_each(|i| _ = writeln!(s, "    add.u64 %ca{i}, %ca{i}, 128;"));
    s += "    add.u64 %cs, %cs, 16;\n    add.u64 %cw, %cw, 64;\n    add.u64 %cd, %cd, 4;\n";
    s += "    sub.u32 %r13, %r13, 1;\n    setp.ne.u32 %p, %r13, 0;\n    @%p bra TURN;\n";
    store(&mut s);
    s + "    ret;\n}\n" + SUM
}

/// A peak kernel of [`PEAKS`]: 8 independent accumulator chains per warp, so the tensor cores never wait on a result.
pub(super) fn mma_peak((name, mma, acc): (&str, &str, usize)) -> String {
    let mut s = format!(
        ".visible .entry {name}(.param .u64 pout, .param .u32 piters)
{{
    .reg .pred %p;
    .reg .b32 %r<4>, %x<4>, %y<2>, %k<32>;
    .reg .b64 %rd<2>;
    mov.u32 %r0, %tid.x;
    add.u32 %x0, %r0, 1;\n    add.u32 %x1, %r0, 2;\n    add.u32 %x2, %r0, 3;\n    add.u32 %x3, %r0, 4;\n    mov.u32 %y0, %r0;\n    mov.u32 %y1, %r0;
    ld.param.u32 %r1, [piters];\n"
    );
    (0..32).for_each(|k| _ = writeln!(s, "    mov.u32 %k{k}, 0;"));
    s += "LOOP:\n";
    for c in 0..8 {
        let k: Vec<String> = (acc * c..acc * (c + 1)).map(|k| format!("%k{k}")).collect();
        let k = k.join(", ");
        let _ = writeln!(s, "    mma.sync.aligned.{mma} {{{k}}}, {{%x0, %x1, %x2, %x3}}, {{%y0, %y1}}, {{{k}}};");
    }
    s += "    sub.u32 %r1, %r1, 1;\n    setp.ne.u32 %p, %r1, 0;\n    @%p bra LOOP;\n";
    (1..32).for_each(|k| _ = writeln!(s, "    xor.b32 %k0, %k0, %k{k};"));
    s + "    ld.param.u64 %rd0, [pout];\n    cvta.to.global.u64 %rd0, %rd0;\n    mov.u32 %r2, %ctaid.x;\n    mad.lo.u32 %r2, %r2, 128, %r0;
    mul.wide.u32 %rd1, %r2, 4;\n    add.u64 %rd0, %rd0, %rd1;\n    st.global.u32 [%rd0], %k0;\n    ret;\n}\n"
}

/// [`SUM_SLICES`].
const SUM: &str = "
.visible .entry sum_slices(.param .u64 py, .param .u32 plen, .param .u32 pslices)
{
    .reg .pred %p;
    .reg .b32 %r<4>;
    .reg .f32 %a<4>, %b<4>;
    .reg .b64 %rd<4>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    mad.lo.u32 %r0, %r0, 256, %r1;
    shl.b32 %r0, %r0, 2;
    ld.param.u32 %r2, [plen];
    setp.ge.u32 %p, %r0, %r2;
    @%p bra END;
    ld.param.u32 %r3, [pslices];
    ld.param.u64 %rd0, [py];
    cvta.to.global.u64 %rd0, %rd0;
    mul.wide.u32 %rd1, %r0, 4;
    add.u64 %rd1, %rd0, %rd1;
    ld.global.v4.f32 {%a0, %a1, %a2, %a3}, [%rd1];
    mul.wide.u32 %rd2, %r2, 4;
    mov.u64 %rd3, %rd1;
NEXT:
    sub.u32 %r3, %r3, 1;
    setp.eq.u32 %p, %r3, 0;
    @%p bra STORE;
    add.u64 %rd3, %rd3, %rd2;
    ld.global.v4.f32 {%b0, %b1, %b2, %b3}, [%rd3];
    add.f32 %a0, %a0, %b0;
    add.f32 %a1, %a1, %b1;
    add.f32 %a2, %a2, %b2;
    add.f32 %a3, %a3, %b3;
    bra NEXT;
STORE:
    st.global.v4.f32 [%rd1], {%a0, %a1, %a2, %a3};
END:
    ret;
}
";
