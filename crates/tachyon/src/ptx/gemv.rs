//! [`GEMV_Q4`], the decode matrix-vector product: Q4 weights times Q8 activations, `f32` out.
//!
//! One warp per row, [`GEMV_ROWS`] rows per block. Each lane takes chunks of 32 weights (16 bytes)
//! `lane, lane + 32, …`, with [`UNROLL`] chunks in flight per loop turn and the rest in one final round whose
//! loads are predicated, so even a short row issues all its loads at once. Each chunk is 8 `dp4a` integer dot
//! products, then one scale step in `f32`.
//!
//! [`GEMV_Q4_NARROW`] is the same product for rows of at most 256 weights (8 chunks): 8 lanes per row, so a warp
//! takes 4 rows and a block 16, and a matrix of such rows runs in a quarter of the blocks. Its rows are a multiple
//! of 4. Launch `rows.div_ceil(16)` blocks of 128 threads.

use std::fmt::Write;

/// Name of the Q4 × Q8 matrix-vector product.
pub const GEMV_Q4: &str = "gemv_q4";
/// Rows of [`GEMV_Q4`] per block, one warp each.
pub const GEMV_ROWS: u32 = 4;
/// Name of [`GEMV_Q4`] for rows of at most 256 weights.
pub const GEMV_Q4_NARROW: &str = "gemv_q4_narrow";
/// Chunks of 32 weights each [`GEMV_Q4`] lane has in flight per loop turn.
pub const UNROLL: usize = 4;

/// Registers: `%rd0..3` walk this lane's weights, scales, Q8 values and Q8 scales; `%r10` is its chunk index,
/// `%r7` the row's chunk count, `%r4` the row, `%r1` the lane.
const GEMV_HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, LMASK;
    shr.u32 %r2, %r0, LSHIFT;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r4, %r3, ROWS, %r2;
    ld.param.u32 %r5, [prows];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra EXIT;
    ld.param.u32 %r6, [pcols];
    shr.u32 %r7, %r6, 5;
    ld.param.u64 %rd0, [pw];
    ld.param.u64 %rd1, [pws];
    ld.param.u64 %rd2, [pq];
    ld.param.u64 %rd3, [ps];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    shr.u32 %r8, %r6, 1;
    mul.wide.u32 %rd4, %r4, %r8;
    add.u64 %rd0, %rd0, %rd4;
    mul.wide.u32 %rd4, %r4, %r7;
    add.u64 %rd1, %rd1, %rd4;
    mul.wide.u32 %rd4, %r1, 16;
    add.u64 %rd0, %rd0, %rd4;
    shr.u32 %r9, %r1, 1;
    mul.wide.u32 %rd4, %r9, 2;
    add.u64 %rd1, %rd1, %rd4;
    mul.wide.u32 %rd4, %r1, 32;
    add.u64 %rd2, %rd2, %rd4;
    mul.wide.u32 %rd4, %r1, 8;
    add.u64 %rd3, %rd3, %rd4;
    mov.u32 %r10, %r1;
    mov.f32 %sum, 0f00000000;
";

const GEMV_TAIL: &str = "
    setp.ne.u32 %p, %r1, 0;
    @%p bra EXIT;
    ld.param.u64 %rd5, [py];
    cvta.to.global.u64 %rd5, %rd5;
    mul.wide.u32 %rd6, %r4, 4;
    add.u64 %rd5, %rd5, %rd6;
    st.global.f32 [%rd5], %sum;
EXIT:
    ret;
}
";

/// Loads chunk `i` of the current turn: 16 bytes of weights, its scale, 32 Q8 values and their scales. A guarded
/// load runs only under predicate `%q{i}`, after zeroing what makes a missing chunk add nothing.
fn load(ptx: &mut String, i: usize, guarded: bool, lanes: usize) {
    let guard = if guarded {
        let _ = writeln!(ptx, "    mov.f32 %sx{i}, 0f00000000;\n    mov.f32 %ss{i}, 0f00000000;\n    mov.b16 %h{i}, 0;");
        format!("@%q{i} ")
    } else {
        String::new()
    };
    let (wr, ar, l) = (4 * i, 8 * i, lanes * i);
    let _ = writeln!(ptx, "    {guard}ld.global.cs.v4.u32 {{%w{wr}, %w{}, %w{}, %w{}}}, [%rd0+{}];", wr + 1, wr + 2, wr + 3, 16 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{ar}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 1, ar + 2, ar + 3, 32 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 4, ar + 5, ar + 6, ar + 7, 32 * l + 16);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v2.f32 {{%sx{i}, %ss{i}}}, [%rd3+{}];", 8 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.b16 %h{i}, [%rd1+{l}];");
}

/// Adds chunk `i` to `%sum`: `d · (s · Σ n·q − 8 · s·Σq)`, the integer sum from 8 `dp4a`.
fn dot(ptx: &mut String, i: usize) {
    for word in 0..4 {
        let (wr, ar) = (4 * i + word, 8 * i + 2 * word);
        let acc = if word == 0 { "0".to_string() } else { format!("%acc{i}") };
        let _ = writeln!(ptx, "    and.b32 %r12, %w{wr}, 0x0F0F0F0F;\n    dp4a.u32.s32 %acc{i}, %r12, %a{ar}, {acc};");
        let _ = writeln!(ptx, "    shr.u32 %r13, %w{wr}, 4;\n    and.b32 %r13, %r13, 0x0F0F0F0F;\n    dp4a.u32.s32 %acc{i}, %r13, %a{}, %acc{i};", ar + 1);
    }
    let _ = writeln!(ptx, "    cvt.f32.f16 %sw{i}, %h{i};\n    cvt.rn.f32.s32 %f{i}, %acc{i};\n    mul.f32 %f{i}, %f{i}, %sx{i};");
    let _ = writeln!(ptx, "    fma.rn.f32 %f{i}, %ss{i}, 0fC1000000, %f{i};\n    fma.rn.f32 %sum, %f{i}, %sw{i}, %sum;");
}

/// [`GEMV_Q4`] with `u` chunks in flight per loop turn, or with `lanes` 8, [`GEMV_Q4_NARROW`].
pub(super) fn gemv_q4(u: usize, lanes: usize) -> String {
    let name = if lanes == 32 { GEMV_Q4 } else { GEMV_Q4_NARROW };
    let mut s = format!(
        ".visible .entry {name}(.param .u64 pw, .param .u64 pws, .param .u64 pq, .param .u64 ps, .param .u64 py, .param .u32 prows, .param .u32 pcols)
{{
    .reg .pred %p, %q<{u}>;
    .reg .b16 %h<{u}>;
    .reg .b32 %r<16>, %w<{}>, %a<{}>, %acc<{u}>;
    .reg .f32 %sx<{u}>, %ss<{u}>, %sw<{u}>, %f<{u}>, %sum, %o;
    .reg .b64 %rd<8>;",
        4 * u,
        8 * u
    );
    let head = GEMV_HEAD.replace("LMASK", &(lanes - 1).to_string()).replace("LSHIFT", &lanes.trailing_zeros().to_string());
    s += &head.replace("ROWS", &(128 / lanes).to_string());
    let _ = writeln!(s, "TURN:\n    add.u32 %r11, %r10, {};\n    setp.ge.u32 %p, %r11, %r7;\n    @%p bra LAST;", lanes * (u - 1));
    (0..u).for_each(|i| load(&mut s, i, false, lanes));
    (0..u).for_each(|i| dot(&mut s, i));
    let l = lanes * u;
    let _ = writeln!(s, "    add.u64 %rd0, %rd0, {};\n    add.u64 %rd1, %rd1, {l};\n    add.u64 %rd2, %rd2, {};", 16 * l, 32 * l);
    let _ = writeln!(s, "    add.u64 %rd3, %rd3, {};\n    add.u32 %r10, %r10, {l};\n    bra TURN;\nLAST:", 8 * l);
    for i in 0..u {
        let _ = writeln!(s, "    add.u32 %r11, %r10, {};\n    setp.lt.u32 %q{i}, %r11, %r7;", lanes * i);
        load(&mut s, i, true, lanes);
    }
    (0..u).for_each(|i| dot(&mut s, i));
    let mut k = lanes / 2;
    while k > 0 {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %o, %sum, {k}, 31, -1;\n    add.f32 %sum, %sum, %o;");
        k /= 2;
    }
    s + GEMV_TAIL
}