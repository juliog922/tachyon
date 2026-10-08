//! The GPU kernels, generated as PTX text and compiled by the driver for the GPU at hand.
//!
//! [`module`] targets `sm_80` with PTX ISA 7.1, so one text runs on every supported GPU and driver; the driver
//! compiles it to machine code once and caches it. Generating the text lets one template serve every unroll
//! depth, and later every data type, without the CUDA Toolkit.
//!
//! - [`QUANT_Q8`] quantizes an `f32` vector to Q8, the layout in [`crate::quant`]. One warp per block of 32.
//!   Arguments: `x: *const f32, q: *mut i8, s: *mut [f32; 2], len: u32`. Launch `len.div_ceil(256)` blocks of 256.
//! - [`GEMV_Q4`] computes `y = W·x` for decode: Q4 weights times Q8 activations, `f32` out. One warp per row,
//!   [`GEMV_ROWS`] rows per block. Each lane takes chunks of 32 weights (16 bytes) `lane, lane + 32, …`, with
//!   [`UNROLL`] chunks in flight per loop turn and the rest in one final round whose loads are predicated, so even a
//!   short row issues all its loads at once. Each chunk is 8 `dp4a` integer dot products, then one scale step in
//!   `f32`. Arguments: `w: *const u8, scales: *const f16, q: *const i8, s: *const [f32; 2], y: *mut f32, rows: u32,
//!   cols: u32`, with `cols` a multiple of 64. Launch `rows.div_ceil(GEMV_ROWS)` blocks of `32 · GEMV_ROWS`.

use std::fmt::Write;

/// Name of the activation quantizer.
pub const QUANT_Q8: &str = "quant_q8";
/// Name of the Q4 × Q8 matrix-vector product.
pub const GEMV_Q4: &str = "gemv_q4";
/// Rows of [`GEMV_Q4`] per block, one warp each.
pub const GEMV_ROWS: u32 = 4;
/// Chunks of 32 weights each [`GEMV_Q4`] lane has in flight per loop turn.
pub const UNROLL: usize = 4;

/// Every kernel, as one PTX module.
pub fn module() -> String {
    format!(".version 7.1\n.target sm_80\n.address_size 64\n\n{QUANT}\n{}", gemv_q4(UNROLL))
}

const QUANT: &str = "
.visible .entry quant_q8(.param .u64 px, .param .u64 pq, .param .u64 ps, .param .u32 plen)
{
    .reg .pred %p, %z;
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %d<6>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [plen];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra EXIT;
    ld.param.u64 %d0, [px];
    ld.param.u64 %d1, [pq];
    ld.param.u64 %d2, [ps];
    cvta.to.global.u64 %d0, %d0;
    cvta.to.global.u64 %d1, %d1;
    cvta.to.global.u64 %d2, %d2;
    mul.wide.u32 %d3, %r3, 4;
    add.u64 %d3, %d0, %d3;
    ld.global.f32 %f0, [%d3];
    abs.f32 %f1, %f0;
    shfl.sync.bfly.b32 %f2, %f1, 16, 31, -1;
    max.f32 %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 8, 31, -1;
    max.f32 %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 4, 31, -1;
    max.f32 %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 2, 31, -1;
    max.f32 %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 1, 31, -1;
    max.f32 %f1, %f1, %f2;
    div.rn.f32 %f3, %f1, 0f42FE0000;
    setp.gt.f32 %z, %f1, 0f00000000;
    div.rn.f32 %f4, 0f42FE0000, %f1;
    selp.f32 %f4, %f4, 0f00000000, %z;
    mul.rn.f32 %f5, %f0, %f4;
    cvt.rni.s32.f32 %r5, %f5;
    cvt.u64.u32 %d4, %r3;
    add.u64 %d4, %d1, %d4;
    st.global.s8 [%d4], %r5;
    shfl.sync.bfly.b32 %r6, %r5, 16, 31, -1;
    add.s32 %r5, %r5, %r6;
    shfl.sync.bfly.b32 %r6, %r5, 8, 31, -1;
    add.s32 %r5, %r5, %r6;
    shfl.sync.bfly.b32 %r6, %r5, 4, 31, -1;
    add.s32 %r5, %r5, %r6;
    shfl.sync.bfly.b32 %r6, %r5, 2, 31, -1;
    add.s32 %r5, %r5, %r6;
    shfl.sync.bfly.b32 %r6, %r5, 1, 31, -1;
    add.s32 %r5, %r5, %r6;
    and.b32 %r7, %r3, 31;
    setp.ne.u32 %p, %r7, 0;
    @%p bra EXIT;
    cvt.rn.f32.s32 %f6, %r5;
    mul.rn.f32 %f6, %f6, %f3;
    shr.u32 %r7, %r3, 5;
    mul.wide.u32 %d5, %r7, 8;
    add.u64 %d5, %d2, %d5;
    st.global.v2.f32 [%d5], {%f3, %f6};
EXIT:
    ret;
}
";

/// Registers: `%rd0..3` walk this lane's weights, scales, Q8 values and Q8 scales; `%r10` is its chunk index,
/// `%r7` the row's chunk count, `%r4` the row, `%r1` the lane.
const GEMV_HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shr.u32 %r2, %r0, 5;
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
    shfl.sync.bfly.b32 %o, %sum, 16, 31, -1;
    add.f32 %sum, %sum, %o;
    shfl.sync.bfly.b32 %o, %sum, 8, 31, -1;
    add.f32 %sum, %sum, %o;
    shfl.sync.bfly.b32 %o, %sum, 4, 31, -1;
    add.f32 %sum, %sum, %o;
    shfl.sync.bfly.b32 %o, %sum, 2, 31, -1;
    add.f32 %sum, %sum, %o;
    shfl.sync.bfly.b32 %o, %sum, 1, 31, -1;
    add.f32 %sum, %sum, %o;
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
fn load(ptx: &mut String, i: usize, guarded: bool) {
    let guard = if guarded {
        let _ = writeln!(ptx, "    mov.f32 %sx{i}, 0f00000000;\n    mov.f32 %ss{i}, 0f00000000;\n    mov.b16 %h{i}, 0;");
        format!("@%q{i} ")
    } else {
        String::new()
    };
    let (wr, ar) = (4 * i, 8 * i);
    let _ = writeln!(ptx, "    {guard}ld.global.cs.v4.u32 {{%w{wr}, %w{}, %w{}, %w{}}}, [%rd0+{}];", wr + 1, wr + 2, wr + 3, 512 * i);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{ar}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 1, ar + 2, ar + 3, 1024 * i);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 4, ar + 5, ar + 6, ar + 7, 1024 * i + 16);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v2.f32 {{%sx{i}, %ss{i}}}, [%rd3+{}];", 256 * i);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.b16 %h{i}, [%rd1+{}];", 32 * i);
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

/// [`GEMV_Q4`] with `u` chunks in flight per loop turn.
fn gemv_q4(u: usize) -> String {
    let mut s = format!(
        ".visible .entry gemv_q4(.param .u64 pw, .param .u64 pws, .param .u64 pq, .param .u64 ps, .param .u64 py, .param .u32 prows, .param .u32 pcols)
{{
    .reg .pred %p, %q<{u}>;
    .reg .b16 %h<{u}>;
    .reg .b32 %r<16>, %w<{}>, %a<{}>, %acc<{u}>;
    .reg .f32 %sx<{u}>, %ss<{u}>, %sw<{u}>, %f<{u}>, %sum, %o;
    .reg .b64 %rd<8>;",
        4 * u,
        8 * u
    );
    s += &GEMV_HEAD.replace("ROWS", &GEMV_ROWS.to_string());
    let _ = writeln!(s, "TURN:\n    add.u32 %r11, %r10, {};\n    setp.ge.u32 %p, %r11, %r7;\n    @%p bra LAST;", 32 * (u - 1));
    (0..u).for_each(|i| load(&mut s, i, false));
    (0..u).for_each(|i| dot(&mut s, i));
    let _ = writeln!(s, "    add.u64 %rd0, %rd0, {};\n    add.u64 %rd1, %rd1, {};\n    add.u64 %rd2, %rd2, {};", 512 * u, 32 * u, 1024 * u);
    let _ = writeln!(s, "    add.u64 %rd3, %rd3, {};\n    add.u32 %r10, %r10, {};\n    bra TURN;\nLAST:", 256 * u, 32 * u);
    for i in 0..u {
        let _ = writeln!(s, "    add.u32 %r11, %r10, {};\n    setp.lt.u32 %q{i}, %r11, %r7;", 32 * i);
        load(&mut s, i, true);
    }
    (0..u).for_each(|i| dot(&mut s, i));
    s + GEMV_TAIL
}

#[cfg(test)]
mod tests {
    /// Writes the module to `target/ptx/kernels.ptx`, where `ci.sh` checks it with `ptxas`.
    #[test]
    fn module_is_written_for_ptxas() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/ptx");
        std::fs::create_dir_all(&dir).unwrap();
        let ptx = super::module();
        assert!(ptx.contains(".entry gemv_q4") && ptx.contains(".entry quant_q8"));
        std::fs::write(dir.join("kernels.ptx"), ptx).unwrap();
    }
}
