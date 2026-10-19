//! Which product kernel each shape takes on this GPU: blocks of 64 tokens × 64 rows, of 128 × 64, or of 128 × 128,
//! and for Q8 also 64 × 64 and 128 × 64 taking 128 of `k` a turn, raced the first time a shape comes. Larger blocks
//! read fewer bytes per operation but leave multiprocessors idle on short batches; where they cross depends on the
//! GPU (its multiprocessors, its clocks, its caches), so it is measured, not modelled. Used by the encoder and by
//! Gemma 4's vision and audio towers.

use crate::Result;
use crate::cuda::{Event, Module, Stream, arg};
use crate::ptx::{
    GEMM_CLIP, GEMM_CLIP_128, GEMM_CLIP_TALL, GEMM_F16, GEMM_F16_128, GEMM_F16_TALL, GEMM_Q8, GEMM_Q8_128, GEMM_Q8_128K, GEMM_Q8_K, GEMM_Q8_TALL, SPIN,
};

/// The kernels of one product that race: tokens and rows per block, the multiple of columns it takes, kernel,
/// threads.
pub(crate) type Kernels = &'static [(u32, u32, u32, &'static str, u32)];
/// The `f16` products.
pub(crate) const F16: Kernels = &[(64, 64, 64, GEMM_F16, 128), (128, 64, 64, GEMM_F16_128, 256), (128, 128, 64, GEMM_F16_TALL, 512)];
/// The 8-bit products.
pub(crate) const Q8: Kernels = &[
    (64, 64, 64, GEMM_Q8, 128),
    (128, 64, 64, GEMM_Q8_128, 256),
    (128, 128, 64, GEMM_Q8_TALL, 512),
    (64, 64, 128, GEMM_Q8_K, 128),
    (128, 64, 128, GEMM_Q8_128K, 256),
];
/// The clipped `f16` products of Gemma 4's towers.
pub(crate) const CLIP: Kernels = &[(64, 64, 64, GEMM_CLIP, 128), (128, 64, 64, GEMM_CLIP_128, 256), (128, 128, 64, GEMM_CLIP_TALL, 512)];

/// Kernel `k` of `kernels` for `tokens` × `[rows, cols]`, its grid and threads; where the kernel's blocks do not
/// divide the matrix, the kernel of 128 tokens × 64 rows in its place.
pub(crate) fn launch(kernels: Kernels, k: usize, tokens: u32, shape: [u32; 2]) -> (&'static str, u32, u32) {
    let (t, r, _, kernel, threads) = kernels[fit(kernels, k, shape)];
    (kernel, tokens.div_ceil(t) * (shape[0] / r), threads)
}

/// Kernel `k`, or 1 where its blocks' rows or columns do not divide `[rows, cols]`.
fn fit(kernels: Kernels, k: usize, [rows, cols]: [u32; 2]) -> usize {
    if rows % kernels[k].1 == 0 && cols % kernels[k].2 == 0 { k } else { 1 }
}

/// How a timeline names kernel `k` of a [`Kernels`] for `shape`: its block's tokens, its rows if not 64, and the `k`
/// of a turn if 128.
pub(crate) fn label(kernels: Kernels, k: usize, shape: [u32; 2]) -> String {
    let (t, r, c, ..) = kernels[fit(kernels, k, shape)];
    format!("/{t}{}{}", if r == 64 { String::new() } else { format!("×{r}") }, if c == 128 { " k128" } else { "" })
}
use std::cell::{Cell, RefCell};

/// Launches of each kernel timed in a race.
const LAPS: usize = 3;
/// What the [`SPIN`] before a race holds the GPU for while the CPU queues its launches, in nanoseconds.
const HOLD: u64 = 200_000;

/// The kernel picked for each product shape: rows, columns, tokens.
#[derive(Default)]
pub(crate) struct Products {
    picked: RefCell<Vec<([u32; 3], usize)>>,
    /// Whether a race ran since [`Products::raced`] last answered.
    ran: Cell<bool>,
    /// While a graph records, which cannot wait for a race: shapes not yet raced take kernel 0, blocks of 64.
    pub recording: Cell<bool>,
}

impl Products {
    /// The kernel of `n` of a [`Kernels`] for `shape`, racing them the first time on `stream` unless a graph records.
    /// `run(k)` launches kernel `k`; a race launches each several times, so a product that adds into its output
    /// leaves it wrong, and the pass that raced runs again (see [`Products::raced`]).
    pub fn pick(&self, stream: &Stream, module: &Module, shape: [u32; 3], (n, run): (usize, &dyn Fn(usize) -> Result<()>)) -> Result<usize> {
        if let Some(&(_, k)) = self.picked.borrow().iter().find(|p| p.0 == shape) {
            return Ok(k);
        }
        if self.recording.get() {
            return Ok(0);
        }
        let k = race(stream, module, n, run)?;
        self.picked.borrow_mut().push((shape, k));
        self.ran.set(true);
        Ok(k)
    }

    /// Whether a race ran since the last call: the work since then is to be done again.
    pub fn raced(&self) -> bool {
        self.ran.replace(false)
    }
}

/// The fastest of `run(0)` to `run(n − 1)`: [`LAPS`] launches each, back to back behind a [`SPIN`], timed on the GPU
/// after one launch each to warm up.
fn race(stream: &Stream, module: &Module, n: usize, run: &dyn Fn(usize) -> Result<()>) -> Result<usize> {
    let events = (0..=n).map(|_| stream.context().event(true)).collect::<Result<Vec<_>>>()?;
    // SAFETY: SPIN takes one u64, and runs as one block of 32 threads.
    unsafe { stream.launch(&module.function(SPIN)?, [1, 1, 1], [32, 1, 1], 0, &[arg(&HOLD)]) }?;
    (0..n).try_for_each(run)?;
    stream.record(&events[0])?;
    for (k, event) in events[1..].iter().enumerate() {
        (0..LAPS).try_for_each(|_| run(k))?;
        stream.record(event)?;
    }
    fastest(&events)
}

/// The index of the shortest of the times between consecutive `events`, once they have passed.
fn fastest(events: &[Event]) -> Result<usize> {
    events[events.len() - 1].sync()?;
    let times = events.windows(2).map(|w| w[1].since(&w[0])).collect::<Result<Vec<f32>>>()?;
    Ok((0..times.len()).fold(0, |best, k| if times[k] < times[best] { k } else { best }))
}
