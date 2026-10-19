//! Which product kernel each shape takes on this GPU: blocks of 64 tokens × 64 rows, of 128 × 64, or of 128 × 128,
//! raced the first time a shape comes. Larger blocks read fewer bytes per operation but leave multiprocessors idle
//! on short batches; where they cross depends on the GPU (its multiprocessors, its clocks, its caches), so it is
//! measured, not modelled. Used by the encoder and by Gemma 4's vision and audio towers.

use crate::Result;
use crate::cuda::{Event, Module, Stream, arg};
use crate::ptx::{GEMM_CLIP, GEMM_CLIP_128, GEMM_CLIP_TALL, GEMM_F16, GEMM_F16_128, GEMM_F16_TALL, GEMM_Q8, GEMM_Q8_128, GEMM_Q8_TALL, SPIN};

/// The kernels of one product that race: tokens and rows per block, kernel, threads.
pub(crate) type Kernels = [(u32, u32, &'static str, u32); 3];
/// The `f16` products.
pub(crate) const F16: Kernels = [(64, 64, GEMM_F16, 128), (128, 64, GEMM_F16_128, 256), (128, 128, GEMM_F16_TALL, 512)];
/// The 8-bit products.
pub(crate) const Q8: Kernels = [(64, 64, GEMM_Q8, 128), (128, 64, GEMM_Q8_128, 256), (128, 128, GEMM_Q8_TALL, 512)];
/// The clipped `f16` products of Gemma 4's towers.
pub(crate) const CLIP: Kernels = [(64, 64, GEMM_CLIP, 128), (128, 64, GEMM_CLIP_128, 256), (128, 128, GEMM_CLIP_TALL, 512)];

/// Kernel `k` of `kernels` for `tokens` × `rows`, its grid and threads: blocks of 128 rows where `rows` is a multiple
/// of 128, else the kernel of 128 tokens × 64 rows in their place.
pub(crate) fn launch(kernels: &Kernels, k: usize, tokens: u32, rows: u32) -> (&'static str, u32, u32) {
    let (t, r, kernel, threads) = kernels[fit(kernels, k, rows)];
    (kernel, tokens.div_ceil(t) * (rows / r), threads)
}

/// Kernel `k`, or 1 where its blocks' rows do not divide `rows`.
fn fit(kernels: &Kernels, k: usize, rows: u32) -> usize {
    if rows % kernels[k].1 == 0 { k } else { 1 }
}

/// How a timeline names kernel `k` of a [`Kernels`] for `rows`: its block's tokens, and rows if not 64.
pub(crate) fn label(kernels: &Kernels, k: usize, rows: u32) -> String {
    let (t, r, ..) = kernels[fit(kernels, k, rows)];
    if r == 64 { format!("/{t}") } else { format!("/{t}×{r}") }
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
    /// The kernel of a [`Kernels`] for `shape`, racing them the first time on `stream` unless a graph records. `run(k)` launches kernel `k`; a race launches each several times, so a product that
    /// adds into its output leaves it wrong, and the pass that raced runs again (see [`Products::raced`]).
    pub fn pick(&self, stream: &Stream, module: &Module, shape: [u32; 3], run: &dyn Fn(usize) -> Result<()>) -> Result<usize> {
        if let Some(&(_, k)) = self.picked.borrow().iter().find(|p| p.0 == shape) {
            return Ok(k);
        }
        if self.recording.get() {
            return Ok(0);
        }
        let k = race(stream, module, run)?;
        self.picked.borrow_mut().push((shape, k));
        self.ran.set(true);
        Ok(k)
    }

    /// Whether a race ran since the last call: the work since then is to be done again.
    pub fn raced(&self) -> bool {
        self.ran.replace(false)
    }
}

/// The fastest of `run(0)` to `run(2)`: [`LAPS`] launches each, back to back behind a [`SPIN`], timed on the GPU
/// after one launch each to warm up.
fn race(stream: &Stream, module: &Module, run: &dyn Fn(usize) -> Result<()>) -> Result<usize> {
    let events = (0..4).map(|_| stream.context().event(true)).collect::<Result<Vec<_>>>()?;
    // SAFETY: SPIN takes one u64, and runs as one block of 32 threads.
    unsafe { stream.launch(&module.function(SPIN)?, [1, 1, 1], [32, 1, 1], 0, &[arg(&HOLD)]) }?;
    (0..3).try_for_each(run)?;
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
