//! Which `f16` product kernel each shape takes on this GPU: blocks of 64 tokens or of 128, raced the first time a
//! shape comes. The 128-token blocks read fewer bytes per operation but leave multiprocessors idle on short batches;
//! where the two cross depends on the GPU (its multiprocessors, its clocks, its caches), so it is measured, not
//! modelled. Used by the encoder and by Gemma 4's vision and audio towers.

use crate::Result;
use crate::cuda::{Event, Module, Stream, arg};
use crate::ptx::SPIN;
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
    /// While a graph records, which cannot wait for a race: shapes not yet raced take blocks of 64.
    pub recording: Cell<bool>,
}

impl Products {
    /// Kernel 0 (blocks of 128 tokens) or 1 (blocks of 64) for `shape`, racing them the first time on `stream`
    /// unless a graph records. `run(k)` launches kernel `k`; a race launches each several times, so a product that
    /// adds into its output leaves it wrong, and the pass that raced runs again (see [`Products::raced`]).
    pub fn pick(&self, stream: &Stream, module: &Module, shape: [u32; 3], run: &dyn Fn(usize) -> Result<()>) -> Result<usize> {
        if let Some(&(_, k)) = self.picked.borrow().iter().find(|p| p.0 == shape) {
            return Ok(k);
        }
        if self.recording.get() {
            return Ok(1);
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

/// The faster of `run(0)` and `run(1)`: [`LAPS`] launches each, back to back behind a [`SPIN`], timed on the GPU
/// after one launch each to warm up.
fn race(stream: &Stream, module: &Module, run: &dyn Fn(usize) -> Result<()>) -> Result<usize> {
    let events = (0..3).map(|_| stream.context().event(true)).collect::<Result<Vec<_>>>()?;
    // SAFETY: SPIN takes one u64, and runs as one block of 32 threads.
    unsafe { stream.launch(&module.function(SPIN)?, [1, 1, 1], [32, 1, 1], 0, &[arg(&HOLD)]) }?;
    (0..2).try_for_each(run)?;
    stream.record(&events[0])?;
    for (k, event) in events[1..].iter().enumerate() {
        (0..LAPS).try_for_each(|_| run(k))?;
        stream.record(event)?;
    }
    faster(&events)
}

/// 1 when the time between the last two of `events` is shorter than between the first two, once they have passed.
fn faster(events: &[Event]) -> Result<usize> {
    events[2].sync()?;
    Ok(usize::from(events[2].since(&events[1])? < events[1].since(&events[0])?))
}
