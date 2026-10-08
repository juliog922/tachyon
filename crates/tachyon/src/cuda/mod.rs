//! The CUDA Driver API behind owning types.
//!
//! [`Context`] retains a GPU's primary context and makes it current on the
//! calling thread; every other type is made from it and keeps it alive. None of
//! them is `Send`: a GPU is driven by the one thread that created its context,
//! so no call needs a lock or a context switch.
//!
//! Work is queued on a [`Stream`]. Host↔device copies and kernel launches are
//! asynchronous and therefore `unsafe`: the memory they touch must stay alive
//! and unchanged until the stream is synchronized. A sequence of launches and
//! copies is recorded once with [`Stream::capture`] and replayed as a [`Graph`]
//! with one driver call, which is how a decode step runs.
//!
//! Driver failures surface as [`Error::Cuda`] with the driver's code; a module
//! the JIT compiler rejects surfaces as [`Error::Ptx`] with its log.

mod driver;

use crate::{Error, Result};
use driver::{DevPtr, Handle, cu};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::rc::Rc;

pub(crate) use driver::error_name;

/// The CUDA version the driver supports, as `1000 × major + 10 × minor` (`13010` for 13.1).
pub fn driver_version() -> Result<i32> {
    driver::driver().map(|d| d.version)
}

/// A kernel argument for [`Stream::launch`]: a pointer to the value, as `cuLaunchKernel` takes it.
pub fn arg<T>(value: &T) -> *mut c_void {
    std::ptr::from_ref(value).cast_mut().cast()
}

/// What a GPU is, as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// CUDA's index of the device.
    pub ordinal: u32,
    /// Marketing name, such as `NVIDIA GeForce RTX 3060 Laptop GPU`.
    pub name: String,
    /// Compute capability, (major, minor).
    pub compute: (u32, u32),
    /// Streaming multiprocessors.
    pub sms: u32,
    /// Device memory in bytes.
    pub vram: u64,
    /// L2 cache in bytes.
    pub l2: u32,
    /// Memory bus width in bits; 0 when the driver does not report it.
    pub bus_width: u32,
    /// Memory clock in kHz; 0 when the driver does not report it.
    pub memory_clock: u32,
    /// PCI address, such as `0000:01:00.0`.
    pub pci: String,
}

/// A PCIe link as Linux reports it: rate per lane in GT/s and lanes, now and at most.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pcie {
    /// Current rate per lane; idle GPUs lower it to save power.
    pub speed: f32,
    /// Current lanes.
    pub width: u32,
    /// Highest rate per lane the link supports.
    pub max_speed: f32,
    /// Most lanes the device supports.
    pub max_width: u32,
}

impl Pcie {
    /// Bytes per second one direction carries at the link's top rate on its current lanes,
    /// after line encoding and before packet overhead.
    pub fn bandwidth(&self) -> f64 {
        let encoding = if self.max_speed <= 5.0 { 0.8 } else { 128.0 / 130.0 };
        f64::from(self.max_speed) * 1e9 * encoding / 8.0 * f64::from(self.width)
    }
}

impl DeviceInfo {
    /// Datasheet memory bandwidth in bytes per second; 0 when unknown. Targets use the measured figure.
    pub fn nominal_bandwidth(&self) -> f64 {
        2.0 * f64::from(self.memory_clock) * 1e3 * f64::from(self.bus_width) / 8.0
    }

    /// The PCIe link from sysfs; `None` where Linux does not show it, as under WSL2.
    pub fn pcie(&self) -> Option<Pcie> {
        let read = |f: &str| -> Option<f32> {
            std::fs::read_to_string(format!("/sys/bus/pci/devices/{}/{f}", self.pci)).ok()?.split_whitespace().next()?.parse().ok()
        };
        Some(Pcie {
            speed: read("current_link_speed")?,
            width: read("current_link_width")? as u32,
            max_speed: read("max_link_speed")?,
            max_width: read("max_link_width")? as u32,
        })
    }
}

/// Every GPU the driver sees.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    let mut n = 0;
    cu!(cuDeviceGetCount(&raw mut n))?;
    (0..n as u32).map(device).collect()
}

/// The GPU with CUDA index `ordinal`.
pub fn device(ordinal: u32) -> Result<DeviceInfo> {
    let (mut dev, mut name, mut vram) = (0, [0 as c_char; 256], 0);
    cu!(cuDeviceGet(&raw mut dev, ordinal as c_int))?;
    cu!(cuDeviceGetName(name.as_mut_ptr(), 256, dev))?;
    cu!(cuDeviceTotalMem(&raw mut vram, dev))?;
    let [major, minor, sms, l2, domain, bus, slot] = attributes(dev, [75, 76, 16, 38, 50, 33, 34])?;
    let [bus_width, memory_clock] = attributes(dev, [37, 36]).unwrap_or([0; 2]);
    Ok(DeviceInfo {
        ordinal,
        // SAFETY: the driver NUL-terminates the name inside the buffer.
        name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(),
        compute: (major, minor),
        sms,
        vram: vram as u64,
        l2,
        bus_width,
        memory_clock,
        pci: format!("{domain:04x}:{bus:02x}:{slot:02x}.0"),
    })
}

/// Device attributes, by `CUdevice_attribute` number.
fn attributes<const N: usize>(dev: c_int, ids: [c_int; N]) -> Result<[u32; N]> {
    let mut out = [0; N];
    for (slot, id) in out.iter_mut().zip(ids) {
        let mut value = 0;
        cu!(cuDeviceGetAttribute(&raw mut value, id, dev))?;
        *slot = value as u32;
    }
    Ok(out)
}

/// A GPU's primary context, current on the thread that created it. Clones share it.
#[derive(Clone)]
pub struct Context(Rc<Primary>);

struct Primary {
    device: c_int,
    info: DeviceInfo,
}

impl Drop for Primary {
    fn drop(&mut self) {
        let _ = cu!(cuDevicePrimaryCtxRelease(self.device));
    }
}

impl Context {
    /// Retains GPU `ordinal`'s primary context and makes it current on this thread.
    /// A thread drives one GPU: creating a second context makes it current instead.
    pub fn new(ordinal: u32) -> Result<Context> {
        let info = device(ordinal)?;
        if info.compute < (8, 0) {
            return Err(Error::UnsupportedGpu { major: info.compute.0, minor: info.compute.1 });
        }
        let (mut device, mut handle) = (0, null_mut());
        cu!(cuDeviceGet(&raw mut device, ordinal as c_int))?;
        cu!(cuDevicePrimaryCtxRetain(&raw mut handle, device))?;
        let ctx = Context(Rc::new(Primary { device, info }));
        cu!(cuCtxSetCurrent(handle))?;
        Ok(ctx)
    }

    /// The GPU this context drives.
    pub fn info(&self) -> &DeviceInfo {
        &self.0.info
    }

    /// Free and total device memory, in bytes.
    pub fn memory(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        cu!(cuMemGetInfo(&raw mut free, &raw mut total)).map(|()| (free, total))
    }

    /// `len` bytes of device memory, uninitialized.
    pub fn alloc(&self, len: usize) -> Result<DevBuf> {
        let mut ptr = 0;
        cu!(cuMemAlloc(&raw mut ptr, len)).map(|()| DevBuf { ptr, len, _ctx: self.clone() })
    }

    /// `len` bytes of zeroed page-locked host memory, which the GPU reaches by DMA and can map.
    pub fn alloc_host(&self, len: usize) -> Result<HostBuf> {
        const DEVICEMAP: u32 = 2;
        let mut ptr = null_mut();
        cu!(cuMemHostAlloc(&raw mut ptr, len, DEVICEMAP))?;
        // SAFETY: the driver allocated `len` writable bytes at `ptr`; zeroing makes them valid `u8`s.
        unsafe { ptr.cast::<u8>().write_bytes(0, len) };
        Ok(HostBuf { ptr: ptr.cast(), len, _ctx: self.clone() })
    }

    /// A stream that does not synchronize with the legacy default stream.
    pub fn stream(&self) -> Result<Stream> {
        const NON_BLOCKING: u32 = 1;
        let mut handle = null_mut();
        cu!(cuStreamCreate(&raw mut handle, NON_BLOCKING)).map(|()| Stream { handle, ctx: self.clone() })
    }

    /// An event; `timing` lets [`Event::since`] measure with it, at a small cost per record.
    pub fn event(&self, timing: bool) -> Result<Event> {
        const DISABLE_TIMING: u32 = 2;
        let mut handle = null_mut();
        cu!(cuEventCreate(&raw mut handle, if timing { 0 } else { DISABLE_TIMING })).map(|()| Event { handle, ctx: self.clone() })
    }

    /// Compiles PTX for this GPU. The driver caches the machine code, so only the first load pays the JIT.
    pub fn load(&self, ptx: &str) -> Result<Module> {
        const ERROR_LOG: c_int = 5;
        const ERROR_LOG_SIZE: c_int = 6;
        let image = CString::new(ptx).map_err(|_| Error::Ptx("the PTX contains a NUL byte".into()))?;
        let mut log = [0u8; 4096];
        let mut options = [ERROR_LOG, ERROR_LOG_SIZE];
        let mut values = [log.as_mut_ptr().cast(), log.len() as *mut c_void];
        let mut handle = null_mut();
        match cu!(cuModuleLoadDataEx(&raw mut handle, image.as_ptr().cast(), 2, options.as_mut_ptr(), values.as_mut_ptr())) {
            Ok(()) => Ok(Module { handle, ctx: self.clone() }),
            Err(e) if log[0] == 0 => Err(e),
            Err(_) => Err(Error::Ptx(CStr::from_bytes_until_nul(&log).map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned()))),
        }
    }
}

/// Device memory, freed on drop.
pub struct DevBuf {
    ptr: DevPtr,
    len: usize,
    _ctx: Context,
}

impl DevBuf {
    /// Its device address, for kernel arguments.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        let _ = cu!(cuMemFree(self.ptr));
    }
}

/// Page-locked host memory, freed on drop. Reads and writes as a byte slice.
pub struct HostBuf {
    ptr: *mut u8,
    len: usize,
    _ctx: Context,
}

impl HostBuf {
    /// The address kernels read and write this memory at, over PCIe and without a copy.
    pub fn device_ptr(&self) -> Result<u64> {
        let mut ptr = 0;
        cu!(cuMemHostGetDevicePointer(&raw mut ptr, self.ptr.cast(), 0)).map(|()| ptr)
    }
}

impl Deref for HostBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` holds `len` initialized bytes for as long as `self` lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for HostBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `deref`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        let _ = cu!(cuMemFreeHost(self.ptr.cast()));
    }
}

/// Declares a type that owns one driver handle and releases it with `$release` on drop.
macro_rules! owner {
    ($(#[$doc:meta])* $name:ident, $release:ident) => {
        $(#[$doc])*
        pub struct $name {
            handle: Handle,
            ctx: Context,
        }

        impl $name {
            /// The context it belongs to.
            pub fn context(&self) -> &Context {
                &self.ctx
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                let _ = cu!($release(self.handle));
            }
        }
    };
}

owner!(
    /// A queue of GPU work, run in order.
    Stream, cuStreamDestroy
);
owner!(
    /// A point in a stream, to wait for or to time.
    Event, cuEventDestroy
);
owner!(
    /// Compiled kernels, unloaded on drop.
    Module, cuModuleUnload
);
owner!(
    /// Recorded stream work, replayed with [`Stream::replay`]; destroyed on drop.
    Graph, cuGraphExecDestroy
);

fn within(len: usize, at: usize, n: usize) -> Result<DevPtr> {
    at.checked_add(n).filter(|&end| end <= len).map(|_| at as DevPtr).ok_or(Error::OutOfRange)
}

impl Stream {
    /// Queues a copy of `src` to `dst` at byte `at`.
    ///
    /// # Safety
    /// `src` must stay alive and unchanged until the stream is synchronized.
    pub unsafe fn upload(&self, dst: &DevBuf, at: usize, src: &[u8]) -> Result<()> {
        let at = within(dst.len, at, src.len())?;
        cu!(cuMemcpyHtoDAsync(dst.ptr + at, src.as_ptr().cast(), src.len(), self.handle))
    }

    /// Queues a copy of `dst.len()` bytes of `src`, from byte `at`, into `dst`.
    ///
    /// # Safety
    /// `dst` must stay alive and untouched until the stream is synchronized.
    pub unsafe fn download(&self, dst: &mut [u8], src: &DevBuf, at: usize) -> Result<()> {
        let at = within(src.len, at, dst.len())?;
        cu!(cuMemcpyDtoHAsync(dst.as_mut_ptr().cast(), src.ptr + at, dst.len(), self.handle))
    }

    /// Queues a copy of all of `src` to the start of `dst`. Freeing device memory waits for queued work.
    pub fn copy(&self, dst: &DevBuf, src: &DevBuf) -> Result<()> {
        within(dst.len, 0, src.len)?;
        cu!(cuMemcpyDtoDAsync(dst.ptr, src.ptr, src.len, self.handle))
    }

    /// Queues setting every byte of `dst` to `byte`.
    pub fn fill(&self, dst: &DevBuf, byte: u8) -> Result<()> {
        cu!(cuMemsetD8Async(dst.ptr, byte, dst.len, self.handle))
    }

    /// Queues `f` on a grid of `grid` blocks of `block` threads with `shared` bytes of dynamic shared memory.
    ///
    /// # Safety
    /// `args` must point at one value per kernel parameter, of the parameter's type (see [`arg`]),
    /// and every memory the kernel touches must stay alive until the stream is synchronized.
    pub unsafe fn launch(&self, f: &Function<'_>, grid: [u32; 3], block: [u32; 3], shared: u32, args: &[*mut c_void]) -> Result<()> {
        let [gx, gy, gz] = grid;
        let [bx, by, bz] = block;
        cu!(cuLaunchKernel(f.handle, gx, gy, gz, bx, by, bz, shared, self.handle, args.as_ptr().cast_mut(), null_mut()))
    }

    /// Records the work `record` queues on this stream, without running it, as a [`Graph`].
    pub fn capture(&self, record: impl FnOnce(&Stream) -> Result<()>) -> Result<Graph> {
        const THREAD_LOCAL: c_int = 1;
        cu!(cuStreamBeginCapture(self.handle, THREAD_LOCAL))?;
        let recorded = record(self);
        let (mut graph, mut exec) = (null_mut(), null_mut());
        let ended = cu!(cuStreamEndCapture(self.handle, &raw mut graph));
        let made = recorded.and(ended).and_then(|()| cu!(cuGraphInstantiateWithFlags(&raw mut exec, graph, 0)));
        if !graph.is_null() {
            let _ = cu!(cuGraphDestroy(graph));
        }
        made.map(|()| Graph { handle: exec, ctx: self.ctx.clone() })
    }

    /// Queues one run of `graph`: every recorded launch and copy, in one driver call.
    ///
    /// # Safety
    /// Every buffer and module the recorded work used must still be alive.
    pub unsafe fn replay(&self, graph: &Graph) -> Result<()> {
        cu!(cuGraphLaunch(graph.handle, self.handle))
    }

    /// Marks this point of the stream on `event`.
    pub fn record(&self, event: &Event) -> Result<()> {
        cu!(cuEventRecord(event.handle, self.handle))
    }

    /// Waits until everything queued so far has run.
    pub fn sync(&self) -> Result<()> {
        cu!(cuStreamSynchronize(self.handle))
    }
}

impl Event {
    /// Waits until the stream reaches the point last recorded.
    pub fn sync(&self) -> Result<()> {
        cu!(cuEventSynchronize(self.handle))
    }

    /// Milliseconds of GPU time from `start` to this event, both recorded with timing on and completed.
    pub fn since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0.0;
        cu!(cuEventElapsedTime(&raw mut ms, start.handle, self.handle)).map(|()| ms)
    }
}

impl Module {
    /// The kernel declared `.visible .entry name`.
    pub fn function(&self, name: &str) -> Result<Function<'_>> {
        const INVALID_VALUE: i32 = 1;
        let name = CString::new(name).map_err(|_| Error::Cuda(INVALID_VALUE))?;
        let mut handle = null_mut();
        cu!(cuModuleGetFunction(&raw mut handle, self.handle, name.as_ptr())).map(|()| Function { handle, _module: PhantomData })
    }
}

/// A kernel of a [`Module`].
pub struct Function<'m> {
    handle: Handle,
    _module: PhantomData<&'m Module>,
}

#[cfg(test)]
mod tests {
    use super::Pcie;

    #[test]
    fn pcie_bandwidth_follows_encoding() {
        let gen4 = Pcie { speed: 2.5, width: 16, max_speed: 16.0, max_width: 16 };
        assert!((gen4.bandwidth() / 1e9 - 31.5).abs() < 0.1);
        let gen2 = Pcie { max_speed: 5.0, width: 8, ..gen4 };
        assert!((gen2.bandwidth() / 1e9 - 4.0).abs() < 1e-9);
    }
}
