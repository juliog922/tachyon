//! The NVIDIA driver, bound at run time.
//!
//! `libcuda.so.1` ships with the driver, not with the CUDA Toolkit, so binding
//! it with `dlopen` leaves Tachyon without build-time dependencies. Every
//! function is fetched with `cuGetProcAddress_v2` at the ABI of CUDA 12.0: a
//! newer driver returns the 12.0 version of each signature, so the declarations
//! below stay valid under CUDA 13 and later.

use crate::{Error, Result};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::os::unix::ffi::OsStringExt;
use std::sync::OnceLock;

/// An opaque driver object: context, stream, event, module, function or graph.
pub type Handle = *mut c_void;
/// A device address.
pub type DevPtr = u64;
type Status = c_int;
type GetProc = unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> Status;

/// The CUDA version whose ABI every function is fetched at, and the oldest driver accepted.
pub const ABI: c_int = 12_000;
/// Where the driver is looked for when `TACHYON_LIBCUDA` is unset: the loader's path, then WSL2's.
const SEARCH: [&CStr; 2] = [c"libcuda.so.1", c"/usr/lib/wsl/lib/libcuda.so.1"];
const CUDA_ERROR_NOT_FOUND: Status = 500;

unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(lib: *mut c_void, name: *const c_char) -> *mut c_void;
}

macro_rules! api {
    ($($name:ident($($arg:ty),*);)*) => {
        /// The driver functions Tachyon calls, as `cuda.h` declares them for CUDA 12.0.
        #[allow(non_snake_case)]
        pub struct Api { $(pub $name: unsafe extern "C" fn($($arg),*) -> Status,)* }

        impl Api {
            fn resolve(get: GetProc) -> Result<Api> {
                Ok(Api { $($name: {
                    let f = find(get, concat!(stringify!($name), "\0"))?;
                    // SAFETY: `f` is the driver's function of that name at `ABI`, whose C signature is the one declared.
                    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> Status>(f) }
                },)* })
            }
        }
    };
}

api! {
    cuInit(c_uint);
    cuGetErrorName(Status, *mut *const c_char);
    cuDeviceGetCount(*mut c_int);
    cuDeviceGet(*mut c_int, c_int);
    cuDeviceGetName(*mut c_char, c_int, c_int);
    cuDeviceGetAttribute(*mut c_int, c_int, c_int);
    cuDeviceTotalMem(*mut usize, c_int);
    cuDevicePrimaryCtxRetain(*mut Handle, c_int);
    cuDevicePrimaryCtxRelease(c_int);
    cuCtxSetCurrent(Handle);
    cuMemGetInfo(*mut usize, *mut usize);
    cuMemAlloc(*mut DevPtr, usize);
    cuMemFree(DevPtr);
    cuMemHostAlloc(*mut *mut c_void, usize, c_uint);
    cuMemFreeHost(*mut c_void);
    cuMemHostGetDevicePointer(*mut DevPtr, *mut c_void, c_uint);
    cuMemcpyHtoDAsync(DevPtr, *const c_void, usize, Handle);
    cuMemcpyDtoHAsync(*mut c_void, DevPtr, usize, Handle);
    cuMemcpyDtoDAsync(DevPtr, DevPtr, usize, Handle);
    cuMemsetD8Async(DevPtr, u8, usize, Handle);
    cuStreamCreate(*mut Handle, c_uint);
    cuStreamDestroy(Handle);
    cuStreamSynchronize(Handle);
    cuStreamBeginCapture(Handle, c_int);
    cuStreamEndCapture(Handle, *mut Handle);
    cuGraphInstantiateWithFlags(*mut Handle, Handle, u64);
    cuGraphLaunch(Handle, Handle);
    cuGraphDestroy(Handle);
    cuGraphExecDestroy(Handle);
    cuEventCreate(*mut Handle, c_uint);
    cuEventDestroy(Handle);
    cuEventRecord(Handle, Handle);
    cuEventSynchronize(Handle);
    cuEventElapsedTime(*mut f32, Handle, Handle);
    cuModuleLoadDataEx(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void);
    cuModuleUnload(Handle);
    cuModuleGetFunction(*mut Handle, Handle, *const c_char);
    cuLaunchKernel(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void);
}

/// The loaded and initialized driver.
pub struct Driver {
    /// The CUDA version it supports, as `1000 × major + 10 × minor`.
    pub version: i32,
    /// Its functions.
    pub api: Api,
}

/// The driver, loaded and initialized on first use; the outcome is kept for the life of the process.
pub fn driver() -> Result<&'static Driver> {
    static DRIVER: OnceLock<Result<Driver>> = OnceLock::new();
    DRIVER.get_or_init(load).as_ref().map_err(Clone::clone)
}

/// `Ok` for `CUDA_SUCCESS`, [`Error::Cuda`] otherwise.
pub fn status(code: Status) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(Error::Cuda(code)) }
}

/// Calls a driver function and turns its status into a [`Result`].
macro_rules! cu {
    ($f:ident($($a:expr),* $(,)?)) => {
        $crate::cuda::driver::driver().and_then(|d| {
            // SAFETY: callers pass handles owned by live wrappers and pointers to live values of the C types declared in `Api`.
            $crate::cuda::driver::status(unsafe { (d.api.$f)($($a),*) })
        })
    };
}
pub(crate) use cu;

/// The driver's name for an error code, such as `CUDA_ERROR_OUT_OF_MEMORY`.
pub fn error_name(code: i32) -> &'static str {
    let mut name = std::ptr::null();
    if cu!(cuGetErrorName(code, &raw mut name)).is_err() || name.is_null() {
        return "unknown";
    }
    // SAFETY: the driver points `name` at a static NUL-terminated string.
    unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("unknown")
}

fn open() -> *mut c_void {
    const RTLD_NOW: c_int = 2;
    let custom = std::env::var_os("TACHYON_LIBCUDA").and_then(|p| CString::new(p.into_vec()).ok());
    let paths: Vec<&CStr> = custom.as_deref().map_or(SEARCH.to_vec(), |p| vec![p]);
    // SAFETY: every path is NUL-terminated; a library `dlopen` loads stays loaded for the life of the process.
    paths.iter().map(|p| unsafe { dlopen(p.as_ptr(), RTLD_NOW) }).find(|lib| !lib.is_null()).unwrap_or(std::ptr::null_mut())
}

fn load() -> Result<Driver> {
    let lib = open();
    if lib.is_null() {
        return Err(Error::NoDriver);
    }
    // SAFETY: `lib` is a loaded library and both names are NUL-terminated.
    let (version_of, get) = unsafe { (dlsym(lib, c"cuDriverGetVersion".as_ptr()), dlsym(lib, c"cuGetProcAddress_v2".as_ptr())) };
    if version_of.is_null() {
        return Err(Error::NoDriver);
    }
    let mut version = 0;
    // SAFETY: `cuDriverGetVersion(int*)` has kept this signature since CUDA 2.2, and `version` is a live local.
    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_int) -> Status>(version_of)(&raw mut version) };
    if version < ABI || get.is_null() {
        return Err(Error::DriverTooOld(version));
    }
    // SAFETY: `cuGetProcAddress_v2`, present from CUDA 12.0, has the `GetProc` signature.
    let api = Api::resolve(unsafe { std::mem::transmute::<*mut c_void, GetProc>(get) })?;
    // SAFETY: `cuInit` takes no pointers.
    status(unsafe { (api.cuInit)(0) })?;
    Ok(Driver { version, api })
}

/// The driver function named by NUL-terminated `name`, at [`ABI`].
fn find(get: GetProc, name: &str) -> Result<*mut c_void> {
    let (mut f, mut found) = (std::ptr::null_mut(), 0);
    // SAFETY: `name` is NUL-terminated and both out-pointers are live locals.
    match (unsafe { get(name.as_ptr().cast(), &raw mut f, ABI, 0, &raw mut found) }, f.is_null()) {
        (0, false) => Ok(f),
        (0, true) => Err(Error::Cuda(CUDA_ERROR_NOT_FOUND)),
        (code, _) => Err(Error::Cuda(code)),
    }
}
