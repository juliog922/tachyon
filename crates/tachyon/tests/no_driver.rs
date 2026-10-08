//! Without a driver every call fails cleanly with `Error::NoDriver`.
//!
//! Its own test binary, because the driver is loaded once per process.

#[test]
fn a_missing_driver_is_reported() {
    // SAFETY: the only test of this binary; nothing else reads the environment concurrently.
    unsafe { std::env::set_var("TACHYON_LIBCUDA", "/nonexistent/libcuda.so.1") };
    assert_eq!(tachyon::cuda::devices(), Err(tachyon::Error::NoDriver));
    assert_eq!(tachyon::cuda::Context::new(0).err(), Some(tachyon::Error::NoDriver));
    assert!(tachyon::Error::NoDriver.to_string().contains("TACHYON_LIBCUDA"));
}
