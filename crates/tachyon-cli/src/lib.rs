//! The parts of the `tachyon` command line that benchmarks and tests reach: [`net`], downloads over HTTPS with
//! nothing but the standard library, [`models`], the catalog and the commands that manage models, and [`serve`], the daemon other processes reach
//! through `interface/`.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("tachyon targets x86_64 Linux only");

pub mod encoders;
pub mod models;
pub mod net;
pub mod serve;
