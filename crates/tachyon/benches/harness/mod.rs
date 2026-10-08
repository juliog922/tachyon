//! Benchmark harness: warm-up, timed runs, median and p99, `perf` counters,
//! JSON results and a regression check against a stored baseline.
//!
//! A run's time is whatever the measured closure returns, so GPU benches time
//! with events and CPU benches with [`cpu`].
//!
//! Each suite writes `target/bench/<suite>.json`. With a baseline at
//! `bench/results/<machine>/<suite>.json`, a median more than 3% slower fails
//! the suite. `TACHYON_BENCH_SAVE=1` stores the run as the new baseline. The
//! machine is `$TACHYON_BENCH_MACHINE`, else the host name.

// Each bench includes this module and uses part of it.
#![allow(dead_code)]

pub mod gpu;

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read as _;
use std::os::fd::FromRawFd as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const TOLERANCE: f64 = 1.03;
/// `perf` events: (type, config). Hardware cycles, instructions, cache misses; software page faults.
const COUNTERS: [(u32, u64, &str); 4] = [(0, 0, "cycles"), (0, 1, "instructions"), (0, 3, "cache_misses"), (1, 2, "page_faults")];

/// Times `f` with a monotonic clock.
pub fn cpu(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

struct Row {
    name: String,
    median: Duration,
    p99: Duration,
    rate: Option<(f64, &'static str)>,
    counters: [Option<u64>; 4],
}

/// A named set of measurements.
pub struct Suite {
    name: &'static str,
    rows: Vec<Row>,
}

impl Suite {
    /// An empty suite.
    pub fn new(name: &'static str) -> Suite {
        Suite { name, rows: Vec::new() }
    }

    /// Runs `f` `runs / 5 + 1` times to warm up, then `runs` times; records and returns the median.
    pub fn run(&mut self, name: &str, runs: usize, mut f: impl FnMut() -> Duration) -> Duration {
        for _ in 0..=runs / 5 {
            f();
        }
        let counters = COUNTERS.map(|(kind, config, _)| Counter::open(kind, config));
        counters.iter().flatten().for_each(|c| c.ctl(0x2400));
        let mut times: Vec<Duration> = (0..runs).map(|_| f()).collect();
        counters.iter().flatten().for_each(|c| c.ctl(0x2401));
        times.sort();
        let (median, p99) = (times[runs / 2], times[(runs * 99).div_ceil(100) - 1]);
        let counters = counters.map(|c| c.and_then(Counter::read).map(|n| n / runs as u64));
        self.rows.push(Row { name: name.into(), median, p99, rate: None, counters });
        median
    }

    /// Shows `value unit` beside the last measurement, such as a bandwidth.
    pub fn rate(&mut self, value: f64, unit: &'static str) {
        self.rows.last_mut().expect("a measurement").rate = Some((value, unit));
    }

    /// Prints the table, writes the results, and checks or saves the baseline.
    pub fn finish(self) -> ExitCode {
        let machine =
            std::env::var("TACHYON_BENCH_MACHINE").unwrap_or_else(|_| std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().into());
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("the workspace root").to_path_buf();
        let baseline = root.join(format!("bench/results/{machine}/{}.json", self.name));
        let json = self.json();
        let save = std::env::var("TACHYON_BENCH_SAVE").is_ok_and(|v| v == "1");
        let out = if save { baseline.clone() } else { root.join(format!("target/bench/{}.json", self.name)) };
        std::fs::create_dir_all(out.parent().unwrap()).and_then(|()| std::fs::write(&out, &json)).expect("writable results");
        println!("{}", self.table(&machine));
        println!("results: {}", out.display());
        match std::fs::read_to_string(&baseline) {
            Ok(old) if !save => self.compare(&old),
            _ => ExitCode::SUCCESS,
        }
    }

    fn table(&self, machine: &str) -> String {
        let mut t = format!("{} on {machine}\n{:<40} {:>12} {:>12} {:>16}\n", self.name, "", "median", "p99", "rate");
        for r in &self.rows {
            let rate = r.rate.map_or(String::new(), |(v, u)| format!("{v:.2} {u}"));
            let _ = writeln!(t, "{:<40} {:>12.3?} {:>12.3?} {rate:>16}", r.name, r.median, r.p99);
        }
        t
    }

    fn json(&self) -> String {
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let mut line = format!("{{\"name\":\"{}\",\"median_ns\":{},\"p99_ns\":{}", r.name, r.median.as_nanos(), r.p99.as_nanos());
                if let Some((v, u)) = r.rate {
                    let _ = write!(line, ",\"rate\":{v:.3},\"unit\":\"{u}\"");
                }
                for ((_, _, name), n) in COUNTERS.iter().zip(r.counters) {
                    let _ = write!(line, ",\"{name}\":{}", n.map_or("null".into(), |n| n.to_string()));
                }
                line + "}"
            })
            .collect();
        format!("[\n{}\n]\n", rows.join(",\n"))
    }

    /// Fails when a median is more than [`TOLERANCE`] times its baseline.
    fn compare(&self, baseline: &str) -> ExitCode {
        let mut ok = true;
        for line in baseline.lines() {
            let field = |key: &str| line.split(&format!("\"{key}\":")).nth(1).map(|v| v.split([',', '}']).next().unwrap_or("").trim_matches('"'));
            let (Some(name), Some(Ok(old))) = (field("name"), field("median_ns").map(str::parse::<f64>)) else {
                continue;
            };
            if let Some(r) = self.rows.iter().find(|r| r.name == name) {
                let ratio = r.median.as_nanos() as f64 / old;
                if ratio > TOLERANCE {
                    println!("REGRESSION {name}: {:.1}% slower than the baseline", (ratio - 1.0) * 100.0);
                    ok = false;
                }
            }
        }
        if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
    }
}

/// One `perf` counter of this thread, user space only.
struct Counter(File);

impl Counter {
    fn open(kind: u32, config: u64) -> Option<Counter> {
        let mut attr = [0u64; 16];
        attr[0] = u64::from(kind) | 128 << 32;
        attr[1] = config;
        attr[5] = 1 | 1 << 5 | 1 << 6; // disabled, exclude_kernel, exclude_hv
        // SAFETY: `perf_event_open(attr, this thread, any CPU, no group, FD_CLOEXEC)` with a live 128-byte attribute block.
        let fd = unsafe { syscall(298, attr.as_ptr() as usize, 0, usize::MAX, usize::MAX, 8) };
        // SAFETY: a successful call returns a new descriptor that nothing else owns.
        (fd >= 0).then(|| Counter(unsafe { File::from_raw_fd(fd as i32) }))
    }

    /// `PERF_EVENT_IOC_ENABLE` (0x2400) or `PERF_EVENT_IOC_DISABLE` (0x2401).
    fn ctl(&self, request: usize) {
        use std::os::fd::AsRawFd as _;
        // SAFETY: an argument-less ioctl on a perf descriptor this counter owns.
        unsafe { syscall(16, self.0.as_raw_fd() as usize, request, 0, 0, 0) };
    }

    fn read(mut self) -> Option<u64> {
        let mut v = [0u8; 8];
        self.0.read_exact(&mut v).ok().map(|()| u64::from_ne_bytes(v))
    }
}

/// # Safety
/// The arguments must be valid for system call `nr`.
unsafe fn syscall(nr: usize, a0: usize, a1: usize, a2: usize, a3: usize, a4: usize) -> isize {
    let r: isize;
    // SAFETY: the caller upholds the call's contract; `syscall` clobbers only rcx and r11.
    unsafe {
        std::arch::asm!("syscall", inlateout("rax") nr as isize => r, in("rdi") a0, in("rsi") a1, in("rdx") a2, in("r10") a3, in("r8") a4,
            lateout("rcx") _, lateout("r11") _, options(nostack));
    }
    r
}