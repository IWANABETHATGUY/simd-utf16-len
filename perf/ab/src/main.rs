//! Times `utf16_len` from two builds of this crate on the same machine.
//!
//! `scripts/perf-ab.sh` builds this harness twice from the same directory:
//! once against the base ref's crate and once against the working tree's.
//! The head binary runs the comparison and starts the base binary for the
//! other side. Every process times one input, and a run times each input's
//! base and head sides back to back in fresh processes, alternating which
//! goes first. Separate binaries give identical code identical addresses:
//! with both sides in one binary, the copy that landed in the worse spot for
//! the branch predictors measured 10 to 15% slower on some inputs with no
//! code change. Each input reports the median over the runs of the head/base
//! time ratio.
//!
//! The head processes also time the standard-library baseline from the
//! README, for a report-only comparison of head against std.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

#[path = "../../../benches/inputs.rs"]
mod inputs;

/// Target duration of one timed batch.
const BATCH: Duration = Duration::from_millis(1);
const DEFAULT_ROUNDS: usize = 31;
const DEFAULT_RUNS: usize = 7;
const WARMUP_ROUNDS: usize = 10;
/// A run only counts as slower beyond this, since no-change runs can all land
/// a fraction of a percent on the slower side.
const SLOWER_RUN_PCT: f64 = 1.0;

const USAGE: &str = "usage: simd-utf16-len-ab --base-exe <path> [--fail-above <percent>] [--json <path>] [--runs <n>] [--rounds <n>]";

struct Options {
    /// The binary built against the base ref; it answers `--child` like this one.
    base_exe: Option<PathBuf>,
    /// Fail when an input's median time change exceeds this many percent.
    fail_above: Option<f64>,
    json: Option<String>,
    runs: usize,
    rounds: usize,
    /// Time this one input and print the raw result for the parent process.
    child: Option<String>,
}

/// Median per-call times, head/base time ratio, and std/head speedup of one run.
struct Run {
    base_ns: f64,
    head_ns: f64,
    std_ns: f64,
    ratio: f64,
    speedup: f64,
}

/// One process's median per-call times of its own `utf16_len` and of std.
struct Timing {
    own_ns: f64,
    std_ns: f64,
}

struct Measurement {
    name: &'static str,
    bytes: usize,
    base_ns: f64,
    head_ns: f64,
    std_ns: f64,
    /// Head/base time ratio of each run, sorted ascending.
    ratios: Vec<f64>,
    /// Std/head time ratio of each run, sorted ascending.
    speedups: Vec<f64>,
}

impl Measurement {
    /// How many times faster head is than the standard-library baseline.
    fn speedup(&self) -> f64 {
        percentile(&self.speedups, 0.5)
    }

    /// Change in time per call at percentile `p`; negative means head is faster.
    fn change(&self, p: f64) -> f64 {
        percentile(&self.ratios, p) - 1.0
    }

    fn slower_runs(&self) -> usize {
        self.ratios
            .iter()
            .filter(|&&ratio| (ratio - 1.0) * 100.0 > SLOWER_RUN_PCT)
            .count()
    }

    /// Noisy inputs can put the median past the limit with no code change, but
    /// their runs then disagree, so a regression must also slow every run by
    /// more than `SLOWER_RUN_PCT`.
    fn regressed(&self, limit: f64) -> bool {
        self.change(0.5) * 100.0 > limit && self.slower_runs() == self.ratios.len()
    }
}

fn main() -> ExitCode {
    let options = match parse_args() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let inputs = inputs::all();

    if let Some(name) = &options.child {
        let Some((_, input)) = inputs.iter().find(|(known, _)| known == name) else {
            eprintln!("unknown input: {name}");
            return ExitCode::from(2);
        };
        let timing = measure(input, options.rounds);
        println!("{name}\t{}\t{}", timing.own_ns, timing.std_ns);
        return ExitCode::SUCCESS;
    }

    let Some(base_exe) = &options.base_exe else {
        eprintln!("--base-exe is required\n{USAGE}");
        return ExitCode::from(2);
    };

    // Check head against std rather than base, so fixing a bug in base doesn't fail.
    for (name, input) in &inputs {
        let (head, expected) = (simd_utf16_len::utf16_len(input), std_len(input));
        if head != expected {
            eprintln!("{name}: head returned {head} but std counts {expected}");
            return ExitCode::from(2);
        }
    }

    let results = match measure_in_children(&inputs, base_exe, &options) {
        Ok(results) => results,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let regressions: Vec<_> = match options.fail_above {
        Some(limit) => results
            .iter()
            .filter(|m| m.regressed(limit))
            .map(|m| m.name)
            .collect(),
        None => Vec::new(),
    };

    let env = Environment::detect();
    let report = markdown(&results, &regressions, &options, &env);
    print!("{report}");
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        append(&path, &report);
    }
    if let Some(path) = &options.json
        && let Err(error) = std::fs::write(path, json(&results, &regressions, &options, &env))
    {
        eprintln!("failed to write {path}: {error}");
        return ExitCode::from(2);
    }

    if regressions.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        base_exe: None,
        fail_above: None,
        json: None,
        runs: DEFAULT_RUNS,
        rounds: DEFAULT_ROUNDS,
        child: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--base-exe" => options.base_exe = Some(PathBuf::from(value()?)),
            "--fail-above" => {
                let limit = value()?;
                options.fail_above = Some(
                    limit
                        .parse()
                        .map_err(|_| format!("invalid percent: {limit}"))?,
                );
            }
            "--json" => options.json = Some(value()?),
            "--runs" => options.runs = count(value()?)?,
            "--rounds" => options.rounds = count(value()?)?,
            "--child" => options.child = Some(value()?),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(options)
}

fn count(value: String) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("invalid count: {value}"))
}

/// Times every input `options.runs` times, each side in its own fresh
/// process, and combines the runs.
fn measure_in_children(
    inputs: &[(&'static str, String)],
    base_exe: &Path,
    options: &Options,
) -> Result<Vec<Measurement>, String> {
    let head_exe =
        std::env::current_exe().map_err(|error| format!("cannot find myself: {error}"))?;
    let mut runs: Vec<Vec<Run>> = Vec::with_capacity(options.runs);
    for run in 0..options.runs {
        let mut timings = Vec::with_capacity(inputs.len());
        for (i, (name, _)) in inputs.iter().enumerate() {
            // Alternate which side goes first, so drift during a pair can't
            // favor one side.
            let base_first = (run + i) % 2 == 0;
            let (first, second) = if base_first {
                (base_exe, head_exe.as_path())
            } else {
                (head_exe.as_path(), base_exe)
            };
            let a = time_in_child(first, name, options.rounds)?;
            let b = time_in_child(second, name, options.rounds)?;
            let (base, head) = if base_first { (a, b) } else { (b, a) };
            timings.push(Run {
                base_ns: base.own_ns,
                head_ns: head.own_ns,
                std_ns: head.std_ns,
                ratio: head.own_ns / base.own_ns,
                speedup: head.std_ns / head.own_ns,
            });
        }
        runs.push(timings);
    }

    Ok(inputs
        .iter()
        .enumerate()
        .map(|(i, (name, input))| {
            let sorted = |field: fn(&Run) -> f64| {
                let mut values: Vec<f64> = runs.iter().map(|run| field(&run[i])).collect();
                values.sort_by(f64::total_cmp);
                values
            };
            Measurement {
                name,
                bytes: input.len(),
                base_ns: percentile(&sorted(|run| run.base_ns), 0.5),
                head_ns: percentile(&sorted(|run| run.head_ns), 0.5),
                std_ns: percentile(&sorted(|run| run.std_ns), 0.5),
                ratios: sorted(|run| run.ratio),
                speedups: sorted(|run| run.speedup),
            }
        })
        .collect())
}

/// Runs `exe` on one input in a fresh process and reads its timing back.
fn time_in_child(exe: &Path, name: &str, rounds: usize) -> Result<Timing, String> {
    let output = Command::new(exe)
        .args(["--child", name, "--rounds", &rounds.to_string()])
        .output()
        .map_err(|error| format!("failed to start {}: {error}", exe.display()))?;
    let text = String::from_utf8_lossy(&output.stdout);
    let timing = if output.status.success() {
        parse_timing(text.trim_end(), name)
    } else {
        None
    };
    timing.ok_or_else(|| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        format!("{} failed on {name}:\n{text}{stderr}", exe.display())
    })
}

fn parse_timing(line: &str, name: &str) -> Option<Timing> {
    let mut fields = line.split('\t');
    if fields.next()? != name {
        return None;
    }
    let mut number = || fields.next()?.parse().ok();
    Some(Timing {
        own_ns: number()?,
        std_ns: number()?,
    })
}

/// The README's standard-library baseline, which skips counting for ASCII.
fn std_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

fn measure(input: &str, rounds: usize) -> Timing {
    let own = |s: &str| simd_utf16_len::utf16_len(s);
    let standard = |s: &str| std_len(s);

    // std gets its own iteration count, since it can be 70x slower and would
    // otherwise dominate the time.
    let iters = calibrate(&own, input);
    let std_iters = calibrate(&standard, input);
    for _ in 0..WARMUP_ROUNDS {
        time_batch(&own, input, iters);
        time_batch(&standard, input, std_iters);
    }

    let per_call_ns = |d: Duration, iters: u64| d.as_secs_f64() * 1e9 / (2 * iters) as f64;
    let mut own_ns = Vec::with_capacity(rounds);
    let mut std_ns = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        // Own, std, std, own: both sit at the same average position, so drift
        // within a round affects them equally.
        let o1 = time_batch(&own, input, iters);
        let s1 = time_batch(&standard, input, std_iters);
        let s2 = time_batch(&standard, input, std_iters);
        let o2 = time_batch(&own, input, iters);
        own_ns.push(per_call_ns(o1 + o2, iters));
        std_ns.push(per_call_ns(s1 + s2, std_iters));
    }
    own_ns.sort_by(f64::total_cmp);
    std_ns.sort_by(f64::total_cmp);

    Timing {
        own_ns: percentile(&own_ns, 0.5),
        std_ns: percentile(&std_ns, 0.5),
    }
}

/// The iteration count that makes one batch take at least `BATCH`.
fn calibrate(f: &impl Fn(&str) -> usize, input: &str) -> u64 {
    let mut iters = 1;
    while time_batch(f, input, iters) < BATCH {
        iters *= 2;
    }
    iters
}

/// Kept out of line so `utf16_len` and std each run their own copy of the loop.
#[inline(never)]
fn time_batch(f: &impl Fn(&str) -> usize, input: &str, iters: u64) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f(black_box(input)));
    }
    start.elapsed()
}

/// Nearest-rank percentile of ascending `values`.
fn percentile(values: &[f64], p: f64) -> f64 {
    values[((values.len() - 1) as f64 * p).round() as usize]
}

struct Environment {
    base: String,
    head: String,
    cpu: String,
    features: Vec<&'static str>,
    rustc: String,
}

impl Environment {
    fn detect() -> Self {
        let label = |var, default: &str| std::env::var(var).unwrap_or_else(|_| default.to_owned());
        Self {
            base: label("AB_BASE_LABEL", "base"),
            head: label("AB_HEAD_LABEL", "head"),
            cpu: cpu_model().unwrap_or_else(|| "unknown CPU".to_owned()),
            features: cpu_features(),
            rustc: command_output("rustc", &["--version"])
                .unwrap_or_else(|| "unknown rustc".to_owned()),
        }
    }
}

fn cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let field = |text: String, key: &str| {
            text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_owned())
            })
        };
        // aarch64 Linux has no "model name" in /proc/cpuinfo, but lscpu decodes the part number.
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| field(info, "model name"))
            .or_else(|| command_output("lscpu", &[]).and_then(|info| field(info, "Model name")))
    }
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(target_os = "windows")]
    {
        // PROCESSOR_IDENTIFIER only has the family and model numbers.
        command_output(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
        )
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// The SIMD extensions that decide which code path runs.
fn cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if is_x86_feature_detected!("avx512bw") {
            features.push("avx512bw");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
    }
    features
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (output.status.success() && !text.is_empty()).then(|| text.to_owned())
}

fn percent(change: f64) -> String {
    format!("{:+.1}%", change * 100.0)
}

fn markdown(
    results: &[Measurement],
    regressions: &[&str],
    options: &Options,
    env: &Environment,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "### Perf A/B on {} {}: `{}` → `{}`\n",
        std::env::consts::OS,
        std::env::consts::ARCH,
        env.base,
        env.head,
    )
    .unwrap();
    writeln!(
        out,
        "{} ({}), {}, {} runs of {} rounds, each side in its own fresh process per input. A negative change means head is faster.\n",
        env.cpu,
        env.features.join(", "),
        env.rustc,
        options.runs,
        options.rounds,
    )
    .unwrap();
    out.push_str("| Input | Bytes | Base ns/call | Head ns/call | Time change | Range across runs | Runs >1% slower |\n");
    out.push_str("|:------|------:|-------------:|-------------:|------------:|------------------:|----------------:|\n");
    for m in results {
        writeln!(
            out,
            "| {} | {} | {:.1} | {:.1} | {} | {} to {} | {}/{} |",
            m.name,
            m.bytes,
            m.base_ns,
            m.head_ns,
            percent(m.change(0.5)),
            percent(m.change(0.0)),
            percent(m.change(1.0)),
            m.slower_runs(),
            m.ratios.len(),
        )
        .unwrap();
    }
    out.push('\n');
    match options.fail_above {
        None => out.push_str("Report only: no failure threshold was set.\n"),
        Some(limit) if regressions.is_empty() => {
            writeln!(
                out,
                "No input got more than {limit}% slower with every run agreeing."
            )
            .unwrap();
        }
        Some(limit) => {
            let names: Vec<_> = regressions.iter().map(|name| format!("`{name}`")).collect();
            writeln!(
                out,
                "**More than {limit}% slower, with every run agreeing:** {}",
                names.join(", ")
            )
            .unwrap();
        }
    }

    out.push_str("\n#### Head vs the standard library\n\n");
    out.push_str("Report only. The baseline returns the length for ASCII input and otherwise uses `encode_utf16().count()`. Above 1x means head is faster.\n\n");
    out.push_str("| Input | Bytes | Head ns/call | Std ns/call | Speedup |\n");
    out.push_str("|:------|------:|-------------:|------------:|--------:|\n");
    for m in results {
        writeln!(
            out,
            "| {} | {} | {:.1} | {:.1} | {:.1}x |",
            m.name,
            m.bytes,
            m.head_ns,
            m.std_ns,
            m.speedup(),
        )
        .unwrap();
    }
    out
}

fn json(
    results: &[Measurement],
    regressions: &[&str],
    options: &Options,
    env: &Environment,
) -> String {
    let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let list = |items: &[&str]| {
        items
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = String::from("{\n");
    writeln!(out, "  \"base\": {},", quote(&env.base)).unwrap();
    writeln!(out, "  \"head\": {},", quote(&env.head)).unwrap();
    writeln!(out, "  \"os\": {},", quote(std::env::consts::OS)).unwrap();
    writeln!(out, "  \"arch\": {},", quote(std::env::consts::ARCH)).unwrap();
    writeln!(out, "  \"cpu\": {},", quote(&env.cpu)).unwrap();
    writeln!(out, "  \"features\": [{}],", list(&env.features)).unwrap();
    writeln!(out, "  \"rustc\": {},", quote(&env.rustc)).unwrap();
    writeln!(out, "  \"runs\": {},", options.runs).unwrap();
    writeln!(out, "  \"rounds\": {},", options.rounds).unwrap();
    match options.fail_above {
        Some(limit) => writeln!(out, "  \"fail_above_pct\": {limit},").unwrap(),
        None => out.push_str("  \"fail_above_pct\": null,\n"),
    }
    writeln!(out, "  \"regressions\": [{}],", list(regressions)).unwrap();
    out.push_str("  \"results\": [\n");
    for (i, m) in results.iter().enumerate() {
        let separator = if i + 1 < results.len() { "," } else { "" };
        writeln!(
            out,
            "    {{\"name\": {}, \"bytes\": {}, \"base_ns\": {:.3}, \"head_ns\": {:.3}, \"std_ns\": {:.3}, \"change_pct\": {:.2}, \"min_pct\": {:.2}, \"max_pct\": {:.2}, \"slower_runs\": {}, \"speedup\": {:.2}}}{separator}",
            quote(m.name),
            m.bytes,
            m.base_ns,
            m.head_ns,
            m.std_ns,
            m.change(0.5) * 100.0,
            m.change(0.0) * 100.0,
            m.change(1.0) * 100.0,
            m.slower_runs(),
            m.speedup(),
        )
        .unwrap();
    }
    out.push_str("  ]\n}\n");
    out
}

fn append(path: &str, text: &str) {
    use std::io::Write as _;
    let written = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .and_then(|mut file| file.write_all(text.as_bytes()));
    if let Err(error) = written {
        eprintln!("failed to append to {path}: {error}");
    }
}
