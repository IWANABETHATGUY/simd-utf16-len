//! Times `utf16_len` from two builds of this crate in one process.
//!
//! `head` is the working tree and `base` is the ref exported by
//! `scripts/perf-ab.sh`. Both sides alternate in short batches on the same
//! machine, so runner noise affects them equally. Each input reports the
//! median of its per-round head/base time ratios.

use std::fmt::Write as _;
use std::hint::black_box;
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";
const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";
const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";
const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

/// Target duration of one timed batch.
const BATCH: Duration = Duration::from_millis(1);
const DEFAULT_ROUNDS: usize = 101;
const WARMUP_ROUNDS: usize = 10;

const USAGE: &str =
    "usage: simd-utf16-len-ab [--fail-above <percent>] [--json <path>] [--rounds <n>]";

struct Options {
    /// Fail when an input's median time change exceeds this many percent.
    fail_above: Option<f64>,
    json: Option<String>,
    rounds: usize,
}

struct Measurement {
    name: &'static str,
    bytes: usize,
    base_ns: f64,
    head_ns: f64,
    /// Per-round head/base time ratios, sorted ascending.
    ratios: Vec<f64>,
}

impl Measurement {
    /// Change in time per call at percentile `p`; negative means head is faster.
    fn change(&self, p: f64) -> f64 {
        percentile(&self.ratios, p) - 1.0
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

    let inputs = [
        ("ascii", ASCII.to_owned()),
        ("cjk", CJK.to_owned()),
        ("emoji", EMOJI.to_owned()),
        ("mixed", MIXED.to_owned()),
        ("ascii_large", ASCII.repeat(64)),
        ("cjk_large", CJK.repeat(64)),
        ("emoji_large", EMOJI.repeat(64)),
        ("mixed_large", MIXED.repeat(64)),
    ];

    for (name, input) in &inputs {
        let (base, head) = (base::utf16_len(input), head::utf16_len(input));
        if base != head {
            eprintln!("{name}: base returned {base} but head returned {head}");
            return ExitCode::from(2);
        }
    }

    let results: Vec<_> = inputs
        .iter()
        .map(|(name, input)| measure(name, input, options.rounds))
        .collect();
    let regressions: Vec<_> = match options.fail_above {
        Some(limit) => results
            .iter()
            .filter(|m| m.change(0.5) * 100.0 > limit)
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
        fail_above: None,
        json: None,
        rounds: DEFAULT_ROUNDS,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--fail-above" => {
                let limit = value()?;
                options.fail_above = Some(
                    limit
                        .parse()
                        .map_err(|_| format!("invalid percent: {limit}"))?,
                );
            }
            "--json" => options.json = Some(value()?),
            "--rounds" => {
                let rounds = value()?;
                options.rounds = rounds
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("invalid round count: {rounds}"))?;
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(options)
}

fn measure(name: &'static str, input: &str, rounds: usize) -> Measurement {
    let base = |s: &str| base::utf16_len(s);
    let head = |s: &str| head::utf16_len(s);

    // Size batches from head, then give both sides the same iteration count.
    let mut iters = 1;
    while time_batch(&head, input, iters) < BATCH {
        iters *= 2;
    }
    for _ in 0..WARMUP_ROUNDS {
        time_batch(&base, input, iters);
        time_batch(&head, input, iters);
    }

    let per_call_ns = |d: Duration| d.as_secs_f64() * 1e9 / (2 * iters) as f64;
    let mut base_ns = Vec::with_capacity(rounds);
    let mut head_ns = Vec::with_capacity(rounds);
    let mut ratios = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        // Base, head, head, base: drift within a round affects both sides equally.
        let b1 = time_batch(&base, input, iters);
        let h1 = time_batch(&head, input, iters);
        let h2 = time_batch(&head, input, iters);
        let b2 = time_batch(&base, input, iters);
        let (b, h) = (b1 + b2, h1 + h2);
        ratios.push(h.as_secs_f64() / b.as_secs_f64());
        base_ns.push(per_call_ns(b));
        head_ns.push(per_call_ns(h));
    }
    for values in [&mut base_ns, &mut head_ns, &mut ratios] {
        values.sort_by(f64::total_cmp);
    }

    Measurement {
        name,
        bytes: input.len(),
        base_ns: percentile(&base_ns, 0.5),
        head_ns: percentile(&head_ns, 0.5),
        ratios,
    }
}

/// Kept out of line so each side runs its own copy of the loop.
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
        std::env::var("PROCESSOR_IDENTIFIER").ok()
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
        "{} ({}), {}, {} rounds. A negative change means head is faster.\n",
        env.cpu,
        env.features.join(", "),
        env.rustc,
        options.rounds,
    )
    .unwrap();
    out.push_str(
        "| Input | Bytes | Base ns/call | Head ns/call | Time change | Middle 50% of rounds |\n",
    );
    out.push_str(
        "|:------|------:|-------------:|-------------:|------------:|---------------------:|\n",
    );
    for m in results {
        writeln!(
            out,
            "| {} | {} | {:.1} | {:.1} | {} | {} to {} |",
            m.name,
            m.bytes,
            m.base_ns,
            m.head_ns,
            percent(m.change(0.5)),
            percent(m.change(0.25)),
            percent(m.change(0.75)),
        )
        .unwrap();
    }
    out.push('\n');
    match options.fail_above {
        None => out.push_str("Report only: no failure threshold was set.\n"),
        Some(limit) if regressions.is_empty() => {
            writeln!(out, "No input got more than {limit}% slower.").unwrap();
        }
        Some(limit) => {
            let names: Vec<_> = regressions.iter().map(|name| format!("`{name}`")).collect();
            writeln!(out, "**More than {limit}% slower:** {}", names.join(", ")).unwrap();
        }
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
            "    {{\"name\": {}, \"bytes\": {}, \"base_ns\": {:.3}, \"head_ns\": {:.3}, \"change_pct\": {:.2}, \"p25_pct\": {:.2}, \"p75_pct\": {:.2}}}{separator}",
            quote(m.name),
            m.bytes,
            m.base_ns,
            m.head_ns,
            m.change(0.5) * 100.0,
            m.change(0.25) * 100.0,
            m.change(0.75) * 100.0,
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
