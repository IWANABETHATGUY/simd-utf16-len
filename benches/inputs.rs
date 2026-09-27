//! Benchmark inputs shared by `benches/utf16_len.rs`, `examples/bench_compare.rs`,
//! and `perf/ab`, grouped by the code path they exercise.

// Each includer uses a different subset.
#![allow(dead_code)]

pub const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";

pub const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";

pub const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";

pub const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

/// Every input, from a few bytes to about 11 KB.
pub fn all() -> Vec<(&'static str, String)> {
    let long_ascii = ASCII.repeat(64);
    vec![
        // Shorter than one 16-byte vector.
        ("ascii_tiny", "hello world".to_owned()),
        ("utf8_tiny", "héllo wörld".to_owned()),
        // Shorter than one 64-byte ASCII block, so the ASCII scan works a word at a time.
        (
            "ascii_short",
            "The quick brown fox jumps over the lazy dog.".to_owned(),
        ),
        ("utf8_short", "Привет, мир! Как дела сегодня?".to_owned()),
        ("ascii", ASCII.to_owned()),
        ("cjk", CJK.to_owned()),
        ("emoji", EMOJI.to_owned()),
        ("mixed", MIXED.to_owned()),
        // Leave 3 and 15 bytes after the last full 16-byte vector.
        ("cjk_tail3", "中".repeat(65)),
        ("cjk_tail15", "中".repeat(69)),
        // Longer than one 4,080-byte batch of the counting loop.
        ("ascii_large", long_ascii.clone()),
        ("cjk_large", CJK.repeat(64)),
        ("emoji_large", EMOJI.repeat(64)),
        ("mixed_large", MIXED.repeat(64)),
        // Mostly ASCII, like source code with one non-ASCII character.
        ("early_non_ascii", format!("é{long_ascii}")),
        ("late_non_ascii", format!("{long_ascii}é")),
    ]
}
