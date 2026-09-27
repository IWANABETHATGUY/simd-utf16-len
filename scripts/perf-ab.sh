#!/usr/bin/env bash
# Compare `utf16_len` in the working tree against a base ref, in one binary.
#
# Usage: scripts/perf-ab.sh [base-ref] [--fail-above <percent>] [--json <path>] [--rounds <n>]
#
# The base ref defaults to `main`. Pass `HEAD` to measure the no-change spread.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
base_ref=${1:-main}
if [ $# -gt 0 ]; then
  shift
fi

base_dir="$root/target/ab-base"
rm -rf "$base_dir"
mkdir -p "$base_dir"
# Keep LF line endings, which Git on Windows would otherwise convert to CRLF.
git -C "$root" -c core.autocrlf=false archive "$base_ref" | tar -x -C "$base_dir"
# Rename the base package so Cargo can link it next to the working-tree copy.
# Perl on Windows can't edit in place without a backup file.
perl -pi.bak -e 's/^name = "simd-utf16-len"$/name = "simd-utf16-len-base"/' "$base_dir/Cargo.toml"
rm -f "$base_dir/Cargo.toml.bak"
if ! grep -q '^name = "simd-utf16-len-base"' "$base_dir/Cargo.toml"; then
  echo "failed to rename the base package in $base_dir/Cargo.toml" >&2
  exit 1
fi

AB_BASE_LABEL=$(git -C "$root" rev-parse --short "$base_ref^{commit}")
AB_HEAD_LABEL=$(git -C "$root" rev-parse --short "${AB_HEAD_REF:-HEAD}^{commit}")
if ! git -C "$root" diff --quiet HEAD --; then
  AB_HEAD_LABEL="$AB_HEAD_LABEL with local changes"
fi
export AB_BASE_LABEL AB_HEAD_LABEL

# Start every function on a 16 KB boundary, so both copies of identical code
# share the low address bits that CPU caches and branch predictors index by.
# With 64-byte alignment, no-change runs still showed short inputs up to 8%
# slower on one side. Windows object files can't align sections beyond 8 KB.
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) align_log2=13 ;;
  *) align_log2=14 ;;
esac
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C llvm-args=-align-all-functions=$align_log2"

cargo run --release --quiet --manifest-path "$root/perf/ab/Cargo.toml" -- "$@"
