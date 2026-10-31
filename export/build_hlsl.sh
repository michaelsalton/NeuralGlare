#!/usr/bin/env bash
# Compile the engine entry points (slang/engine_glare.slang) to plain HLSL for both engines.
# Run from Git Bash on Windows:  bash export/build_hlsl.sh
# Writes engines/unity/NeuralGlareURP/Assets/NeuralGlare/Generated/*.hlsl (inside Assets/ so Unity
# recompiles the .compute wrappers when it changes) and engines/unreal/NeuralGlarePlugin/Generated/*.ush.
# Those folders are output only: never edit them, rerun this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLANGC="${SLANGC:-/d/Development/Libraries/Slang/bin/slangc.exe}"
EXPECTED_VERSION="2026.12"  # the compiler slangpy 0.43.1 is built against (decision log 2026-10-25)

SOURCE="$ROOT/slang/engine_glare.slang"
UNITY_OUT="$ROOT/engines/unity/NeuralGlareURP/Assets/NeuralGlare/Generated"
UNREAL_OUT="$ROOT/engines/unreal/NeuralGlarePlugin/Generated"
KERNELS=(Prefilter BlurH BlurV Upsample)

version="$("$SLANGC" -v 2>&1 | tr -d '\r')"
if [[ "$version" != "$EXPECTED_VERSION" ]]; then
    echo "error: $SLANGC is $version, expected $EXPECTED_VERSION" >&2
    exit 1
fi

mkdir -p "$UNITY_OUT" "$UNREAL_OUT"

# slangc 2026.12 writes one entry point per HLSL output, so one file per kernel.
#   -no-mangle                 keep parameter names (_Source, ...): Unity binds by name
#   -line-directive-mode none  no #line pointing at this machine's paths
for kernel in "${KERNELS[@]}"; do
    out="$UNITY_OUT/glare_$kernel.hlsl"
    "$SLANGC" "$SOURCE" -I "$ROOT/slang" \
        -target hlsl -profile cs_5_0 \
        -no-mangle -line-directive-mode none \
        -entry "$kernel" -o "$out"

    # Engine rule: scalar MLP path only.
    if grep -qiE 'coopvec|cooperative|linalg' "$out"; then
        echo "error: $out uses cooperative-vector or LinAlg intrinsics" >&2
        exit 1
    fi

    cp "$out" "$UNREAL_OUT/glare_$kernel.ush"
    echo "$kernel -> glare_$kernel.hlsl, glare_$kernel.ush"
done

# Record the compiler next to its output.
for dir in "$UNITY_OUT" "$UNREAL_OUT"; do
    printf 'slangc %s\nsource slang/engine_glare.slang\n' "$version" > "$dir/slangc_version.txt"
done
