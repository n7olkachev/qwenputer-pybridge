#!/bin/bash
# Build build/mpy-bridge.elf: the P4 MicroPython bridge firmware for the chipformer RV32 machine.
# Host build: brew's riscv64-elf-gcc (rv32im/ilp32 libgcc multilib) + Debian's picolibc package
# (third_party/picolibc, Architecture: all) for libc/libm, as in the author's Makefile.
#   [OUT=build/mpy-bridge-dev.elf] [BUILD=build-rom-dev] [COPT_EXTRA=-DRV_VM_BUDGET=5000] firmware/build.sh [make args]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
MP=third_party/micropython
P=$MP/ports/chipformer
mkdir -p "$P" build
# the parser patch (FIRST-set pruning, firmware/patches/parse-first-sets.patch), once
grep -q MICROPY_PARSE_FIRST_SETS "$MP/py/parse.c" || (cd "$MP" && git apply "$ROOT/firmware/patches/parse-first-sets.patch")
grep -q MICROPY_PY_BUILTINS_STR_SLICE_STEP "$MP/py/objstrunicode.c" || (cd "$MP" && git apply "$ROOT/firmware/patches/str-slice-step.patch")
cp firmware/mpy/*.c firmware/mpy/*.h firmware/mpy/Makefile firmware/mpy/manifest.py "$P"/ && rm -rf "$P/modules" && cp -r firmware/mpy/modules "$P"/
HOT="py/parse py/lexer py/qstr py/map py/vm"
COPT="-DMICROPY_OPT_COMPUTED_GOTO=1 -DMICROPY_OPT_MAP_LOOKUP_CACHE=1 -DMICROPY_GC_CONSERVATIVE_CLEAR=0 -DMICROPY_ALLOC_PARSE_CHUNK_INIT=512 ${COPT_EXTRA:-}"
MAKE=$(command -v gmake || command -v make)
BUILD=${BUILD:-build-rom}  # the default build dir; the shipped firmware is the extmod build (see mpconfigport.h)
(cd "$P" && $MAKE -j8 CONFIG=basic MAP=rom BUILD="$BUILD" \
    CROSS_COMPILE=riscv64-elf- \
    PICO="$ROOT/third_party/picolibc/usr/riscv64-unknown-elf" \
    PICOLIB="$ROOT/third_party/picolibc/usr/riscv64-unknown-elf/lib/rv32im/ilp32" \
    ENV="$ROOT/qwenputer/experimental/riscv/chipformer/rv/env" \
    HOT="$HOT" COPT="$COPT" "$@" > "$ROOT/build/mpy-build.log" 2>&1) \
  || { tail -40 "$ROOT/build/mpy-build.log"; exit 1; }
grep -E '^\.(text|rodata|data|bss|tdata|tbss)' "$ROOT/build/mpy-build.log" | tail -6
OUT=${OUT:-build/mpy-bridge.elf}
cp "$P/$BUILD/firmware.elf" "$OUT"
riscv64-elf-size "$OUT"
