#!/bin/bash
# the essentials for a GPU box: firmware ELF + sources, tools, bench, gpu scripts, README (no build trees)
set -e
cd "$(dirname "$0")/.."
mkdir -p build
tar czf build/qwenputer-pybridge-gpu.tar.gz \
  README.md gpu tools bench/*.py bench/*.sh bench/data bench/results/*.direct.* \
  build/mpy-bridge.elf firmware/mpy firmware/patches firmware/build.sh paper/paper.md
ls -la build/qwenputer-pybridge-gpu.tar.gz | awk '{print $5 " bytes", $9}'
