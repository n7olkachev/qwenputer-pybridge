FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc-riscv64-unknown-elf picolibc-riscv64-unknown-elf make python3 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /work
