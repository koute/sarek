#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export RUST_BACKTRACE=1

export PYTHON_SYS_EXECUTABLE=python3
echo "Python version: `python3 --version`"

cargo build
cargo build --example mnist
cargo build --example cifar10
