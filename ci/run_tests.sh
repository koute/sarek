#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export RUST_BACKTRACE=1

export PYTHON_SYS_EXECUTABLE=python3.7
echo "Python version: `python3.7 --version`"

cargo build
cargo build --example mnist
cargo build --example cifar10
