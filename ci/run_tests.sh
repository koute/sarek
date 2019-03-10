#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export RUST_BACKTRACE=1

echo "Python version: `python --version`"

cargo build
cargo build --example mnist
cargo build --example cifar10
