#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export RUST_BACKTRACE=1

cargo build
cargo build --example mnist
cargo build --example cifar10
