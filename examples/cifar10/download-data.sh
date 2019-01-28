#!/bin/sh

set -euo pipefail

if [ ! -d data/cifar-10-batches-bin ]; then
    mkdir -p data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    tar -xf cifar-10-binary.tar.gz
    rm -f cifar-10-binary.tar.gz

    echo "CIFAR-10 data set downloaded!"
else
    echo "CIFAR-10 data set is already downloaded."
fi
