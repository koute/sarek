[![Build Status](https://api.travis-ci.org/koute/sarek.svg)](https://travis-ci.org/koute/sarek)

# A work-in-progress, experimental neural network library for Rust

[![Documentation](https://docs.rs/sarek/badge.svg)](https://docs.rs/sarek/*/sarek/)

The goal of this crate is to provide a Rust neural network library which is simple,
easy to use, minimal, fast and production ready.

**This is still experimental!** Seriously, I mean it.

## Requirements

Currently this only works on Rust nightly and requires Python 3.7, TensorFlow 1.12, and Numpy 1.15
**at runtime** as it's using TensorFlow Keras (through Python!). Might work with older versions;
no guarantees though.

Warning: TensorFlow 1.12 (and possibly 1.11) has a broken dropout layer; if possible use 1.13

## Tentative short-term roadmap

   * [X] Support TensorFlow Keras as a backend (battle tested, but very heavy)
      * [ ] Support basic layer types:
         * [X] Fully connected
         * [X] Dropout
         * [X] Activation
            * [X] Logistic
            * [X] TanH
            * [X] ReLU
            * [X] LeakyReLU
            * [X] ELU
         * [X] Softmax
         * [ ] Batch normalization
         * [ ] Convolutional
         * [ ] Pooling
         * [ ] RNN (maybe)
   * [X] Add a MNIST example
   * [X] Add LSUV weight initialization
      * [ ] Replace the random orthogonal matrix generator with a pure Rust one
   * [ ] Add a native backend (reimplement the compute parts using pure Rust code)
      * [ ] Use multiple threads
      * [ ] Use SIMD
   * [ ] Make the Python + TensorFlow dependency optional (compile time)
   * [ ] Make the Python + TensorFlow dependency optional (runtime, so that the user doesn't need libpython3.so)
   * [ ] Add full API documentation and add `#![deny(missing_docs)]`

## Long term roadmap

   * Add other backends:
      * [ ] Figure out a compute abstraction:
         * a) Define a custom Rust-like compute language and a source-level translator
         * b) Write the compute parts in Rust, compile to WASM and write a recompiler of WASM bytecode
         * c) Write a SIPR-V recompiler, and write the compute parts in something that can compile to SPIR-V
      * [ ] OpenCL
      * [ ] Vulkan
      * [ ] WebGL (?)
   * [ ] Export a C API
   * [ ] Compile to WebAssembly and publish to NPM as a JS library

## I don't know anything about machine learning nor neural networks; what is this *actually* for?

In a nutshell a neural network is a mathematical construct which can automatically learn how to transform
data from one representation into another one. Say, for example, that you have an old black and white photo
of your grandma, and you'd like to colorize it. A neural network can help you with that!

What you would do in such a case is to find a bunch of photos which are already in color and convert them
to black and white. You've just got yourself a data set for training a neural network! Now you can tell
a neural network - hey, here's a bunch of black and white photos, and here are the same photos but they're
in color; please learn how to transform the black and white photos into the color ones!

And that's what the neural network will do. If properly trained your neural network will learn
how to *generalize*, or in other words - it will be able to turn more-or-less *any* black and white
photo into a color one, even if it never saw that particular photo before!

## Contributing

If you want to contribute something significant (especially expose new stuff from TensorFlow)
please create an issue first to discuss it. This is not a research-level library, nor it is
supposed to be a huge kitchen sink.

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
