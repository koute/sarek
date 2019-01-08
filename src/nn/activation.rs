/// A type of activation layer.
///
/// If you don't know which one to use then use `ELU`,
/// as it tends to be the best performing one([1]).
///
/// [1]: https://arxiv.org/pdf/1710.11272.pdf
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[non_exhaustive]
pub enum Activation {
    /// The most basic, always positive, saturating activation type.
    ///
    /// Its behavior is arguably the worst because:
    ///
    ///   * it's saturating, so it kills your gradient when
    ///     backpropagating if your neuron's output is close
    ///     to the regions of saturation,
    ///   * its output is always positive, which unnecessairly
    ///     constraints your gradients - they're going to be
    ///     either all positive, or all negative.
    ///
    /// Do not use it if you can help it.
    ///
    /// `Logistic` activation type is described by the following equation:
    ///
    /// ```text
    /// 1.0 / (1.0 + exp(x))
    /// ```
    Logistic,

    /// Basic saturating activation type.
    ///
    /// `TanH` is an improvement over `Logistic` as it's output
    /// isn't always positive anymore, however it's still saturating.
    ///
    /// `TanH` activation type is described by the following equation:
    ///
    /// ```text
    /// tanh(x)
    /// ```
    TanH,

    /// Always positive, non-saturating activation type, most often used in deep networks.
    ///
    /// `ReLU` is even better than `TanH` - it's not saturating (only in positives),
    /// and it converges a lot (often several times) faster in practice
    /// than both `Logistic` and `TanH`.
    ///
    /// However, the negative saturation behavior of ReLU is still a problem.
    /// Since the ReLU activations clamp all negative outputs to zero it's
    /// possible for some of them to end up with such weights that will make
    /// them effectively dead, never (or extremely rarely) activating for any data
    /// that we might realistically feed to our network.
    ///
    /// Technically dead ReLU neurons could be "fixed" by running our network
    /// a few times gathering statistics as to how many times each neuron
    /// was activated, and based on that information we could then reinitialize
    /// our dead neurons and continue training.
    ///
    /// `ReLU` activation type is described by the following equation:
    ///
    /// ```text
    /// x if x >= 0.0
    /// 0 if x <  0.0
    /// ````
    ReLU,

    /// `LeakyReLU` is an attempt to fix the `ReLU` by having a very slight leakage
    /// where a normal ReLU would be otherwise dead, so its outputs are not always
    /// only positive.
    ///
    /// `LeakyReLU` activation type is described by the following equation:
    ///
    /// ```text
    ///        x if x >= 0.0
    /// 0.01 * x if x <  0.0
    /// ````
    LeakyReLU,

    /// `ELU`([1]) is yet another attempt at fixing `ReLU`; It's arguably the superior
    /// activation function of them all.
    ///
    /// [1]: https://arxiv.org/pdf/1511.07289v5.pdf
    ///
    /// `ELU` activation type is described by the following equation:
    ///
    /// ````text
    ///            x if x >= 0.0
    /// exp(x) - 1.0 if x <  0.0
    /// ````
    ELU
}
