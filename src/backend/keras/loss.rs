#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[non_exhaustive]
pub enum Loss {
    MeanSquaredError,

    /// A loss function for multiclass classification tasks.
    ///
    /// To be used if the output categories are encoded as
    /// integers, e.g.:
    ///    * \[1\]
    ///    * \[2\]
    ///    * \[3\]
    SparseCategoricalCrossEntropy,

    /// A loss function for multiclass classification tasks.
    ///
    /// To be used if the output categories are encoded using
    /// an one-hot encoding, e.g.:
    ///    * \[1, 0, 0\]
    ///    * \[0, 1, 0\]
    ///    * \[0, 0, 1\]
    CategoricalCrossEntropy
}
