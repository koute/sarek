use {
    decorum
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OptimizerSGD {
    pub(crate) learning_rate: decorum::Ordered< f32 >
}

impl OptimizerSGD {
    pub fn new() -> OptimizerSGD {
        OptimizerSGD {
            learning_rate: 0.01.into()
        }
    }

    pub fn set_learning_rate( &mut self, value: f32 ) -> &mut Self {
        self.learning_rate = value.into();
        self
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OptimizerNadam {
    pub(crate) learning_rate: decorum::Ordered< f32 >
}

impl OptimizerNadam {
    pub fn new() -> OptimizerNadam {
        OptimizerNadam {
            learning_rate: 0.00225.into()
        }
    }

    pub fn set_learning_rate( &mut self, value: f32 ) -> &mut Self {
        self.learning_rate = value.into();
        self
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[non_exhaustive]
pub enum Optimizer {
    SGD( OptimizerSGD ),
    Nadam( OptimizerNadam )
}

impl From< OptimizerSGD > for Optimizer {
    #[inline]
    fn from( optimizer: OptimizerSGD ) -> Self {
        Optimizer::SGD( optimizer )
    }
}

impl From< OptimizerNadam > for Optimizer {
    #[inline]
    fn from( optimizer: OptimizerNadam ) -> Self {
        Optimizer::Nadam( optimizer )
    }
}
