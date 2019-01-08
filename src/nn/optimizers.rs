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
pub struct OptimizerAdam {
    pub(crate) learning_rate: decorum::Ordered< f32 >
}

impl OptimizerAdam {
    pub fn new() -> OptimizerAdam {
        OptimizerAdam {
            learning_rate: 0.001.into()
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
    Adam( OptimizerAdam )
}

impl From< OptimizerSGD > for Optimizer {
    #[inline]
    fn from( optimizer: OptimizerSGD ) -> Self {
        Optimizer::SGD( optimizer )
    }
}

impl From< OptimizerAdam > for Optimizer {
    #[inline]
    fn from( optimizer: OptimizerAdam ) -> Self {
        Optimizer::Adam( optimizer )
    }
}
