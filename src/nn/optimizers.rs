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
    pub(crate) learning_rate: decorum::Ordered< f32 >,
    pub(crate) beta_1: decorum::Ordered< f32 >,
    pub(crate) beta_2: decorum::Ordered< f32 >,
    pub(crate) epsilon: decorum::Ordered< f32 >,
    pub(crate) schedule_decay: decorum::Ordered< f32 >
}

impl OptimizerNadam {
    pub fn new() -> OptimizerNadam {
        OptimizerNadam {
            learning_rate: 0.00225.into(),
            beta_1: 0.9.into(),
            beta_2: 0.999.into(),
            epsilon: 1e-7.into(),
            schedule_decay: 0.004.into()
        }
    }

    pub fn set_learning_rate( &mut self, value: f32 ) -> &mut Self {
        self.learning_rate = value.into();
        self
    }

    pub fn set_beta_1( &mut self, value: f32 ) -> &mut Self {
        self.beta_1 = value.into();
        self
    }

    pub fn set_beta_2( &mut self, value: f32 ) -> &mut Self {
        self.beta_2 = value.into();
        self
    }

    pub fn set_epsilon( &mut self, value: f32 ) -> &mut Self {
        self.epsilon = value.into();
        self
    }

    pub fn set_schedule_decay( &mut self, value: f32 ) -> &mut Self {
        self.schedule_decay = value.into();
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
