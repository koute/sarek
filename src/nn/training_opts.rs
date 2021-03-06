use {
    crate::{
        nn::{
            optimizers::{
                Optimizer,
                OptimizerNadam
            }
        }
    }
};

pub struct TrainingOpts {
    pub(crate) optimizer: Optimizer,
    pub(crate) batch_size: Option< usize >,
    pub(crate) normalize_inputs: bool
}

impl TrainingOpts {
    pub fn new() -> Self {
        TrainingOpts {
            optimizer: OptimizerNadam::new().into(),
            batch_size: None,
            normalize_inputs: true
        }
    }

    pub fn optimizer( &self ) -> &Optimizer {
        &self.optimizer
    }

    pub fn set_optimizer< T >( &mut self, optimizer: T ) -> &mut Self where T: Into< Optimizer > {
        self.optimizer = optimizer.into();
        self
    }

    pub fn set_batch_size( &mut self, batch_size: usize ) {
        assert_ne!( batch_size, 0, "The batch size cannot be zero" );
        self.batch_size = Some( batch_size );
    }

    pub fn disable_input_normalization( &mut self ) {
        self.normalize_inputs = false;
    }
}
