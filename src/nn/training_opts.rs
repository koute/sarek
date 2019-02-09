use {
    crate::{
        nn::{
            optimizers::{
                Optimizer,
                OptimizerAdam
            }
        }
    }
};

pub struct TrainingOpts {
    pub(crate) optimizer: Optimizer,
    pub(crate) batch_size: Option< usize >,
    pub(crate) pretrain_weights: bool,
    pub(crate) normalize_inputs: bool
}

impl TrainingOpts {
    pub fn new() -> Self {
        TrainingOpts {
            optimizer: OptimizerAdam::new().into(),
            batch_size: None,
            pretrain_weights: true,
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

    pub fn disable_weight_pretraining( &mut self ) {
        self.pretrain_weights = false;
    }

    pub fn disable_input_normalization( &mut self ) {
        self.normalize_inputs = false;
    }
}
