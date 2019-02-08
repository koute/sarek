use {
    std::{
        error::{
            Error
        },
        fmt,
        ops::{
            Deref,
            DerefMut
        },
        time::{
            Instant
        }
    },
    log::{
        info
    },
    rand::{
        SeedableRng,
        seq::{
            SliceRandom
        }
    },
    rand_pcg::{
        Pcg32
    },
    crate::{
        backend::{
            keras::{
                context::{
                    Context
                },
                model::{
                    ModelCompilationError,
                    ModelInstance
                },
                py_array::{
                    PyArray
                }
            }
        },
        core::{
            data_set::{
                DataSet
            },
            data_source::{
                DataSource
            }
        },
        nn::{
            lsuv::{
                InitializeWeightsError,
                initialize_weights
            },
            model::{
                Model
            },
            training_opts::{
                TrainingOpts
            }
        }
    }
};

#[derive(Debug)]
pub enum TrainerInitializationError {
    ModelCompilationError( ModelCompilationError ),
    InitializeWeightsError( InitializeWeightsError )
}

impl fmt::Display for TrainerInitializationError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        let error: &Error = match self {
            TrainerInitializationError::ModelCompilationError( ref error ) => error,
            TrainerInitializationError::InitializeWeightsError( ref error ) => error
        };

        write!( fmt, "failed to initialize a model for training: {}", error )
    }
}

impl From< ModelCompilationError > for TrainerInitializationError {
    fn from( value: ModelCompilationError ) -> Self {
        TrainerInitializationError::ModelCompilationError( value )
    }
}

impl From< InitializeWeightsError > for TrainerInitializationError {
    fn from( value: InitializeWeightsError ) -> Self {
        TrainerInitializationError::InitializeWeightsError( value )
    }
}

impl Error for TrainerInitializationError {}

pub struct Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    batch_size: usize,
    model_instance: ModelInstance,
    data_set: DataSet< I, O >,
    position: usize,
    indexes: Vec< usize >,
    epoch_counter: usize,
    rng: Pcg32
}

impl< I, O > Deref for Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    type Target = ModelInstance;
    fn deref( &self ) -> &Self::Target {
        &self.model_instance
    }
}

impl< I, O > DerefMut for Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.model_instance
    }
}

impl< I, O > Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    pub fn new( ctx: &Context, model: Model, data_set: DataSet< I, O > ) -> Result< Self, TrainerInitializationError > {
        Self::new_with_opts( ctx, model, data_set, TrainingOpts::new() )
    }

    pub fn new_with_opts( ctx: &Context, model: Model, data_set: DataSet< I, O >, training_opts: TrainingOpts )
        -> Result< Self, TrainerInitializationError >
    {
        let input_shape = model.input_shape();
        assert_eq!(
            data_set.input_shape(),
            input_shape,
            "Model's input shape is {}, and yet the training input data's shape is {}",
            input_shape,
            data_set.input_shape()
        );

        let output_shape = model.output_shape();
        assert_eq!(
            data_set.output_shape(),
            output_shape,
            "Model's output shape is {}, and yet the training output data's shape is {}",
            output_shape,
            data_set.output_shape()
        );

        let batch_size = training_opts.batch_size.unwrap_or( 32 );
        let pretrain_weights = training_opts.pretrain_weights;
        let mut model_instance = ModelInstance::compile( ctx, model, Some( training_opts ) )?;
        let mut rng = Pcg32::seed_from_u64( 123456 );

        if pretrain_weights {
            initialize_weights( ctx, &mut rng, &mut model_instance, data_set.input_data() )?;
        }

        let length = data_set.len();
        let mut trainer = Trainer {
            model_instance,
            batch_size,
            data_set,
            position: 0,
            indexes: (0..length).collect(),
            epoch_counter: 0,
            rng
        };

        trainer.shuffle();
        Ok( trainer )
    }

    pub fn set_batch_size( &mut self, batch_size: usize ) {
        assert_ne!( batch_size, 0, "The batch size cannot be zero" );
        self.batch_size = batch_size;
    }

    pub fn model_instance( &mut self ) -> &mut ModelInstance {
        &mut self.model_instance
    }

    pub fn data_set( &self ) -> &DataSet< I, O > {
        &self.data_set
    }

    /// Trains the network for the whole epoch (the whole data set)
    /// and returns the training loss.
    pub fn train( &mut self ) -> f32 {
        if self.epoch_counter == 0 {
            info!( "Starting training on a data set with {} elements...", self.data_set.len() );
        }

        let now = Instant::now();
        Context::gil( move |py| {
            let input_data = self.data_set.input_data();
            let output_data = self.data_set.expected_output_data();
            let length = self.data_set.len();
            let epoch_size = length;

            let mut inputs = PyArray::new( py, input_data.shape().prepend( self.batch_size ), input_data.data_type() );
            let mut outputs = PyArray::new( py, output_data.shape().prepend( self.batch_size ), output_data.data_type() );

            let input_element_size = input_data.shape().product() * input_data.data_type().byte_size();
            let output_element_size = output_data.shape().product() * output_data.data_type().byte_size();

            let mut batch_index = 0;
            let mut loss = 0.0;
            let mut count = 0;
            while count != epoch_size {
                if self.position >= length {
                    self.position = 0;
                    self.shuffle();
                }

                let index = self.indexes[ self.position ];
                self.position += 1;
                count += 1;

                self.data_set.input_data().gather_bytes_into(
                    index,
                    &mut inputs.as_bytes_mut()[ batch_index * input_element_size..(batch_index + 1) * input_element_size ]
                );

                self.data_set.expected_output_data().gather_bytes_into(
                    index,
                    &mut outputs.as_bytes_mut()[ batch_index * output_element_size..(batch_index + 1) * output_element_size ]
                );

                batch_index += 1;
                if batch_index == self.batch_size {
                    batch_index = 0;
                    loss += self.model_instance.train_on_batch( py, &inputs, &outputs );
                }
            }

            self.epoch_counter += 1;
            info!( "Finished a training epoch #{} in {}s with a loss of {}", self.epoch_counter, now.elapsed().as_secs(), loss );
            loss
        })
    }

    fn shuffle( &mut self ) {
        self.indexes.shuffle( &mut self.rng );
    }
}
