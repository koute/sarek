use {
    std::{
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
        RngCore,
        seq::{
            SliceRandom
        }
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
            model::{
                Model
            },
            training_opts::{
                TrainingOpts
            }
        }
    }
};

pub struct Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    batch_size: usize,
    model_instance: ModelInstance,
    data_set: DataSet< I, O >,
    position: usize,
    indexes: Vec< usize >
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
    pub fn new( ctx: &Context, model: Model, data_set: DataSet< I, O > ) -> Result< Self, ModelCompilationError > {
        Self::new_with_opts( ctx, model, data_set, TrainingOpts::new() )
    }

    pub fn new_with_opts( ctx: &Context, model: Model, data_set: DataSet< I, O >, training_opts: TrainingOpts )
        -> Result< Self, ModelCompilationError >
    {
        let batch_size = training_opts.batch_size.unwrap_or( 32 );
        let model_instance = ModelInstance::compile( ctx, model, Some( training_opts ) )?;

        let input_shape = model_instance.input_shape();
        assert_eq!(
            data_set.input_shape(),
            input_shape,
            "Model's input shape is {}, and yet the training input data's shape is {}",
            input_shape,
            data_set.input_shape()
        );

        let output_shape = model_instance.output_shape();
        assert_eq!(
            data_set.output_shape(),
            output_shape,
            "Model's output shape is {}, and yet the training output data's shape is {}",
            output_shape,
            data_set.output_shape()
        );

        let length = data_set.len();
        let mut trainer = Trainer {
            model_instance,
            batch_size,
            data_set,
            position: 0,
            indexes: (0..length).collect()
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

    pub fn test< S >( &self, inputs: &S ) -> f32 where S: DataSource {
        unimplemented!();
    }

    /// Trains the network for the whole epoch (the whole data set)
    /// and returns the training loss.
    pub fn train( &mut self ) -> f32 {
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

            info!( "Finished a training epoch in {}s with a loss of {}", now.elapsed().as_secs(), loss );
            loss
        })
    }

    fn shuffle( &mut self ) {
        let mut rng = rand::thread_rng();
        self.shuffle_with_rng( &mut rng );
    }

    fn shuffle_with_rng( &mut self, rng: &mut RngCore ) {
        self.indexes.shuffle( rng );
    }
}
