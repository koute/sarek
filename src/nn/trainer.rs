use {
    std::{
        error::{
            Error
        },
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
                Context,
                ModelCompilationError,
                ModelInstance
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
            layers::{
                LayerMultiply,
                LayerShift
            },
            model::{
                Model
            },
            training_opts::{
                TrainingOpts
            },
            weight_init::{
                InitializeWeightsError,
                initialize_weights
            }
        }
    }
};

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum TrainerInitializationError {
    #[display(fmt = "failed to initialize a model for training: {}", "_0")]
    ModelCompilationError( ModelCompilationError ),
    #[display(fmt = "failed to initialize a model for training: {}", "_0")]
    InitializeWeightsError( InitializeWeightsError )
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

fn average_over< I, O, F >( data_set: &DataSet< I, O >, callback: F ) -> Vec< f64 >
    where I: DataSource + Send + Sync,
            O: DataSource + Send + Sync,
            F: Fn( usize, f64 ) -> f64
{
    let element_size = data_set.input_data().shape().product();
    let mut results = Vec::new();
    results.resize( element_size, 0.0 );

    let chunk_size = 128;
    let mut buffer: Vec< f32 > = Vec::new();
    let buffer_size = chunk_size * element_size;
    buffer.reserve( buffer_size );
    unsafe { buffer.set_len( buffer_size ); }

    for training_chunk in data_set.chunks( chunk_size ) {
        let chunk_size = training_chunk.len();
        let buffer = &mut buffer[ ..chunk_size * element_size ];
        training_chunk.input_data().gather_into( .., buffer );
        for element in buffer.chunks_exact( element_size ) {
            debug_assert_eq!( element.len(), results.len() );

            let iter = results.iter_mut()
                .zip( element.iter().cloned() );

            for (index, (result, value)) in iter.enumerate() {
                *result += callback( index, value as f64 );
            }
        }
    }

    let length = data_set.len() as f64;
    for result in results.iter_mut() {
        *result /= length;
    }

    results
}

fn normalize_inputs< I, O >( data_set: &DataSet< I, O > ) -> (Vec< f32 >, Vec< f32 >)
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    info!( "Calculating input normalization matrices..." );

    let mean_shift = average_over( &data_set, |_, x| -x );
    let mut variance_adjustment = average_over( &data_set, |index, x| {
        let a = x + mean_shift[ index ];
        a * a
    });

    for value in variance_adjustment.iter_mut() {
        if *value > 0.0000001 {
            *value = 1.0 / value.sqrt();
        } else {
            *value = 1.0;
        }
    }

    let mean_shift: Vec< _ > = mean_shift.into_iter().map( |x| x as f32 ).collect();
    let variance_adjustment: Vec< _ > = variance_adjustment.into_iter().map( |x| x as f32 ).collect();

    (mean_shift, variance_adjustment)
}

impl< I, O > Trainer< I, O >
    where I: DataSource + Send + Sync,
          O: DataSource + Send + Sync
{
    pub fn new( ctx: &Context, model: Model, data_set: DataSet< I, O > ) -> Result< Self, TrainerInitializationError > {
        Self::new_with_opts( ctx, model, data_set, TrainingOpts::new() )
    }

    pub fn new_with_opts( ctx: &Context, mut model: Model, data_set: DataSet< I, O >, training_opts: TrainingOpts )
        -> Result< Self, TrainerInitializationError >
    {
        let total_weight_count: usize = model.layers.iter()
            .scan( model.input_shape(), |input_shape, layer| {
                use crate::nn::layers::LayerPrototype;
                let weight_count = layer.weight_count( &input_shape );
                let output_shape = layer.output_shape( &input_shape );
                *input_shape = output_shape;
                Some( weight_count )
            })
            .sum();

        info!( "Creating a trainer for a model with {} weights...", total_weight_count );

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

        if training_opts.normalize_inputs {
            let (mean_shift, variance_adjustment) = normalize_inputs( &data_set );
            let layers = model.layers;
            model.layers = Vec::with_capacity( layers.len() + 1 );

            model.layers.push( LayerShift::new( mean_shift ).into() );
            model.layers.push( LayerMultiply::new( variance_adjustment ).into() );
            model.layers.extend( layers.into_iter() );
        }

        let mut rng = Pcg32::seed_from_u64( 123456 );
        initialize_weights( ctx, &mut rng, &mut model, data_set.input_data() )?;

        let batch_size = training_opts.batch_size.unwrap_or( 32 );
        let model_instance = ModelInstance::compile( ctx, model, Some( training_opts ) )?;
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

        let input_data = self.data_set.input_data();
        let output_data = self.data_set.expected_output_data();
        let length = self.data_set.len();
        let epoch_size = length;

        let input_element_size = input_data.shape().product() * input_data.data_type().byte_size();
        let output_element_size = output_data.shape().product() * output_data.data_type().byte_size();

        let mut batch_index = 0;
        let mut count = 0;
        let position = &mut self.position;
        let indexes = &mut self.indexes;
        let rng = &mut self.rng;
        let data_set = &self.data_set;
        let batch_size = self.batch_size;

        let now = Instant::now();
        let loss = self.model_instance.train_for_epoch( batch_size, move |inputs, outputs| {
            while count != epoch_size {
                if *position >= length {
                    *position = 0;
                    indexes.shuffle( rng );
                }

                let index = indexes[ *position ];
                *position += 1;
                count += 1;

                data_set.input_data().gather_bytes_into(
                    index,
                    &mut inputs[ batch_index * input_element_size..(batch_index + 1) * input_element_size ]
                );

                data_set.expected_output_data().gather_bytes_into(
                    index,
                    &mut outputs[ batch_index * output_element_size..(batch_index + 1) * output_element_size ]
                );

                batch_index += 1;
                if batch_index == batch_size {
                    batch_index = 0;
                    return true;
                }
            }

            false
        });

        self.epoch_counter += 1;
        info!( "Finished a training epoch #{} in {}s with a loss of {}", self.epoch_counter, now.elapsed().as_secs(), loss );
        loss
    }

    fn shuffle( &mut self ) {
        self.indexes.shuffle( &mut self.rng );
    }
}
