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
            Context,
            ModelCompilationError,
            ModelInstance
        },
        core::{
            data_set::{
                DataSet
            },
            data_source::{
                DataSource,
                DataSourceExt,
                DataSourceList,
                DataSourceListExt
            },
            slice_source::{
                SliceSource
            }
        },
        nn::{
            layers::{
                LayerAdd,
                LayerConstant,
                LayerMul
            },
            model::{
                Model
            },
            model::{
                NullaryLayer
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
    where I: DataSourceList + Send + Sync,
          O: DataSourceList + Send + Sync
{
    batch_size: usize,
    model_instance: ModelInstance,
    data_set: DataSet< I, O >,
    position: usize,
    indexes: Vec< usize >,
    epoch_counter: usize,
    rng: Pcg32,
    input_element_sizes: Vec< usize >,
    output_element_sizes: Vec< usize >
}

impl< I, O > Deref for Trainer< I, O >
    where I: DataSourceList + Send + Sync,
          O: DataSourceList + Send + Sync
{
    type Target = ModelInstance;
    fn deref( &self ) -> &Self::Target {
        &self.model_instance
    }
}

impl< I, O > DerefMut for Trainer< I, O >
    where I: DataSourceList + Send + Sync,
          O: DataSourceList + Send + Sync
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.model_instance
    }
}

fn average_over< I, F >( input_data: &I, callback: F ) -> Vec< f64 >
    where I: DataSource,
          F: Fn( usize, f64 ) -> f64
{
    let element_size = input_data.shape().product();
    let mut results = Vec::new();
    results.resize( element_size, 0.0 );

    let chunk_size = 128;
    let mut buffer: Vec< f32 > = Vec::new();
    let buffer_size = chunk_size * element_size;
    buffer.reserve( buffer_size );
    unsafe { buffer.set_len( buffer_size ); }

    for training_chunk in input_data.chunks( chunk_size ) {
        let chunk_size = training_chunk.len();
        let buffer = &mut buffer[ ..chunk_size * element_size ];
        training_chunk.gather_into( .., buffer );
        for element in buffer.chunks_exact( element_size ) {
            debug_assert_eq!( element.len(), results.len() );

            let iter = results.iter_mut()
                .zip( element.iter().cloned() );

            for (index, (result, value)) in iter.enumerate() {
                *result += callback( index, f64::from( value ) );
            }
        }
    }

    let length = input_data.len() as f64;
    for result in results.iter_mut() {
        *result /= length;
    }

    results
}

fn normalize_inputs< I >( input_data: &I ) -> (Vec< f32 >, Vec< f32 >)
    where I: DataSource
{
    info!( "Calculating input normalization matrices..." );

    let mean_shift = average_over( input_data, |_, x| -x );
    let mut variance_adjustment = average_over( input_data, |index, x| {
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
    where I: DataSourceList + Send + Sync,
          O: DataSourceList + Send + Sync
{
    pub fn new( ctx: &Context, model: Model, data_set: DataSet< I, O > ) -> Result< Self, TrainerInitializationError > {
        Self::new_with_opts( ctx, model, data_set, TrainingOpts::new() )
    }

    pub fn new_with_opts( ctx: &Context, mut model: Model, data_set: DataSet< I, O >, training_opts: TrainingOpts )
        -> Result< Self, TrainerInitializationError >
    {
        let total_weight_count: usize = model.node_indexes().map( |node_index| {
            let node = model.get_node( node_index );
            model.weight_count_of( node )
        }).sum();

        info!( "Creating a trainer for a model with {} weights...", total_weight_count );

        for (io, input_data) in model.inputs().zip( data_set.input_list().data_sources() ) {
            assert_eq!(
                input_data.shape(),
                io.shape,
                "Model's input #{} has a shape of {}, and yet the training input data's shape is {}",
                io.index,
                io.shape,
                input_data.shape()
            );

            assert_eq!(
                input_data.data_type(),
                io.data_type,
                "Model's input #{} has a data type of {}, and yet the training input data's data type is {}",
                io.index,
                io.data_type,
                input_data.data_type()
            );
        }

        for (io, output_data) in model.outputs().zip( data_set.expected_output_list().data_sources() ) {
            assert_eq!(
                output_data.shape(),
                io.shape,
                "Model's output #{} has a shape of {}, and yet the training output data's shape is {}",
                io.index,
                io.shape,
                output_data.shape()
            );

            assert_eq!(
                output_data.data_type(),
                io.data_type,
                "Model's output #{} has a data type of {}, and yet the training output data's data type is {}",
                io.index,
                io.data_type,
                output_data.data_type()
            );
        }

        let input_element_sizes: Vec< _ > =
            model.inputs().map( |io| io.shape.product() * io.data_type.byte_size() ).collect();
        let output_element_sizes: Vec< _ > =
            model.outputs().map( |io| io.shape.product() * io.data_type.byte_size() ).collect();

        if training_opts.normalize_inputs && data_set.len() > 1 {
            let model_inputs: Vec< _ > = model.inputs().collect();
            let model_outputs: Vec< _ > = model.outputs().collect();
            model.modify( |builder| {
                for (io, input_data) in model_inputs.into_iter().zip( data_set.input_list().data_sources() ) {
                    let (mean_shift, variance_adjustment) = normalize_inputs( &input_data );
                    let mean_shift = SliceSource::from( io.shape.clone(), mean_shift );
                    let variance_adjustment = SliceSource::from( io.shape.clone(), variance_adjustment );

                    let outputs = builder.get_node( io.node_index ).outputs();
                    let input_node = builder.get_node( io.node_index );
                    let normalized_input_node = input_node
                        .clone()
                        .chain_into_first_input(
                            LayerConstant::new( mean_shift ).into_node( builder ),
                            LayerAdd::new()
                        )
                        .chain_into_first_input(
                            LayerConstant::new( variance_adjustment ).into_node( builder ),
                            LayerMul::new()
                        );

                    for output_node in outputs {
                        output_node.replace_input( input_node.clone(), normalized_input_node.clone() );
                    }

                    let model_outputs_to_replace = model_outputs.iter()
                        .filter( |output| output.node_index == input_node.node_index() )
                        .map( |output| output.index );

                    for model_output_index in model_outputs_to_replace {
                        builder.set_output( model_output_index, normalized_input_node.clone() );
                    }
                }
            });
        }

        let mut rng = Pcg32::seed_from_u64( 123456 );
        initialize_weights( ctx, &mut rng, &mut model, &data_set.input_list() )?;

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
            rng,
            input_element_sizes,
            output_element_sizes
        };

        trainer.shuffle();

        info!( "Model is ready for training" );
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

        let length = self.data_set.len();
        let epoch_size = length;

        let mut batch_index = 0;
        let mut count = 0;
        let position = &mut self.position;
        let indexes = &mut self.indexes;
        let rng = &mut self.rng;
        let data_set = &self.data_set;
        let batch_size = self.batch_size;
        let input_element_sizes = &self.input_element_sizes;
        let output_element_sizes = &self.output_element_sizes;

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

                for (nth, input_data) in data_set.input_list().data_sources().enumerate() {
                    let element_size = input_element_sizes[ nth ];
                    let buffer = inputs.get_buffer_mut( nth );
                    input_data.gather_bytes_into(
                        index,
                        &mut buffer[ batch_index * element_size..(batch_index + 1) * element_size ]
                    );
                }

                for (nth, output_data) in data_set.expected_output_list().data_sources().enumerate() {
                    let element_size = output_element_sizes[ nth ];
                    let buffer = outputs.get_buffer_mut( nth );
                    output_data.gather_bytes_into(
                        index,
                        &mut buffer[ batch_index * element_size..(batch_index + 1) * element_size ]
                    );
                }

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
