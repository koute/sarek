use {
    log::{
        info
    },
    rand::{
        prelude::{
            *
        }
    },
    std::{
        error::{
            Error
        },
        fmt
    },
    crate::{
        backend::{
            keras::{
                Context,
                ModelCompilationError,
                ModelInstance,
                SetWeightsError,
                ortho_weights
            }
        },
        core::{
            data_source::{
                DataSource
            },
            name::{
                Name
            },
            raw_array_source::{
                RawArraySource
            }
        },
        nn::{
            layers::{
                Layer,
                LayerPrototype
            },
            model::{
                Model
            }
        }
    }
};

fn calculate_variance( buffer: &[f32] ) -> f32 {
    let mean = buffer.iter().cloned().sum::< f32 >() / buffer.len() as f32;
    buffer.iter().cloned().map( |x| {
        let a = x - mean;
        a * a
    }).sum::< f32 >() / buffer.len() as f32
}

#[test]
fn test_calculate_variance() {
    let var = calculate_variance( &[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] );
    assert_eq!( var, 8.25 );
}

fn predict_layer< L, I >( ctx: &Context, layer: L, input_data: I, weights: &[f32] )
    -> Result< RawArraySource, InitializeWeightsError >
    where L: Into< Layer >,
          I: DataSource + Send + Sync
{
    let layer_name = Name::new_unique();
    let mut layer = layer.into();
    layer.set_name( layer_name.clone() );

    let input_shape = input_data.shape();
    let weight_count = layer.weight_count( &input_shape );
    assert_eq!( weight_count, weights.len() );

    let model = Model::new_sequential( input_data.shape(), layer );
    let mut instance = ModelInstance::compile( ctx, model, None )?;
    if !weights.is_empty() {
        instance.set_weights( layer_name, weights )?;
    }

    let output = instance.predict_raw( &input_data );
    Ok( output )
}

#[derive(Debug)]
pub enum InitializeWeightsError {
    ModelCompilationError( ModelCompilationError ),
    SetWeightsError( SetWeightsError )
}

impl From< ModelCompilationError > for InitializeWeightsError {
    fn from( value: ModelCompilationError ) -> Self {
        InitializeWeightsError::ModelCompilationError( value )
    }
}

impl From< SetWeightsError > for InitializeWeightsError {
    fn from( value: SetWeightsError ) -> Self {
        InitializeWeightsError::SetWeightsError( value )
    }
}

impl fmt::Display for InitializeWeightsError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        let error: &Error = match self {
            InitializeWeightsError::ModelCompilationError( ref error ) => error,
            InitializeWeightsError::SetWeightsError( ref error ) => error
        };

        write!( fmt, "failed to LSUV initialize weights: {}", error )
    }
}

impl Error for InitializeWeightsError {}

pub fn initialize_weights< I >(
    ctx: &Context,
    rng: &mut RngCore,
    model_instance: &mut ModelInstance,
    input_data: I
) -> Result< (), InitializeWeightsError >
    where I: DataSource + Send + Sync
{
    info!( "Starting LSUV weight initialization..." );

    let target_variance = 1.0;
    let allowed_margin = 0.1;
    let iteration_limit = 10;

    let batch_size = std::cmp::min( 128, input_data.len() );
    let mut indexes: Vec< _ > = (0..input_data.len()).into_iter().collect();
    indexes.shuffle( rng );
    indexes.truncate( batch_size );

    let mut input_shape = model_instance.input_shape();
    let mut input_buffer = RawArraySource::new_uninitialized( batch_size, input_shape.clone(), input_data.data_type() );
    input_data.gather_bytes_into(
        &indexes,
        input_buffer.as_bytes_mut()
    );

    let layers = model_instance.model().layers.clone();

    let layer_count = {
        let mut count = 0;
        let mut input_shape = input_shape.clone();
        for (layer_index, layer) in layers.iter().enumerate() {
            let weight_count = layer.weight_count( &input_shape );
            input_shape = layer.output_shape( &input_shape );

            if weight_count != 0 {
                count = layer_index;
            }
        }
        count
    };

    for (layer_index, layer) in layers.into_iter().enumerate().take( layer_count ) {
        let weight_count = layer.weight_count( &input_shape );
        let next_input_shape = layer.output_shape( &input_shape );
        if weight_count == 0 {
            info!( "Layer #{}: no weights; only running prediction", layer_index + 1 );
            input_buffer = predict_layer( ctx, layer, input_buffer, &[] )?;
            input_shape = next_input_shape;
            continue;
        }

        match layer {
            Layer::Dense( layer ) => {
                let input_count = input_shape.product() + 1;
                let output_count = layer.size;
                assert_eq!( weight_count, input_count * output_count );

                info!( "Layer #{}: initializing weights", layer_index + 1 );
                let mut weights = ortho_weights( (input_count, output_count).into() ); // TODO
                let weights: &mut [f32] = weights.as_slice_mut().unwrap();
                assert_eq!( weights.len(), weight_count );

                let mut output_buffer = predict_layer( ctx, layer.clone(), &input_buffer, weights )?;
                for iteration in 0..iteration_limit {
                    let variance = calculate_variance( output_buffer.as_slice().expect( "expected an f32 layer output" ) );
                    info!( "Layer #{}: iteration #{}: variance = {}", layer_index + 1, iteration + 1, variance );
                    if (target_variance - variance).abs() <= allowed_margin {
                        break;
                    }

                    let divisor = variance.sqrt();
                    for weight in weights.iter_mut() {
                        *weight /= divisor;
                    }

                    output_buffer = predict_layer( ctx, layer.clone(), &input_buffer, weights )?;
                }

                model_instance.set_weights( layer.name(), &weights )?;
                input_buffer = output_buffer;
            },
            Layer::Convolution( layer ) => {
                // TODO: Initialize the weights here.
                use crate::core::array::ToArrayRef;
                info!( "Layer #{}: using defaults", layer_index + 1 );
                let weights = model_instance.get_weights( layer.name() );
                input_buffer = predict_layer( ctx, &layer, input_buffer, weights.to_slice::< f32 >().unwrap() )?;
                input_shape = next_input_shape;
                continue;
            },
            _ => unreachable!()
        }

        input_shape = next_input_shape;
    }

    log::info!( "Initialized weights of the model!" );
    Ok(())
}
