use {
    log::{
        debug,
        info,
        log_enabled
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
                SetWeightsError
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
            },
            ortho_generator::{
                OrthogonalGenerator
            },
            shape::{
                Shape
            }
        },
        nn::{
            layers::{
                Layer,
                LayerConvolution,
                LayerDense,
                LayerPrototype
            },
            model::{
                Model
            }
        }
    }
};

fn calculate_mean_and_variance( buffer: &[f32] ) -> (f32, f32) {
    let mean = buffer.iter().cloned().sum::< f32 >() / buffer.len() as f32;
    let variance = buffer.iter().cloned().map( |x| {
        let a = x - mean;
        a * a
    }).sum::< f32 >() / buffer.len() as f32;

    (mean, variance)
}

#[test]
fn test_calculate_variance() {
    let (_, variance) = calculate_mean_and_variance( &[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ] );
    assert_eq!( variance, 8.25 );
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

        write!( fmt, "failed to initialize weights: {}", error )
    }
}

impl Error for InitializeWeightsError {}

pub fn initialize_weights< I >
(
    ctx: &Context,
    rng: &mut RngCore,
    model_instance: &mut ModelInstance,
    input_data: I
) -> Result< (), InitializeWeightsError >
    where I: DataSource + Send + Sync
{
    info!( "Starting weight initialization..." );

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
                count = layer_index + 1;
            }
        }
        count
    };

    let mut weights = Vec::new();
    for (layer_index, layer) in layers.iter().enumerate().take( layer_count ) {
        let weight_count = layer.weight_count( &input_shape );
        let output_shape = layer.output_shape( &input_shape );

        if weight_count == 0 {
            info!( "Layer #{}: no weights; only running prediction", layer_index + 1 );
            let output_buffer = predict_layer( ctx, layer, input_buffer, &[] )?;

            if log_enabled!( log::Level::Debug ) {
                let (mean, variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );
                debug!( "Layer #{} output: variance = {}, mean = {}", layer_index + 1, variance, mean );
            }

            input_shape = output_shape;
            input_buffer = output_buffer;
            continue;
        }

        weights.clear();
        let output_buffer = match layer {
            Layer::Dense( layer ) => {
                info!( "Layer #{}: initializing weights for a dense layer", layer_index + 1 );
                initialize_dense_weights( ctx, layer, &layers, layer_index, rng, &input_shape, &input_buffer, &mut weights )
            },
            Layer::Convolution( layer ) => {
                info!( "Layer #{}: initializing weights for a convolution layer", layer_index + 1 );
                initialize_convolutional_weights( ctx, layer, layer_index, rng, &input_shape, &output_shape, &input_buffer, &mut weights )
            },
            _ => unreachable!()
        }?;

        model_instance.set_weights( layer.name(), &weights )?;
        input_buffer = output_buffer;
        input_shape = output_shape;
    }

    info!( "Initialized weights of the model!" );
    Ok(())
}

fn normalize_output(
    ctx: &Context,
    layer: &Layer,
    layer_index: usize,
    target_variance: f32,
    target_mean: f32,
    bias_count: usize,
    weights: &mut Vec< f32 >,
    input_buffer: &RawArraySource,
    mut output_buffer: RawArraySource
) -> Result< RawArraySource, InitializeWeightsError > {
    let iteration_limit = 10;
    let allowed_margin = 0.01;

    let (mut mean, mut variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );
    for iteration in 0..iteration_limit {
        info!( "Layer #{}: iteration #{}: variance = {}, mean = {}", layer_index + 1, iteration + 1, variance, mean );

        if (target_variance - variance).abs() <= allowed_margin {
            break;
        }

        let divisor = variance.sqrt() / target_variance.sqrt();
        for weight in weights[ bias_count.. ].iter_mut() {
            *weight /= divisor;
        }

        output_buffer = predict_layer( ctx, layer.clone(), &input_buffer, &weights )?;
        let (new_mean, new_variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );

        mean = new_mean;
        variance = new_variance;
    }

    for weight in weights[ ..bias_count ].iter_mut() {
        *weight -= mean - target_mean;
    }

    output_buffer = predict_layer( ctx, layer.clone(), &input_buffer, &weights )?;
    Ok( output_buffer )
}

fn initialize_dense_weights(
    ctx: &Context,
    layer: &LayerDense,
    layers: &[Layer],
    layer_index: usize,
    rng: &mut RngCore,
    input_shape: &Shape,
    input_buffer: &RawArraySource,
    weights: &mut Vec< f32 >
) -> Result< RawArraySource, InitializeWeightsError > {
    let weight_count = layer.weight_count( &input_shape );
    let bias_count = layer.size;

    let orthogonal_init = true;
    if orthogonal_init {
        weights.extend( (0..bias_count).map( |_| 0.0 ) );

        let mut generator = OrthogonalGenerator::new();
        generator.generate_into( layer.size, input_shape.product(), rng, weights );
    } else {
        let factor = 2.0_f32;
        let n = input_shape.product() as f32;
        let stddev = (factor / n).sqrt();
        let dist = rand::distributions::Normal::new( 0.0, stddev as f64 );
        weights.extend( (0..bias_count).map( |_| 0.0 ) );
        weights.extend( (bias_count..weight_count).map( |_| dist.sample( rng ) as f32 ) );
    }

    let output_buffer = predict_layer( ctx, layer, input_buffer, &weights )?;
    let next_layer_is_activation = if let Some( &Layer::Activation { .. } ) = layers.get( layer_index + 1 ) {
        true
    } else {
        false
    };

    if !next_layer_is_activation {
        return Ok( output_buffer );
    }

    let target_variance = 0.9;
    let target_mean = 0.02;

    return normalize_output(
        ctx,
        &layer.into(),
        layer_index,
        target_variance,
        target_mean,
        bias_count,
        weights,
        input_buffer,
        output_buffer
    );
}

fn initialize_convolutional_weights(
    ctx: &Context,
    layer: &LayerConvolution,
    layer_index: usize,
    rng: &mut RngCore,
    input_shape: &Shape,
    output_shape: &Shape,
    input_buffer: &RawArraySource,
    weights: &mut Vec< f32 >
) -> Result< RawArraySource, InitializeWeightsError > {
    let weight_count = layer.weight_count( &input_shape );
    let bias_count = layer.filter_count;

    let factor = 2.0_f32;
    let n = (output_shape.x() * output_shape.y() * input_shape.z()) as f32;
    let stddev = (factor / n).sqrt();
    let dist = rand::distributions::Normal::new( 0.0, stddev as f64 );
    weights.extend( (0..bias_count).map( |_| 0.0 ) );
    weights.extend( (bias_count..weight_count).map( |_| dist.sample( rng ) as f32 ) );

    let output_buffer = predict_layer( ctx, layer, input_buffer, &weights )?;
    let target_variance = 1.0;
    let target_mean = 0.01;

    return normalize_output(
        ctx,
        &layer.into(),
        layer_index,
        target_variance,
        target_mean,
        bias_count,
        weights,
        input_buffer,
        output_buffer
    );
}
