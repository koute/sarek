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
        }
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
            data_source::{
                DataSource
            },
            raw_array_source::{
                RawArraySource
            },
            ortho_generator::{
                OrthogonalGenerator
            }
        },
        nn::{
            layers::{
                Layer,
                LayerConvolution,
                LayerDense,
                LayerPrototype,
                Weights
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

fn predict_layer< L, I >( ctx: &Context, layer: L, input_data: I )
    -> Result< RawArraySource, InitializeWeightsError >
    where L: Into< Layer >,
          I: DataSource + Send + Sync
{
    let model = Model::new_sequential( input_data.shape(), layer );
    let mut instance = ModelInstance::compile( ctx, model, None )?;
    let output = instance.predict_raw( &input_data );
    Ok( output )
}

#[derive(Debug, Display, From)]
pub enum InitializeWeightsError {
    #[display(fmt = "failed to initialize weights: {}", "_0")]
    ModelCompilationError( ModelCompilationError )
}

impl Error for InitializeWeightsError {}

fn get_initialization_depth( model: &Model ) -> usize {
    let mut count = 0;
    let mut input_shape = model.input_shape();
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let weight_count = layer.weight_count( &input_shape );
        input_shape = layer.output_shape( &input_shape );

        if weight_count != 0 {
            let has_preset_weights = match layer {
                Layer::Dense( LayerDense { weights, .. } ) => weights.is_some(),
                Layer::Convolution( LayerConvolution { weights, .. } ) => weights.is_some(),
                _ => false
            };

            if !has_preset_weights {
                count = layer_index + 1;
            }
        }
    }
    count
}

fn generate_normal_weights( rng: &mut RngCore, bias_count: usize, weight_count: usize, n: usize ) -> Weights {
    let factor = 2.0_f32;
    let n = n as f32;
    let stddev = (factor / n).sqrt();
    let dist = rand::distributions::Normal::new( 0.0, stddev as f64 );

    let mut weights = Weights::new();
    weights.get_mut().extend( (0..bias_count).map( |_| 0.0 ) );
    weights.get_mut().extend( (bias_count..weight_count).map( |_| dist.sample( rng ) as f32 ) );

    weights
}

fn generate_initial_weights( depth: usize, rng: &mut RngCore, model: &mut Model ) -> Vec< usize > {
    info!( "Populating the model with weights..." );

    let mut initialized_layers = Vec::new();
    let mut input_shape = model.input_shape();
    for (layer_index, layer) in model.layers.iter_mut().take( depth ).enumerate() {
        let output_shape = layer.output_shape( &input_shape );
        let weight_count = layer.weight_count( &input_shape );

        let generated_weights = match *layer {
            Layer::Dense( ref mut layer ) => {
                let bias_count = layer.size;

                let orthogonal_init = true;
                if orthogonal_init {
                    let mut weights = Weights::new();
                    weights.get_mut().extend( (0..bias_count).map( |_| 0.0 ) );

                    let mut generator = OrthogonalGenerator::new();
                    generator.generate_into( layer.size, input_shape.product(), rng, weights.get_mut() );
                    Some( weights )
                } else {
                    let n = input_shape.product();
                    let weights = generate_normal_weights( rng, bias_count, weight_count, n );
                    Some( weights )
                }
            },
            Layer::Convolution( ref layer ) => {
                let bias_count = layer.filter_count;
                let n = output_shape.x() * output_shape.y() * input_shape.z();
                let weights = generate_normal_weights( rng, bias_count, weight_count, n );
                Some( weights )
            },
            _ => {
                assert_eq!( weight_count, 0 );
                None
            }
        };

        if let Some( generated_weights ) = generated_weights {
            layer.set_weights( generated_weights );
            initialized_layers.push( layer_index );
        }
        input_shape = output_shape;
    }

    initialized_layers.reverse();
    initialized_layers
}

fn normalize_layer_output< T >(
    ctx: &Context,
    layer: &mut T,
    layer_index: usize,
    target_variance: f32,
    target_mean: f32,
    bias_count: usize,
    input_buffer: &RawArraySource,
    mut output_buffer: RawArraySource
) -> Result< RawArraySource, InitializeWeightsError >
    where T: LayerPrototype + Into< Layer > + Clone
{
    let iteration_limit = 10;
    let allowed_margin = 0.01;

    let (mut mean, mut variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );
    for iteration in 0..iteration_limit {
        info!( "Layer #{}: iteration #{}: variance = {}, mean = {}", layer_index + 1, iteration + 1, variance, mean );

        if (target_variance - variance).abs() <= allowed_margin {
            break;
        }

        let divisor = variance.sqrt() / target_variance.sqrt();
        let mut weights = layer.take_weights().unwrap();
        for weight in weights.get_mut()[ bias_count.. ].iter_mut() {
            *weight /= divisor;
        }
        layer.set_weights( weights );

        output_buffer = predict_layer( ctx, layer.clone(), &input_buffer )?;
        let (new_mean, new_variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );

        mean = new_mean;
        variance = new_variance;
    }

    let mut weights = layer.take_weights().unwrap();
    for weight in weights.get_mut()[ ..bias_count ].iter_mut() {
        *weight -= mean - target_mean;
    }
    layer.set_weights( weights );

    output_buffer = predict_layer( ctx, layer.clone(), &input_buffer )?;
    Ok( output_buffer )
}

pub fn initialize_weights< I >
(
    ctx: &Context,
    rng: &mut RngCore,
    model: &mut Model,
    input_data: I
) -> Result< (), InitializeWeightsError >
    where I: DataSource + Send + Sync
{
    let depth = get_initialization_depth( model );
    if depth == 0 {
        return Ok(());
    }

    let mut initialized_layers = generate_initial_weights( depth, rng, model );

    info!( "Normalizing layer outputs..." );

    let batch_size = std::cmp::min( 128, input_data.len() );
    let indexes = {
        let mut indexes: Vec< _ > = (0..input_data.len()).into_iter().collect();
        indexes.shuffle( rng );
        indexes.truncate( batch_size );
        indexes
    };

    let mut input_shape = model.input_shape();
    let mut input_buffer = RawArraySource::new_uninitialized( batch_size, input_shape.clone(), input_data.data_type() );
    input_data.gather_bytes_into(
        &indexes,
        input_buffer.as_bytes_mut()
    );

    for layer_index in 0..depth {
        let layer = &mut model.layers[ layer_index ];

        let weight_count = layer.weight_count( &input_shape );
        let output_shape = layer.output_shape( &input_shape );

        if initialized_layers.last().cloned() != Some( layer_index ) {
            if weight_count == 0 {
                info!( "Layer #{}: no weights; only running prediction", layer_index + 1 );
            } else {
                info!( "Layer #{}: weights already set; only running prediction", layer_index + 1 );
            }

            let output_buffer = predict_layer( ctx, layer, input_buffer )?;
            if log_enabled!( log::Level::Debug ) {
                let (mean, variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );
                debug!( "Layer #{} output: variance = {}, mean = {}", layer_index + 1, variance, mean );
            }

            input_buffer = output_buffer;
            input_shape = output_shape;
            continue;
        }

        initialized_layers.pop();

        let output_buffer = predict_layer( ctx, layer.clone(), &input_buffer )?;
        let output_buffer = match layer {
            Layer::Dense( layer ) => {
                info!( "Layer #{}: initializing weights for a dense layer", layer_index + 1 );
                let bias_count = layer.size;
                let target_variance = 0.9;
                let target_mean = 0.02;

                normalize_layer_output(
                    ctx,
                    layer,
                    layer_index,
                    target_variance,
                    target_mean,
                    bias_count,
                    &input_buffer,
                    output_buffer
                )
            },
            Layer::Convolution( layer ) => {
                info!( "Layer #{}: initializing weights for a convolution layer", layer_index + 1 );
                let bias_count = layer.filter_count;
                let target_variance = 1.0;
                let target_mean = 0.01;

                normalize_layer_output(
                    ctx,
                    layer,
                    layer_index,
                    target_variance,
                    target_mean,
                    bias_count,
                    &input_buffer,
                    output_buffer
                )
            },
            _ => unreachable!()
        }?;

        input_buffer = output_buffer;
        input_shape = output_shape;
    }

    info!( "Initialized weights of the model!" );
    Ok(())
}
