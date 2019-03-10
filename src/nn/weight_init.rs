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
        collections::{
            HashSet
        },
        error::{
            Error
        }
    },
    crate::{
        backend::{
            Context,
            ModelCompilationError,
            ModelInstance
        },
        core::{
            data_source::{
                DataSource,
                DataSourceExt,
                DataSourceList,
                DataSourceListExt
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
                AnyUnaryLayer,
                AnyNullaryLayer,
                LayerPrototype,
                Weights
            },
            model::{
                Model
            },
            model::{
                BinaryLayer,
                Node,
                NodeIndex,
                UnaryLayer
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

fn predict_unary_layer< L, I >( ctx: &Context, layer: L, input_data: I )
    -> Result< RawArraySource, InitializeWeightsError >
    where L: UnaryLayer,
          I: DataSource
{
    let model = Model::new_sequential( input_data.shape(), layer );
    let mut instance = ModelInstance::compile( ctx, model, None )?;
    let output = instance.predict_raw( &input_data ).into_iter().next().unwrap();
    Ok( output )
}

fn predict_binary_layer< L, I >( ctx: &Context, layer: L, input_1: I, input_2: I )
    -> Result< RawArraySource, InitializeWeightsError >
    where L: BinaryLayer,
          I: DataSource
{
    let model = Model::new_graph( |builder| {
        let input_1 = builder.add_input( input_1.shape() );
        let input_2 = builder.add_input( input_2.shape() );
        layer.into_node( input_1, input_2 ).add_as_output();
    });

    let mut instance = ModelInstance::compile( ctx, model, None )?;
    let output = instance.predict_raw( &[&input_1, &input_2][..] ).into_iter().next().unwrap();
    Ok( output )
}

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum InitializeWeightsError {
    #[display(fmt = "failed to initialize weights: {}", "_0")]
    ModelCompilationError( ModelCompilationError )
}

impl Error for InitializeWeightsError {}

fn generate_normal_weights( rng: &mut dyn RngCore, bias_count: usize, weight_count: usize, n: usize ) -> Weights {
    let factor = 2.0_f32;
    let n = n as f32;
    let stddev = (factor / n).sqrt();
    let dist = rand::distributions::Normal::new( 0.0, stddev as f64 );

    let mut weights = Weights::new();
    weights.get_mut().extend( (0..bias_count).map( |_| 0.0 ) );
    weights.get_mut().extend( (bias_count..weight_count).map( |_| dist.sample( rng ) as f32 ) );

    weights
}

fn generate_initial_weights( rng: &mut dyn RngCore, model: &mut Model ) -> HashSet< NodeIndex > {
    info!( "Populating the model with weights..." );

    let mut initialized_layers = HashSet::new();
    for node_index in model.node_indexes() {
        let mut input_shapes = model.input_shapes_of( model.get_node( node_index ) );
        let node = model.get_node_mut( node_index );

        match *node {
            Node::Input { .. } => {},
            Node::NullaryNode { .. } => {},
            Node::UnaryNode { ref mut layer, .. } => {
                debug_assert_eq!( input_shapes.len(), 1 );
                let input_shape = input_shapes.next().unwrap();
                let weight_count = layer.weight_count( &input_shape );
                match *layer {
                    AnyUnaryLayer::Dense( ref mut layer ) => {
                        if layer.weights.is_some() {
                            continue;
                        }

                        let bias_count = layer.size;

                        let orthogonal_init = true;
                        if orthogonal_init {
                            let mut weights = Weights::new();
                            weights.get_mut().extend( (0..bias_count).map( |_| 0.0 ) );

                            let mut generator = OrthogonalGenerator::new();
                            generator.generate_into( layer.size, input_shape.product(), rng, weights.get_mut() );
                            layer.set_weights( weights );
                        } else {
                            let n = input_shape.product();
                            let weights = generate_normal_weights( rng, bias_count, weight_count, n );
                            layer.set_weights( weights );
                        }
                    },
                    AnyUnaryLayer::Convolution( ref mut layer ) => {
                        if layer.weights.is_some() {
                            continue;
                        }

                        let output_shape = layer.output_shape( &input_shape );
                        let bias_count = layer.filter_count;
                        let n = output_shape.x() * output_shape.y() * input_shape.z();
                        let weights = generate_normal_weights( rng, bias_count, weight_count, n );
                        layer.set_weights( weights );
                    },
                    _ => {
                        assert_eq!( weight_count, 0 );
                        continue;
                    }
                }

                initialized_layers.insert( node_index );
            },
            Node::BinaryNode { ref mut layer, .. } => {
                debug_assert_eq!( input_shapes.len(), 2 );
                let input_shape_1 = input_shapes.next().unwrap();
                let input_shape_2 = input_shapes.next().unwrap();
                let weight_count = layer.weight_count( &input_shape_1, &input_shape_2 );

                assert_eq!( weight_count, 0 );
                continue;
            }
        }
    }

    initialized_layers
}

fn normalize_layer_output< T, F >(
    layer: &mut T,
    node_index: NodeIndex,
    target_variance: f32,
    target_mean: f32,
    bias_count: usize,
    mut output_buffer: RawArraySource,
    mut predict: F
) -> Result< RawArraySource, InitializeWeightsError >
    where T: LayerPrototype + Clone,
          F: FnMut( T ) -> Result< RawArraySource, InitializeWeightsError >
{
    let iteration_limit = 10;
    let allowed_margin = 0.01;

    let (mut mean, mut variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );
    for iteration in 0..iteration_limit {
        info!( "Layer {}: iteration #{}: variance = {}, mean = {}", node_index, iteration + 1, variance, mean );

        if (target_variance - variance).abs() <= allowed_margin {
            break;
        }

        let divisor = variance.sqrt() / target_variance.sqrt();
        let mut weights = layer.take_weights().unwrap();
        for weight in weights.get_mut()[ bias_count.. ].iter_mut() {
            *weight /= divisor;
        }
        layer.set_weights( weights );

        output_buffer = predict( layer.clone() )?;
        let (new_mean, new_variance) = calculate_mean_and_variance( output_buffer.as_slice().expect( "expected an f32 output" ) );

        mean = new_mean;
        variance = new_variance;
    }

    let mut weights = layer.take_weights().unwrap();
    for weight in weights.get_mut()[ ..bias_count ].iter_mut() {
        *weight -= mean - target_mean;
    }
    layer.set_weights( weights );

    output_buffer = predict( layer.clone() )?;
    Ok( output_buffer )
}

fn collect_input_nodes(
    model: &Model,
    node_index: NodeIndex,
    node_indexes: &mut HashSet< NodeIndex >
) {
    for input_index in model.get_node( node_index ).inputs() {
        if node_indexes.contains( &input_index ) {
            continue;
        }

        node_indexes.insert( input_index );
        collect_input_nodes( model, input_index, node_indexes );
    }
}

pub fn initialize_weights
(
    ctx: &Context,
    rng: &mut dyn RngCore,
    model: &mut Model,
    input_data_list: &dyn DataSourceList
) -> Result< (), InitializeWeightsError >
{
    let initialized_layers = generate_initial_weights( rng, model );
    if initialized_layers.is_empty() || input_data_list.data_sources().len() == 0 {
        return Ok(());
    }

    info!( "Normalizing layer outputs..." );

    let mut nodes_needing_prediction = HashSet::new();
    for &node_index in &initialized_layers {
        collect_input_nodes( model, node_index, &mut nodes_needing_prediction );
    }

    let input_data_length = input_data_list.data_sources().next().unwrap().len();
    assert!( input_data_list.data_sources().all( |src| src.len() == input_data_length ) );

    let batch_size = std::cmp::min( 128, input_data_length );
    let indexes = {
        let mut indexes: Vec< _ > = (0..input_data_length).into_iter().collect();
        indexes.shuffle( rng );
        indexes.truncate( batch_size );
        indexes
    };

    model.traverse_mut( move |model, inputs, node_index| {
        let needs_prediction = nodes_needing_prediction.contains( &node_index );
        let needs_normalization = initialized_layers.contains( &node_index );
        if !needs_prediction && !needs_normalization {
            return Ok( None );
        }

        let mut output_buffer = match *model.get_node( node_index ) {
            Node::Input { input_index, ref shape, .. } => {
                let input_data = &input_data_list.data_sources()[ input_index ];
                let mut output_buffer = RawArraySource::new_uninitialized( batch_size, shape.clone(), input_data.data_type() );
                input_data.gather_bytes_into(
                    &indexes,
                    output_buffer.as_bytes_mut()
                );

                output_buffer
            },
            Node::NullaryNode { ref layer, .. } => {
                match *layer {
                    AnyNullaryLayer::Constant( ref layer ) => {
                        let data = &layer.data;
                        let mut output_buffer = RawArraySource::new_uninitialized( batch_size, data.shape(), data.data_type() );
                        let element_size = data.shape().product() * data.data_type().byte_size();
                        for position in 0..batch_size {
                            data.gather_bytes_into(
                                ..,
                                &mut output_buffer.as_bytes_mut()[ position * element_size..(position + 1) * element_size ]
                            );
                        }

                        output_buffer
                    }
                }
            },
            Node::UnaryNode { ref layer, .. } => {
                assert_eq!( inputs.len(), 1 );
                predict_unary_layer( ctx, layer.clone(), &inputs[0] )?
            },
            Node::BinaryNode { ref layer, .. } => {
                assert_eq!( inputs.len(), 2 );
                predict_binary_layer( ctx, layer.clone(), &inputs[0], &inputs[1] )?
            }
        };

        if log_enabled!( log::Level::Debug ) {
            let slice = output_buffer.as_slice().expect( "expected an f32 output" );
            let (mean, variance) = calculate_mean_and_variance( slice );
            debug!(
                "Layer {} ({}) output: variance = {}, mean = {}",
                node_index,
                model.get_node( node_index ).type_name(),
                variance,
                mean
            );
        }

        if needs_normalization {
            info!( "Layer {} ({}): normalizing outputs", node_index, model.get_node( node_index ).type_name() );
            output_buffer = match *model.get_node_mut( node_index ) {
                Node::UnaryNode { layer: AnyUnaryLayer::Dense( ref mut layer ), .. } => {
                    let bias_count = layer.size;
                    let target_variance = 0.9;
                    let target_mean = 0.02;

                    normalize_layer_output(
                        layer,
                        node_index,
                        target_variance,
                        target_mean,
                        bias_count,
                        output_buffer,
                        |layer| predict_unary_layer( ctx, layer, &inputs[0] )
                    )?
                },
                Node::UnaryNode { layer: AnyUnaryLayer::Convolution( ref mut layer ), .. } => {
                    let bias_count = layer.filter_count;
                    let target_variance = 1.0;
                    let target_mean = 0.01;

                    normalize_layer_output(
                        layer,
                        node_index,
                        target_variance,
                        target_mean,
                        bias_count,
                        output_buffer,
                        |layer| predict_unary_layer( ctx, layer, &inputs[0] )
                    )?
                },
                _ => unreachable!()
            }
        }

        if needs_prediction {
            Ok( Some( output_buffer ) )
        } else {
            Ok( None )
        }
    })
}
