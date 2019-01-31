#![feature(const_slice_len)]
#![allow(unused_parens)]

use {
    std::{
        fmt,
        iter
    },
    sarek::{
        *,
        layers::*,
        optimizers::*
    },
    testutils::{
        assert_f32_slice_eq,
        assert_f32_eq
    }
};

fn init_logger() {
    let _ = env_logger::try_init();
}

fn test_prediction< I >( layers: I, inputs: &[f32], expected_outputs: &[f32] )
    where I: IntoLayerIter
{
    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( inputs.len(), layers );
    let mut instance = ModelInstance::new( &ctx, model ).unwrap();
    let inputs = SliceSource::from( inputs.len().into(), inputs );
    let output = instance.predict( &inputs );
    assert_f32_slice_eq(
        output.to_slice::< f32 >().unwrap(),
        expected_outputs
    );
}

fn test_prediction_exact< I, T >( layers: I, inputs: &[f32], input_count: usize, expected_outputs: &[T] )
    where I: IntoLayerIter,
          T: DataType + PartialEq + fmt::Debug
{
    assert_eq!( inputs.len() % input_count, 0 );

    let input_shape: Shape = (inputs.len() / input_count).into();
    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( input_shape.clone(), layers );
    let mut instance = ModelInstance::new( &ctx, model ).unwrap();
    let inputs = SliceSource::from( input_shape, inputs );
    let output = instance.predict( &inputs );
    assert_eq!(
        output.to_slice::< T >().unwrap(),
        expected_outputs
    );
}

fn training_opts( learning_rate: f32 ) -> TrainingOpts {
    let mut opts = TrainingOpts::new();
    let mut optimizer = OptimizerSGD::new();
    optimizer.set_learning_rate( learning_rate );
    opts.set_batch_size( 1 );
    opts.set_optimizer( optimizer );
    opts.disable_weight_pretraining();
    opts
}

fn test_backpropagation< L >( layer: L, inputs: &[f32], output_errors: &[f32], expected_input_errors: &[f32] )
    where L: Into< Layer >
{
    let kind: Kind = Kind::OutputErrors( output_errors );
    test_backpropagation_generic( layer, inputs, kind, expected_input_errors )
}

enum Kind< 'a, T = f32 > where T: DataType {
    ExpectedOutputs( &'a [T] ),
    OutputErrors( &'a [f32] )
}

fn test_backpropagation_generic< L, T >( layer: L, inputs: &[f32], kind: Kind< T >, expected_input_errors: &[f32] )
    where L: Into< Layer >, T: DataType
{
    // Here we extract the given layer's backpropagated input errors
    // and compare them to the expected ones.
    //
    // We don't do it directly; instead we put the layer
    // into a network with a dense layer at the front,
    // and then we just train the network once. By carefully
    // initializing the network and by using some rudimentaly
    // algebra we can easily extract the actual backpropagated
    // input errors from the dense layer's weight matrix.
    //
    // This saves us the trouble of having to define APIs
    // to actually be able to extract the input errors directly.
    assert_eq!( inputs.len(), expected_input_errors.len() );

    const LEARNING_RATE: f32 = 1.0;

    let network_input: &[f32] = &[1.0];
    let mut original_weights = Vec::new();
    // We define the biases to be zero. The input errors
    // will be written there.
    original_weights.extend( iter::repeat( 0.0 ).take( inputs.len() ) );
    // We set the rest of the weights to the inputs
    // of the layer under the test, and since the dense
    // layer takes only one input equal to 1.0 its
    // outputs will be equal to what we set here.
    original_weights.extend( inputs );

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( network_input.len(), (
        LayerDense::new( inputs.len() )
            .set_name( "dense_layer" ),
        layer.into()
    ));

    let network_input = SliceSource::from( network_input.len().into(), network_input );
    let new_weights = match kind {
        Kind::OutputErrors( output_errors ) => {
            // Now we run prediction once and grab actual outputs
            // of our layer under test.
            let output = {
                let mut instance = ModelInstance::new( &ctx, model.clone() ).unwrap();
                instance.set_weights( "dense_layer", &original_weights ).unwrap();
                instance.predict( &network_input )
            };
            let output = output.to_slice::< f32 >().unwrap();

            // Now we calculate what our expected outputs should be
            // so that we'll get the desired output errors for our
            // layer under test to backpropagate.
            let expected_outputs: Vec< _ > = output
                .iter().cloned()
                .zip( output_errors.iter().cloned() )
                .map( |(output, target_output_error)| {
                    // The output errors for the last layer are calculated like this:
                    //     output_error = (output - expected_output) * 2 / output_count
                    //
                    // So we can rearrange this to manipulate the expected output to get a desired output error:
                    //     expected_output = output - output_error * output_count / 2
                    output - target_output_error * output_errors.len() as f32 / 2.0
                }).collect();

            // We can finally train the network.
            let expected_outputs = SliceSource::from( expected_outputs.len().into(), expected_outputs );
            let data_set = DataSet::new( network_input, expected_outputs );

            let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
            instance.set_weights( "dense_layer", &original_weights ).unwrap();
            instance.train();

            instance.get_weights( "dense_layer" )
        },
        Kind::ExpectedOutputs( expected_outputs ) => {
            let expected_outputs = SliceSource::from( expected_outputs.len().into(), expected_outputs );
            let data_set = DataSet::new( network_input, expected_outputs );

            let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
            instance.set_weights( "dense_layer", &original_weights ).unwrap();
            instance.train();

            instance.get_weights( "dense_layer" )
        }
    };

    // And the only thing left is to extract the backpropagated input errors
    // from the dense layer's newly updated weights.
    let new_weights = new_weights.to_slice::< f32 >().unwrap();
    let input_errors: Vec< _ > = (0..inputs.len()).into_iter().map( |nth| {
        // The new weights are calculated like this:
        //     new_weight = old_weight - output_error
        //
        // So we can extract the backpropagated input errors:
        //     output_error = old_weight - new_weight
        //
        // And since we set the bias weights to zero it's simply:
        //     output_error = -new_weight

        -new_weights[ nth ]
    }).collect();

    assert_f32_slice_eq(
        &input_errors,
        expected_input_errors
    );
}

#[test]
fn test_layer_dense_prediction() {
    init_logger();

    const INPUTS: &'static [f32] = &[ -0.5, 1.5, 0.2 ];
    const WEIGHTS: &'static [f32] = &[ 2.0, -3.0, 0.4, 0.8, 0.6, 0.2, 0.1, 0.3 ];
    const OUTPUTS: &'static [f32] = &[
        WEIGHTS[0] +
        WEIGHTS[2] * INPUTS[0] +
        WEIGHTS[4] * INPUTS[1] +
        WEIGHTS[6] * INPUTS[2],

        WEIGHTS[1] +
        WEIGHTS[3] * INPUTS[0] +
        WEIGHTS[5] * INPUTS[1] +
        WEIGHTS[7] * INPUTS[2]
    ];

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( 3, (
        LayerDense::new( 2 )
            .set_name( "layer" )
    ));

    let mut instance = ModelInstance::new( &ctx, model ).unwrap();
    instance.set_weights( "layer", &WEIGHTS ).unwrap();
    assert_f32_slice_eq(
        instance.get_weights( "layer" ).to_slice::< f32 >().unwrap(),
        WEIGHTS
    );

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let output = instance.predict( &inputs );
    assert_f32_slice_eq(
        output.to_slice::< f32 >().unwrap(),
        OUTPUTS
    );
}

#[test]
fn test_layer_dense_simple_training_one_input_one_output() {
    init_logger();

    const LEARNING_RATE: f32 = 1.11;
    const INPUTS: &'static [f32] = &[
        1.22
    ];
    const WEIGHTS: &'static [f32] = &[
        1.33, 1.44
    ];
    const OUTPUTS: &'static [f32] = &[
        WEIGHTS[0] + WEIGHTS[1] * INPUTS[0]
    ];
    const EXPECTED_OUTPUTS: &'static [f32] = &[
        3.0
    ];
    const OUTPUT_ERRORS: &'static [f32] = &[
        (OUTPUTS[0] - EXPECTED_OUTPUTS[0]) * 2.0 / OUTPUTS.len() as f32
    ];
    const UPDATED_WEIGHTS: &'static [f32] = &[
        WEIGHTS[0] - LEARNING_RATE * OUTPUT_ERRORS[0] * 1.0,
        WEIGHTS[1] - LEARNING_RATE * OUTPUT_ERRORS[0] * INPUTS[0]
    ];
    const LOSS: f32 = (
        OUTPUT_ERRORS[0] * OUTPUT_ERRORS[0]
    ) * OUTPUT_ERRORS.len() as f32 / 4.0;

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( INPUTS.len(), (
        LayerDense::new( OUTPUTS.len() )
            .set_name( "layer" )
    ));

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let expected_outputs = SliceSource::from( EXPECTED_OUTPUTS.len().into(), EXPECTED_OUTPUTS );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
    instance.set_weights( "layer", &WEIGHTS ).unwrap();

    let loss = instance.train();
    assert_f32_eq( loss, LOSS );

    let weights = instance.get_weights( "layer" );
    assert_f32_slice_eq(
        weights.to_slice::< f32 >().unwrap(),
        UPDATED_WEIGHTS
    );
}

#[test]
fn test_layer_dense_simple_training_one_input_three_outputs() {
    init_logger();

    const LEARNING_RATE: f32 = 1.11;
    const INPUTS: &'static [f32] = &[
        1.22
    ];
    const WEIGHTS: &'static [f32] = &[
        1.33, 1.44,
        1.55, 1.66,
        1.77, 1.88
    ];
    const OUTPUTS: &'static [f32] = &[
        WEIGHTS[0] + WEIGHTS[3] * INPUTS[0],
        WEIGHTS[1] + WEIGHTS[4] * INPUTS[0],
        WEIGHTS[2] + WEIGHTS[5] * INPUTS[0],
    ];
    const EXPECTED_OUTPUTS: &'static [f32] = &[
        3.0, 6.0, 9.0
    ];
    const OUTPUT_ERRORS: &'static [f32] = &[
        (OUTPUTS[0] - EXPECTED_OUTPUTS[0]) * 2.0 / OUTPUTS.len() as f32,
        (OUTPUTS[1] - EXPECTED_OUTPUTS[1]) * 2.0 / OUTPUTS.len() as f32,
        (OUTPUTS[2] - EXPECTED_OUTPUTS[2]) * 2.0 / OUTPUTS.len() as f32,
    ];
    const UPDATED_WEIGHTS: &'static [f32] = &[
        WEIGHTS[0] - LEARNING_RATE * OUTPUT_ERRORS[0] * 1.0,
        WEIGHTS[1] - LEARNING_RATE * OUTPUT_ERRORS[1] * 1.0,
        WEIGHTS[2] - LEARNING_RATE * OUTPUT_ERRORS[2] * 1.0,
        WEIGHTS[3] - LEARNING_RATE * OUTPUT_ERRORS[0] * INPUTS[0],
        WEIGHTS[4] - LEARNING_RATE * OUTPUT_ERRORS[1] * INPUTS[0],
        WEIGHTS[5] - LEARNING_RATE * OUTPUT_ERRORS[2] * INPUTS[0]
    ];
    const LOSS: f32 = (
        OUTPUT_ERRORS[0] * OUTPUT_ERRORS[0] +
        OUTPUT_ERRORS[1] * OUTPUT_ERRORS[1] +
        OUTPUT_ERRORS[2] * OUTPUT_ERRORS[2]
    ) * OUTPUT_ERRORS.len() as f32 / 4.0;

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( 1, (
        LayerDense::new( 3 )
            .set_name( "layer" )
    ));

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let expected_outputs = SliceSource::from( EXPECTED_OUTPUTS.len().into(), EXPECTED_OUTPUTS );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
    instance.set_weights( "layer", &WEIGHTS ).unwrap();

    let loss = instance.train();
    assert_f32_eq( loss, LOSS );

    let weights = instance.get_weights( "layer" );
    assert_f32_slice_eq(
        weights.to_slice::< f32 >().unwrap(),
        UPDATED_WEIGHTS
    );
}

#[test]
fn test_layer_dense_simple_training_three_inputs_two_outputs() {
    init_logger();

    const LEARNING_RATE: f32 = 1.11;
    const INPUTS: &'static [f32] = &[ -0.5, 1.5, 0.2 ];
    const WEIGHTS: &'static [f32] = &[ 2.0, -3.0, 0.4, 0.8, 0.6, 0.2, 0.1, 0.3 ];
    const OUTPUTS: &'static [f32] = &[
        WEIGHTS[0] +
        WEIGHTS[2] * INPUTS[0] +
        WEIGHTS[4] * INPUTS[1] +
        WEIGHTS[6] * INPUTS[2],

        WEIGHTS[1] +
        WEIGHTS[3] * INPUTS[0] +
        WEIGHTS[5] * INPUTS[1] +
        WEIGHTS[7] * INPUTS[2]
    ];
    const EXPECTED_OUTPUTS: &'static [f32] = &[ 0.5, 0.25 ];
    const OUTPUT_ERRORS: &'static [f32] = &[
        (OUTPUTS[0] - EXPECTED_OUTPUTS[0]) * 2.0 / OUTPUTS.len() as f32,
        (OUTPUTS[1] - EXPECTED_OUTPUTS[1]) * 2.0 / OUTPUTS.len() as f32
    ];
    const UPDATED_WEIGHTS: &'static [f32] = &[
        WEIGHTS[0] - LEARNING_RATE * OUTPUT_ERRORS[0] * 1.0,
        WEIGHTS[1] - LEARNING_RATE * OUTPUT_ERRORS[1] * 1.0,
        WEIGHTS[2] - LEARNING_RATE * OUTPUT_ERRORS[0] * INPUTS[0],
        WEIGHTS[3] - LEARNING_RATE * OUTPUT_ERRORS[1] * INPUTS[0],
        WEIGHTS[4] - LEARNING_RATE * OUTPUT_ERRORS[0] * INPUTS[1],
        WEIGHTS[5] - LEARNING_RATE * OUTPUT_ERRORS[1] * INPUTS[1],
        WEIGHTS[6] - LEARNING_RATE * OUTPUT_ERRORS[0] * INPUTS[2],
        WEIGHTS[7] - LEARNING_RATE * OUTPUT_ERRORS[1] * INPUTS[2],
    ];
    const LOSS: f32 = (
        OUTPUT_ERRORS[0] * OUTPUT_ERRORS[0] +
        OUTPUT_ERRORS[1] * OUTPUT_ERRORS[1]
    ) * OUTPUT_ERRORS.len() as f32 / 4.0;

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( 3, (
        LayerDense::new( 2 )
            .set_name( "layer" )
    ));

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let expected_outputs = SliceSource::from( EXPECTED_OUTPUTS.len().into(), EXPECTED_OUTPUTS );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
    instance.set_weights( "layer", &WEIGHTS ).unwrap();

    let loss = instance.train();
    assert_f32_eq( loss, LOSS );

    let weights = instance.get_weights( "layer" );
    assert_f32_slice_eq(
        weights.to_slice::< f32 >().unwrap(),
        UPDATED_WEIGHTS
    );
}

#[test]
fn test_layer_dense_simple_training_backpropagation() {
    init_logger();

    const LEARNING_RATE: f32 = 0.1;
    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    const WEIGHTS_1: &'static [f32] = &[ 2.0, -3.0, 0.4, 0.8, 0.6, 0.2 ];
    const WEIGHTS_2: &'static [f32] = &[ 0.11, 0.22, 0.33, 0.44, 0.55, 0.66 ];
    const OUTPUTS_1: &'static [f32] = &[
        WEIGHTS_1[0] +
        WEIGHTS_1[2] * INPUTS[0] +
        WEIGHTS_1[4] * INPUTS[1],

        WEIGHTS_1[1] +
        WEIGHTS_1[3] * INPUTS[0] +
        WEIGHTS_1[5] * INPUTS[1]
    ];
    const OUTPUTS_2: &'static [f32] = &[
        WEIGHTS_2[0] +
        WEIGHTS_2[2] * OUTPUTS_1[0] +
        WEIGHTS_2[4] * OUTPUTS_1[1],

        WEIGHTS_2[1] +
        WEIGHTS_2[3] * OUTPUTS_1[0] +
        WEIGHTS_2[5] * OUTPUTS_1[1]
    ];
    const EXPECTED_OUTPUTS: &'static [f32] = &[ 0.5, 0.25 ];
    const OUTPUT_ERRORS_2: &'static [f32] = &[
        (OUTPUTS_2[0] - EXPECTED_OUTPUTS[0]) * 2.0 / OUTPUTS_2.len() as f32,
        (OUTPUTS_2[1] - EXPECTED_OUTPUTS[1]) * 2.0 / OUTPUTS_2.len() as f32
    ];
    const UPDATED_WEIGHTS_2: &'static [f32] = &[
        WEIGHTS_2[0] - LEARNING_RATE * OUTPUT_ERRORS_2[0] * 1.0,
        WEIGHTS_2[1] - LEARNING_RATE * OUTPUT_ERRORS_2[1] * 1.0,
        WEIGHTS_2[2] - LEARNING_RATE * OUTPUT_ERRORS_2[0] * OUTPUTS_1[0],
        WEIGHTS_2[3] - LEARNING_RATE * OUTPUT_ERRORS_2[1] * OUTPUTS_1[0],
        WEIGHTS_2[4] - LEARNING_RATE * OUTPUT_ERRORS_2[0] * OUTPUTS_1[1],
        WEIGHTS_2[5] - LEARNING_RATE * OUTPUT_ERRORS_2[1] * OUTPUTS_1[1]
    ];
    const OUTPUT_ERRORS_1: &'static [f32] = &[
        OUTPUT_ERRORS_2[0] * WEIGHTS_2[2] +
        OUTPUT_ERRORS_2[1] * WEIGHTS_2[3],

        OUTPUT_ERRORS_2[0] * WEIGHTS_2[4] +
        OUTPUT_ERRORS_2[1] * WEIGHTS_2[5]
    ];
    const UPDATED_WEIGHTS_1: &'static [f32] = &[
        WEIGHTS_1[0] - LEARNING_RATE * OUTPUT_ERRORS_1[0] * 1.0,
        WEIGHTS_1[1] - LEARNING_RATE * OUTPUT_ERRORS_1[1] * 1.0,
        WEIGHTS_1[2] - LEARNING_RATE * OUTPUT_ERRORS_1[0] * INPUTS[0],
        WEIGHTS_1[3] - LEARNING_RATE * OUTPUT_ERRORS_1[1] * INPUTS[0],
        WEIGHTS_1[4] - LEARNING_RATE * OUTPUT_ERRORS_1[0] * INPUTS[1],
        WEIGHTS_1[5] - LEARNING_RATE * OUTPUT_ERRORS_1[1] * INPUTS[1]
    ];
    const LOSS: f32 = (
        OUTPUT_ERRORS_2[0] * OUTPUT_ERRORS_2[0] +
        OUTPUT_ERRORS_2[1] * OUTPUT_ERRORS_2[1]
    ) * OUTPUT_ERRORS_2.len() as f32 / 4.0;

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( 2, (
        LayerDense::new( 2 )
            .set_name( "layer_1" ),
        LayerDense::new( 2 )
            .set_name( "layer_2" )
    ));

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let expected_outputs = SliceSource::from( EXPECTED_OUTPUTS.len().into(), EXPECTED_OUTPUTS );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( LEARNING_RATE ) ).unwrap();
    instance.set_weights( "layer_1", &WEIGHTS_1 ).unwrap();
    instance.set_weights( "layer_2", &WEIGHTS_2 ).unwrap();

    let inputs = SliceSource::from( INPUTS.len().into(), INPUTS );
    let output = instance.predict( &inputs );
    assert_f32_slice_eq(
        output.to_slice::< f32 >().unwrap(),
        OUTPUTS_2
    );

    let loss = instance.train();
    assert_f32_eq( loss, LOSS );

    let weights_2 = instance.get_weights( "layer_2" );
    assert_f32_slice_eq(
        weights_2.to_slice::< f32 >().unwrap(),
        UPDATED_WEIGHTS_2
    );

    let weights_1 = instance.get_weights( "layer_1" );
    assert_f32_slice_eq(
        weights_1.to_slice::< f32 >().unwrap(),
        UPDATED_WEIGHTS_1
    );
}

#[test]
fn test_layer_activation_relu_prediction() {
    init_logger();

    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    const OUTPUTS: &'static [f32] = &[ 0.0, 1.5 ];

    test_prediction(
        LayerActivation::new().set_activation( Activation::ReLU ),
        INPUTS,
        OUTPUTS
    );
}

#[test]
fn test_layer_activation_relu_backpropagation() {
    init_logger();

    let inputs = &[ -0.5, 1.5 ];
    let output_errors = &[ 1.11, 2.22 ];
    let expected_input_errors = &[ 0.0, output_errors[1] ];

    test_backpropagation(
        LayerActivation::new().set_activation( Activation::ReLU ),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_activation_leaky_relu_prediction() {
    init_logger();

    fn max( a: f32, b: f32 ) -> f32 {
        if a > b {
            a
        } else {
            b
        }
    }

    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    let expected_outputs: & [f32] = &[
        max( 0.01 * INPUTS[0], INPUTS[0] ),
        max( 0.01 * INPUTS[1], INPUTS[1] )
    ];

    test_prediction(
        LayerActivation::new().set_activation( Activation::LeakyReLU ),
        INPUTS,
        expected_outputs
    );
}

#[test]
fn test_layer_activation_leaky_relu_backpropagation() {
    init_logger();

    let inputs = &[ -0.5, 1.5 ];
    let output_errors = &[ 1.11, 2.22 ];
    let expected_input_errors = &[
        0.01 * output_errors[0],
        output_errors[1]
    ];
    test_backpropagation(
        LayerActivation::new().set_activation( Activation::LeakyReLU ),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_activation_elu_prediction() {
    init_logger();

    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    let expected_outputs: & [f32] = &[
        (-0.5_f32).exp() - 1.0,
        1.5
    ];

    test_prediction(
        LayerActivation::new().set_activation( Activation::ELU ),
        INPUTS,
        expected_outputs
    );
}

#[test]
fn test_layer_activation_elu_backpropagation() {
    init_logger();

    let inputs: &[f32] = &[ -0.5, 1.5 ];
    let output_errors = &[ 1.11, 2.22 ];
    let expected_input_errors = &[
        ((inputs[0].exp() - 1.0) + 1.0) * output_errors[0],
        output_errors[1]
    ];
    test_backpropagation(
        LayerActivation::new().set_activation( Activation::ELU ),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_activation_tanh_prediction() {
    init_logger();

    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    let expected_outputs: &[f32] = &[
        INPUTS[0].tanh(),
        INPUTS[1].tanh()
    ];

    test_prediction(
        LayerActivation::new().set_activation( Activation::TanH ),
        INPUTS,
        expected_outputs
    );
}

#[test]
fn test_layer_activation_tanh_backpropagation() {
    init_logger();

    let inputs: &[f32] = &[ -0.5, 1.5 ];
    let output_errors = &[ 1.11, 2.22 ];
    let expected_input_errors = &[
        (1.0 - inputs[0].tanh() * inputs[0].tanh()) * output_errors[0],
        (1.0 - inputs[1].tanh() * inputs[1].tanh()) * output_errors[1]
    ];
    test_backpropagation(
        LayerActivation::new().set_activation( Activation::TanH ),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_activation_logistic_prediction() {
    init_logger();

    const INPUTS: &'static [f32] = &[ -0.5, 1.5 ];
    let expected_outputs: &[f32] = &[
        1.0 / (1.0 + (-INPUTS[0]).exp()),
        1.0 / (1.0 + (-INPUTS[1]).exp()),
    ];

    test_prediction(
        LayerActivation::new().set_activation( Activation::Logistic ),
        INPUTS,
        expected_outputs
    );
}

#[test]
fn test_layer_activation_logistic_backpropagation() {
    init_logger();

    let inputs: &[f32] = &[ -0.5, 1.5 ];
    let output_errors = &[ 1.11, 2.22 ];
    let o_1 = 1.0 / (1.0 + (-inputs[0]).exp());
    let o_2 = 1.0 / (1.0 + (-inputs[1]).exp());
    let expected_input_errors = &[
        (o_1 * (1.0 - o_1)) * output_errors[0],
        (o_2 * (1.0 - o_2)) * output_errors[1]
    ];
    test_backpropagation(
        LayerActivation::new().set_activation( Activation::Logistic ),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_softmax_prediction() {
    init_logger();

    let inputs: & [f32] = &[ -0.5, 1.5, 0.33 ];
    let maximum: f32 = 1.5;
    let tmp: &[f32] = &[
        (inputs[0] - maximum).exp(),
        (inputs[1] - maximum).exp(),
        (inputs[2] - maximum).exp()
    ];

    let sum = tmp[0] + tmp[1] + tmp[2];
    let expected_outputs: &[f32] = &[
        tmp[0] / sum,
        tmp[1] / sum,
        tmp[2] / sum
    ];

    test_prediction(
        LayerSoftmax::new(),
        inputs,
        expected_outputs
    );
}

#[test]
fn test_layer_softmax_backpropagation() {
    init_logger();

    let inputs: &[f32] = &[ -0.5, 1.5, 0.33 ];
    let output_errors = &[ 1.11, 2.22, 3.33 ];
    let maximum: f32 = 1.5;
    let tmp: &[f32] = &[
        (inputs[0] - maximum).exp(),
        (inputs[1] - maximum).exp(),
        (inputs[2] - maximum).exp()
    ];

    let sum = tmp[0] + tmp[1] + tmp[2];
    let outputs: &[f32] = &[
        tmp[0] / sum,
        tmp[1] / sum,
        tmp[2] / sum
    ];

    let expected_input_errors = &[
          output_errors[0] * outputs[0] * (1.0 - outputs[0])
        - output_errors[1] * outputs[0] * outputs[1]
        - output_errors[2] * outputs[0] * outputs[2],

        - output_errors[0] * outputs[1] * outputs[0]
        + output_errors[1] * outputs[1] * (1.0 - outputs[1])
        - output_errors[2] * outputs[1] * outputs[2],

        - output_errors[0] * outputs[2] * outputs[0]
        - output_errors[1] * outputs[2] * outputs[1]
        + output_errors[2] * outputs[2] * (1.0 - outputs[2])
    ];

    test_backpropagation(
        LayerSoftmax::new(),
        inputs,
        output_errors,
        expected_input_errors,
    );
}

#[test]
fn test_layer_into_category_prediction_single_input() {
    init_logger();

    let inputs: &[f32] = &[ 0.3, 0.5, 0.2 ];
    let input_count = 1;
    let expected_outputs: &[u32] = &[ 1 ];

    test_prediction_exact(
        LayerIntoCategory::new(),
        inputs,
        input_count,
        expected_outputs
    );
}

#[test]
fn test_layer_into_category_prediction_multiple_inputs() {
    init_logger();

    let inputs: &[f32] = &[
            0.1, 0.9,
            0.8, 0.2
    ];
    let input_count = 2;
    let expected_outputs: &[u32] = &[ 1, 0 ];

    test_prediction_exact(
        LayerIntoCategory::new(),
        inputs,
        input_count,
        expected_outputs
    );
}

fn test_layer_into_category_loss_for( inputs: &[f32], expected_output: u32 ) {
    let sum: f32 = inputs.iter().cloned().sum();
    let expected_loss: f32 =
        -((inputs[ expected_output as usize ] / sum).ln());
    let expected_loss = expected_loss;

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( inputs.len(), (
        LayerIntoCategory::new()
    ));

    let expected_outputs = &[expected_output];
    let inputs = SliceSource::from( inputs.len().into(), inputs );
    let expected_outputs = SliceSource::from( expected_outputs.len().into(), expected_outputs );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, training_opts( 1.0 ) ).unwrap();

    let loss = instance.train();
    assert_f32_eq( loss, expected_loss );
}

#[test]
fn test_layer_into_category_loss_wrong_output_not_normalized() {
    init_logger();
    test_layer_into_category_loss_for(
        &[ 0.2, 0.5, 0.2 ],
        2
    )
}

#[test]
fn test_layer_into_category_loss_wrong_output_normalized() {
    init_logger();
    test_layer_into_category_loss_for(
        &[ 0.3, 0.5, 0.2 ],
        2
    )
}

#[test]
fn test_layer_into_category_loss_correct_output_not_normalized() {
    init_logger();
    test_layer_into_category_loss_for(
        &[ 0.2, 0.5, 0.2 ],
        1
    )
}

#[test]
fn test_layer_into_category_loss_correct_output_normalized() {
    init_logger();
    test_layer_into_category_loss_for(
        &[ 0.3, 0.5, 0.2 ],
        1
    )
}

fn test_layer_into_category_backpropagation( inputs: &[f32], expected_output: u32 ) {
    let sum: f32 = inputs.iter().cloned().sum();
    let mut expected_input_errors: Vec< _ > =
        inputs.iter().cloned().map( |_| 1.0 / sum ).collect();
    expected_input_errors[ expected_output as usize ] = (1.0 - (sum / inputs[ expected_output as usize ])) / sum;

    test_backpropagation_generic(
        LayerIntoCategory::new(),
        inputs,
        Kind::ExpectedOutputs( &[ expected_output ] ),
        &expected_input_errors
    );
}

#[test]
fn test_layer_into_category_backpropagation_wrong_output_not_normalized() {
    init_logger();
    test_layer_into_category_backpropagation(
        &[ 0.2, 0.5, 0.2 ],
        2
    )
}

#[test]
fn test_layer_into_category_backpropagation_wrong_output_normalized() {
    init_logger();
    test_layer_into_category_backpropagation(
        &[ 0.3, 0.5, 0.2 ],
        2
    )
}

#[test]
fn test_layer_into_category_backpropagation_correct_output_not_normalized() {
    init_logger();
    test_layer_into_category_backpropagation(
        &[ 0.2, 0.5, 0.2 ],
        1
    )
}

#[test]
fn test_layer_into_category_backpropagation_correct_output_normalized() {
    init_logger();
    test_layer_into_category_backpropagation(
        &[ 0.3, 0.5, 0.2 ],
        1
    )
}
