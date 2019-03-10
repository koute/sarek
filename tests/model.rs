#![feature(const_slice_len)]
#![allow(unused_parens)]

use {
    sarek::{
        *,
        layers::*,
        optimizers::*
    },
    testutils::{
        assert_f32_eq,
        calculate_mean_and_variance
    }
};

fn init_logger() {
    let _ = env_logger::try_init();
}

fn get_testing_loss_classification( batch_size: usize, inputs: &[f32], expected_outputs: &[u32] ) -> Loss {
    let ctx = Context::new().unwrap();
    let input_shape: Shape = (inputs.len() / batch_size).into();
    let model = Model::new_sequential( input_shape.clone(), (
        LayerIntoCategory::new()
    ));

    let inputs = SliceSource::from( input_shape, inputs );
    let expected_outputs = SliceSource::from( 1.into(), expected_outputs );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = ModelInstance::new( &ctx, model ).unwrap();
    instance.test( &data_set )
}

fn get_testing_loss_regression( element_size: usize, inputs: &[f32], expected_outputs: &[f32] ) -> Loss {
    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( element_size, () );

    let inputs = SliceSource::from( element_size.into(), inputs );
    let expected_outputs = SliceSource::from( element_size.into(), expected_outputs );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut instance = ModelInstance::new( &ctx, model ).unwrap();
    instance.test( &data_set )
}

fn get_training_loss_classification( batch_size: usize, inputs: &[f32], expected_outputs: &[u32] ) -> f32 {
    let ctx = Context::new().unwrap();
    let input_shape: Shape = (inputs.len() / batch_size).into();
    let model = Model::new_sequential( input_shape.clone(), (
        LayerIntoCategory::new()
    ));

    let inputs = SliceSource::from( input_shape, inputs );
    let expected_outputs = SliceSource::from( 1.into(), expected_outputs );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut opts = TrainingOpts::new();
    let mut optimizer = OptimizerSGD::new();
    optimizer.set_learning_rate( 1.0 );
    opts.set_batch_size( batch_size );
    opts.set_optimizer( optimizer );
    opts.disable_input_normalization();

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, opts ).unwrap();
    let loss_1 = instance.train();
    let loss_2 = instance.train();
    assert_eq!( loss_1, loss_2 );

    loss_1
}

fn get_training_loss( batch_size: usize, inputs: &[f32], expected_outputs: &[f32] ) -> f32 {
    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( 1, () );

    let inputs = SliceSource::from( 1.into(), inputs );
    let expected_outputs = SliceSource::from( 1.into(), expected_outputs );
    let data_set = DataSet::new( inputs, expected_outputs );

    let mut opts = TrainingOpts::new();
    let mut optimizer = OptimizerSGD::new();
    optimizer.set_learning_rate( 1.0 );
    opts.set_batch_size( batch_size );
    opts.set_optimizer( optimizer );
    opts.disable_input_normalization();

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, opts ).unwrap();
    let loss_1 = instance.train();
    let loss_2 = instance.train();
    assert_eq!( loss_1, loss_2 );

    loss_1
}

fn loss_for_classification( inputs: &[f32], expected_output: u32 ) -> f32 {
    let sum: f32 = inputs.iter().sum();
    -((inputs[ expected_output as usize ] / sum).ln())
}

#[test]
fn test_training_loss_regression() {
    fn loss_for( input: f32, expected_output: f32 ) -> f32 {
        let err = (input - expected_output) * 2.0;
        err * err / 4.0
    }

    init_logger();

    assert_f32_eq( get_training_loss( 1, &[1.0], &[1.0] ), 0.0 );

    let inputs = &[1.0];
    let expected_outputs = &[0.5];
    let expected_loss = loss_for( inputs[ 0 ], expected_outputs[ 0 ] );
    assert_f32_eq( get_training_loss( 1, inputs, expected_outputs ), expected_loss );

    let inputs = &[1.0, 1.0];
    let expected_outputs = &[0.5, 0.25];
    let expected_loss =
        loss_for( inputs[ 0 ], expected_outputs[ 0 ] ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ] );
    assert_f32_eq( get_training_loss( 1, inputs, expected_outputs ), expected_loss );

    let inputs = &[1.0, 1.0, 1.0];
    let expected_outputs = &[0.5, 0.25, 0.1];
    let expected_loss =
        loss_for( inputs[ 0 ], expected_outputs[ 0 ] ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ] ) +
        loss_for( inputs[ 2 ], expected_outputs[ 2 ] );
    assert_f32_eq( get_training_loss( 1, inputs, expected_outputs ), expected_loss );

    let inputs = &[1.0, 1.0, 1.0, 1.0];
    let expected_outputs = &[0.5, 0.25, 0.1, 0.01];
    let expected_loss = (
        loss_for( inputs[ 0 ], expected_outputs[ 0 ] ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ] ) +
        loss_for( inputs[ 2 ], expected_outputs[ 2 ] ) +
        loss_for( inputs[ 3 ], expected_outputs[ 3 ] )
    );
    assert_f32_eq( get_training_loss( 2, inputs, expected_outputs ), expected_loss );

    let inputs = &[1.0, 1.0, 1.0, 1.0, 2.0, 3.0];
    let expected_outputs = &[0.5, 0.25, 0.1, 0.01, 1.0, -1.0];
    let expected_loss = (
        loss_for( inputs[ 0 ], expected_outputs[ 0 ] ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ] ) +
        loss_for( inputs[ 2 ], expected_outputs[ 2 ] ) +
        loss_for( inputs[ 3 ], expected_outputs[ 3 ] ) +
        loss_for( inputs[ 4 ], expected_outputs[ 4 ] ) +
        loss_for( inputs[ 5 ], expected_outputs[ 5 ] )
    );
    assert_f32_eq( get_training_loss( 2, inputs, expected_outputs ), expected_loss );

    let expected_loss = (
        loss_for( inputs[ 0 ], expected_outputs[ 0 ] ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ] ) +
        loss_for( inputs[ 2 ], expected_outputs[ 2 ] ) +
        loss_for( inputs[ 3 ], expected_outputs[ 3 ] ) +
        loss_for( inputs[ 4 ], expected_outputs[ 4 ] ) +
        loss_for( inputs[ 5 ], expected_outputs[ 5 ] )
    );
    assert_f32_eq( get_training_loss( 3, inputs, expected_outputs ), expected_loss );
}

#[ignore]
#[test]
fn test_training_loss_classification_broken_on_tensorflow_1_13() {
    use loss_for_classification as loss_for;
    init_logger();

    let inputs = &[1.0];
    let expected_outputs = &[0];
    let expected_loss = loss_for( inputs, expected_outputs[ 0 ] );
    assert_f32_eq( get_training_loss_classification( 1, inputs, expected_outputs ), expected_loss );
}

#[test]
fn test_training_loss_classification() {
    use loss_for_classification as loss_for;
    init_logger();

    let inputs = &[0.4, 0.6];
    let expected_outputs = &[0];
    let expected_loss = loss_for( inputs, expected_outputs[ 0 ] );
    assert_f32_eq( get_training_loss_classification( 1, inputs, expected_outputs ), expected_loss );

    let inputs = &[0.4, 0.6];
    let expected_outputs = &[1];
    let expected_loss = loss_for( inputs, expected_outputs[ 0 ] );
    assert_f32_eq( get_training_loss_classification( 1, inputs, expected_outputs ), expected_loss );

    let inputs = &[0.4, 0.6, 0.1, 0.9];
    let expected_outputs = &[0, 0];
    let expected_loss =
        loss_for( &inputs[ 0..2 ], expected_outputs[ 0 ] ) +
        loss_for( &inputs[ 2..4 ], expected_outputs[ 1 ] );
    assert_f32_eq( get_training_loss_classification( 2, inputs, expected_outputs ), expected_loss );
}

#[test]
fn test_testing_loss_classification() {
    use loss_for_classification as loss_for;

    let inputs = &[0.4, 0.6, 0.1, 0.9];
    let expected_outputs = &[0, 0];
    let expected_loss =
        loss_for( &inputs[ 0..2 ], expected_outputs[ 0 ] ) +
        loss_for( &inputs[ 2..4 ], expected_outputs[ 1 ] );

    let loss = get_testing_loss_classification( 2, inputs, expected_outputs );
    assert_f32_eq( loss.get(), expected_loss );
    assert_f32_eq( loss.accuracy().unwrap(), 0.0 );

    let inputs = &[0.4, 0.6, 0.1, 0.9];
    let expected_outputs = &[0, 1];
    let expected_loss =
        loss_for( &inputs[ 0..2 ], expected_outputs[ 0 ] ) +
        loss_for( &inputs[ 2..4 ], expected_outputs[ 1 ] );

    let loss = get_testing_loss_classification( 2, inputs, expected_outputs );
    assert_f32_eq( loss.get(), expected_loss );
    assert_f32_eq( loss.accuracy().unwrap(), 0.5 );
}

#[test]
fn test_testing_loss_regression() {
    fn loss_for( input: f32, expected_output: f32, outputs_length: usize ) -> f32 {
        let err = (input - expected_output) * 2.0 / outputs_length as f32;
        err * err
    }

    let inputs = &[1.0, 1.0, 1.0, 1.0, 2.0, 3.0];
    let expected_outputs = &[0.5, 0.25, 0.1, 0.01, 1.0, -1.0];
    let expected_loss = (
        loss_for( inputs[ 0 ], expected_outputs[ 0 ], 2 ) +
        loss_for( inputs[ 1 ], expected_outputs[ 1 ], 2 )
    ) * 2.0 / 4.0 + (
        loss_for( inputs[ 2 ], expected_outputs[ 2 ], 2 ) +
        loss_for( inputs[ 3 ], expected_outputs[ 3 ], 2 )
    ) * 2.0 / 4.0 + (
        loss_for( inputs[ 4 ], expected_outputs[ 4 ], 2 ) +
        loss_for( inputs[ 5 ], expected_outputs[ 5 ], 2 )
    ) * 2.0 / 4.0;

    let loss = get_testing_loss_regression( 2, inputs, expected_outputs );
    assert_f32_eq( loss.get(), expected_loss );
    assert!( loss.accuracy().is_none() );
}

fn test_input_normalization( inputs: &[f32], expected_mean: f32, expected_variance: f32 ) {
    let expected_outputs: Vec< _ > = inputs.iter().map( |_| 0.0 ).collect();

    let ctx = Context::new().unwrap();
    let model = Model::new_sequential( (2, 2), () );

    let inputs = SliceSource::from( (2, 2).into(), inputs );
    let expected_outputs = SliceSource::from( (2, 2).into(), expected_outputs );
    let data_set = DataSet::new( &inputs, expected_outputs );

    let mut opts = TrainingOpts::new();
    let mut optimizer = OptimizerSGD::new();
    optimizer.set_learning_rate( 1.0 );
    opts.set_optimizer( optimizer );

    let mut instance = Trainer::new_with_opts( &ctx, model, data_set, opts ).unwrap();
    let outputs = instance.predict( &inputs );
    assert_eq!( outputs.len(), 1 );
    let outputs = outputs.into_iter().next().unwrap();
    let outputs = outputs.to_slice::< f32 >().unwrap();
    assert!( outputs.iter().cloned().all( |x| !x.is_nan() ), "The output contains NaNs: {:?}", outputs );

    let (mean, variance) = calculate_mean_and_variance( outputs );
    assert_f32_eq( mean, expected_mean );
    assert_f32_eq( variance, expected_variance );
}

#[test]
fn test_input_normalization_only_one_element() {
    init_logger();

    let inputs = &[
        1.0, 2.0, 3.0, 4.0
    ];

    test_input_normalization( inputs, 2.5, 1.25 );
}


#[test]
fn test_input_normalization_multiple_elements() {
    init_logger();

    let inputs = &[
        1.0, 2.0, 3.0, 4.0,
        10.0, 5.0, 2.0, 0.5
    ];

    test_input_normalization( inputs, 0.0, 1.0 );
}
