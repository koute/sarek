use {
    sarek::{
        Context,
        Model,
        Trainer,
        TrainingOpts,
        ToArrayRef,
        layers::{
            *
        }
    },
    log::{
        info
    },
    std::{
        env,
        thread
    }
};

mod data;

fn main() {
    if env::var( "RUST_LOG" ).is_err() {
        env::set_var( "RUST_LOG", "sarek=info,mnist=info" );
    }

    env_logger::init();

    let training_data = thread::spawn( || data::load_training_data_set() );
    let test_data = thread::spawn( || data::load_test_data_set() );

    let ctx = Context::new().unwrap();

    let training_data = training_data.join().unwrap();
    let model = Model::new_sequential( training_data.input_shape(), (
        LayerDense::new( 512 ),
        LayerActivation::new(),
        LayerDropout::new( 0.2 ),
        LayerDense::new( 10 ),
        LayerSoftmax::new(),
        LayerIntoCategory::new()
    ));

    let mut opts = TrainingOpts::new();
    opts.set_batch_size( 128 );

    let test_data = test_data.join().unwrap();

    let mut instance = Trainer::new_with_opts( &ctx, model, training_data, opts ).unwrap();
    for _ in 0..4 {
        instance.train();
    }

    let predictions = instance.predict( test_data.input_data() );
    let predictions = predictions.to_slice::< u32 >().unwrap();
    let expected = test_data.expected_output_data().to_slice::< u32 >().unwrap();
    assert_eq!( predictions.len(), expected.len() );

    let total_count = predictions.len();
    let mut correct_count = 0;
    for (predicted, expected) in predictions.iter().zip( expected.iter() ) {
        if predicted == expected {
            correct_count += 1;
        }
    }

    info!( "Accuracy on the test set: {:.02}%", (correct_count as f32 / total_count as f32) * 100.0 );
}