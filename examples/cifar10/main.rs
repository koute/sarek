use {
    log::{
        info
    },
    sarek::{
        Context,
        Model,
        Trainer,
        TrainingOpts,
        layers::{
            *
        }
    },
    std::{
        env,
        thread
    }
};

mod data;

fn main() {
    if env::var( "RUST_LOG" ).is_err() {
        env::set_var( "RUST_LOG", "sarek=info,cifar10=info" );
    }

    env_logger::init();

    let training_data = thread::spawn( || data::load_training_data() );
    let test_data = thread::spawn( || data::load_test_data() );

    let ctx = Context::new().unwrap();

    let training_data = training_data.join().unwrap();
    let model = Model::new_sequential( training_data.input_shape(), (
        LayerDense::new( 96 ),
        LayerActivation::new(),
        LayerDense::new( 32 ),
        LayerActivation::new(),
        LayerDense::new( 32 ),
        LayerActivation::new(),
        LayerDense::new( 32 ),
        LayerActivation::new(),
        LayerDense::new( 10 ),
        LayerSoftmax::new(),
        LayerIntoCategory::new()
    ));

    let mut opts = TrainingOpts::new();
    opts.set_batch_size( 128 );

    let test_data = test_data.join().unwrap();

    let mut instance = Trainer::new_with_opts( &ctx, model, training_data, opts ).unwrap();
    for _ in 0..10 {
        instance.train();

        let loss = instance.test( &test_data );
        info!( "Accuracy on the test set: {:.02}% (loss = {})", loss.accuracy().unwrap() * 100.0, loss.get() );
    }
}