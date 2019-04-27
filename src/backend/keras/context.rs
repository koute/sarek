use {
    std::{
        env
    },
    lazy_static::lazy_static,
    log::info,
    pyo3::{
        prelude::*,
        types::{
            PyDict
        }
    },
    crate::{
        backend::{
            ContextCreationError,
            keras::{
                async_runner::{
                    AsyncRunner
                },
                py_utils::{
                    py_err
                }
            }
        }
    }
};

lazy_static! {
    static ref RUNNER: AsyncRunner = AsyncRunner::new();
}

#[derive(Clone)]
pub struct Context {
    _dummy: ()
}

impl Context {
    fn new_internal( py: Python ) -> PyResult< Self > {
        env::set_var( "TF_CPP_MIN_LOG_LEVEL", "3" );

        let sys = py.import( "sys" )?;
        let version: String = sys.get( "version" )?.extract()?;
        info!( "Using Python {}", version.replace( "\n", "" ) );

        py.import( "random" )?.getattr( "seed" )?.call( (1234,), None )?;

        let np = py.import( "numpy" )?;
        let np_version: String = np.get( "version" )?.getattr( "version" )?.extract()?;
        info!( "Using numpy {}", np_version );

        np.getattr( "random" )?.getattr( "seed" )?.call( (1234,), None )?;

        let tf = py.import( "tensorflow" )?;
        let tf_version: String = tf.get( "VERSION" )?.extract()?;
        info!( "Using TensorFlow {}", tf_version );

        let logging = tf.getattr( "logging" )?;
        let log_level = logging.getattr( "ERROR" )?;
        logging.getattr( "set_verbosity" )?.call( (log_level,), None )?;

        tf.getattr( "set_random_seed" )?.call( (1234,), None )?;

        if cfg!( debug_assertions ) {
            info!( "Disabling parallelism to ensure determinism" );

            let kwargs = PyDict::new( py );
            kwargs.set_item( "intra_op_parallelism_threads", 1 )?;
            kwargs.set_item( "inter_op_parallelism_threads", 1 )?;
            let session_conf = tf.getattr( "ConfigProto" )?.call( (), Some( kwargs ) )?;
            let default_graph = tf.getattr( "get_default_graph" )?.call( (), None )?;
            let kwargs = PyDict::new( py );
            kwargs.set_item( "graph", default_graph )?;
            kwargs.set_item( "config", session_conf )?;
            let session = tf.getattr( "Session" )?.call( (), Some( kwargs ) )?;
            tf.getattr( "keras" )?.getattr( "backend" )?.getattr( "set_session" )?.call( (session,), None )?;
        }

        let ctx = Context { _dummy: () };
        Ok( ctx )
    }

    pub fn new() -> Result< Self, ContextCreationError > {
        lazy_static! {
            static ref CONTEXT: Result< Context, ContextCreationError > = {
                env::set_var( "PYTHONHASHSEED", "0" );

                Context::gil( |py| {
                    Context::new_internal( py ).map_err( |err| py_err( py, err ) ).map_err( ContextCreationError )
                })
            };
        }

        CONTEXT.clone()
    }

    pub(crate) fn gil< R, F >( callback: F ) -> R where F: FnOnce( Python ) -> R + Send, R: Send + 'static {
        // For some reason Tensorflow breaks if we access Python from
        // different threads (even if we use a mutex!) so we'll just
        // spawn a dedicated Python thread and run everything on it.
        RUNNER.execute_sync( || {
            let gil = Python::acquire_gil();
            let py = gil.python();
            callback( py )
        })
    }
}
