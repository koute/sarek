use {
    std::{
        env,
        error::Error,
        fmt
    },
    lazy_static::lazy_static,
    log::info,
    pyo3::{
        prelude::*
    },
    crate::{
        backend::{
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

#[derive(Display)]
#[display(fmt = "context creation failed: {}", "_0")]
pub struct ContextCreationError( Box< Error + Send > );

impl fmt::Debug for ContextCreationError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt::Debug::fmt( &self.0, fmt )
    }
}

impl Error for ContextCreationError {}

impl Context {
    fn new_internal( py: Python ) -> PyResult< Self > {
        env::set_var( "TF_CPP_MIN_LOG_LEVEL", "3" );

        let sys = py.import( "sys" )?;
        let version: String = sys.get( "version" )?.extract()?;
        info!( "Using Python {}", version.replace( "\n", "" ) );

        let tf = py.import( "tensorflow" )?;
        let tf_version: String = tf.get( "VERSION" )?.extract()?;
        info!( "Using TensorFlow {}", tf_version );

        let np = py.import( "numpy" )?;
        let np_version: String = np.get( "version" )?.getattr( "version" )?.extract()?;
        info!( "Using numpy {}", np_version );

        let ctx = Context { _dummy: () };
        Ok( ctx )
    }

    pub fn new() -> Result< Self, ContextCreationError > {
        Context::gil( |py| {
            Self::new_internal( py ).map_err( |err| py_err( py, err ) ).map_err( ContextCreationError )
        })
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
