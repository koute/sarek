use {
    crate::{
        backend::{
            context::{
                CloneableError
            }
        },
        core::{
            data_type::{
                Type
            }
        }
    },
    log::{
        error
    },
    std::{
        error::{
            Error
        },
        fmt
    },
    pyo3::{
        prelude::*,
        types::{
            PyDict,
            PyList
        }
    }
};

#[allow(dead_code)]
pub fn py_print( py: Python, obj: &PyObject ) {
    let args = PyDict::new( py );
    args.set_item( "v0", obj.clone_ref( py ) ).unwrap();
    py.run( "print(v0);", None, Some( args ) ).unwrap();
}

pub fn py_to_string< T >( py: Python, obj: T ) -> String where T: ToPyObject {
    let args = PyDict::new( py );
    args.set_item( "v0", obj.to_object( py ) ).unwrap();
    let message = py.eval( r#""{}".format(v0)"#, None, Some( args ) ).unwrap();
    let message: String = message.extract().unwrap();
    message
}

#[derive(Clone)]
struct PyConvertedErr {
    message: String,
    traceback: Option< String >
}

impl Error for PyConvertedErr {}
impl fmt::Display for PyConvertedErr {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        self.message.fmt( fmt )
    }
}

impl fmt::Debug for PyConvertedErr {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        self.message.fmt( fmt )
    }
}

impl PyConvertedErr {
    fn new( py: Python, error: PyErr ) -> Self {
        let traceback = error.ptraceback.as_ref().map( |traceback| {
            let tb_ns = py.import( "traceback" ).unwrap();
            let traceback = tb_ns.getattr( "format_tb" ).unwrap().call( (traceback,), None ).unwrap();
            let traceback = traceback.cast_as::< PyList >().unwrap();
            let mut output = String::new();
            for line in traceback.iter() {
               let line: String = line.extract().unwrap();
               output.push_str( &line );
               output.push( '\n' );
            }
            output
        });
        let message = py_to_string( py, error );

        PyConvertedErr {
            message,
            traceback
        }
    }
}

pub fn py_err( py: Python, error: PyErr ) -> Box< dyn CloneableError > {
    let error = PyConvertedErr::new( py, error );
    Box::new( error )
}

pub trait PyResultExt< T > {
    fn unwrap_py( self, py: Python ) -> T;
}

impl< T > PyResultExt< T > for Result< T, PyErr > {
    fn unwrap_py( self, py: Python ) -> T {
        match self {
            Ok( value ) => value,
            Err( error ) => {
                let error = PyConvertedErr::new( py, error );
                error!( "Python error: {}", error );
                if let Some( ref traceback ) = error.traceback {
                    error!( "Python traceback:\n  {}", traceback.trim() );
                }

                panic!( "`unwrap_py` called on a Python error: {}", error );
            }
        }
    }
}

pub fn py_type_name( ty: Type ) -> &'static str {
    match ty {
        Type::F32 => "float32",
        Type::I32 => "int32",
        Type::I16 => "int16",
        Type::I8 => "int8",
        Type::U32 => "uint32",
        Type::U16 => "uint16",
        Type::U8 => "uint8"
    }
}
