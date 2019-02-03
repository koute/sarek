use {
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
            PyObjectRef
        }
    }
};

#[allow(dead_code)]
pub fn py_print( py: Python, obj: &PyObject ) {
    let args = PyDict::new( py );
    args.set_item( "v0", obj.clone_ref( py ) ).unwrap();
    py.run( "print(v0);", None, Some( args ) ).unwrap();
}

#[allow(dead_code)]
pub fn py_print_ref< T >( py: Python, obj: &T ) where T: AsRef< PyObjectRef > {
    let obj = obj.as_ref();
    let args = PyDict::new( py );
    args.set_item( "v0", obj ).unwrap();
    py.run( "print(v0);", None, Some( args ) ).unwrap();
}

pub fn py_to_string< T >( py: Python, obj: T ) -> String where T: ToPyObject {
    let args = PyDict::new( py );
    args.set_item( "v0", obj.to_object( py ) ).unwrap();
    let message = py.eval( r#""{}".format(v0)"#, None, Some( args ) ).unwrap();
    let message: String = message.extract().unwrap();
    message
}

pub fn py_err( py: Python, error: PyErr ) -> Box< Error + Send > {
    let message = py_to_string( py, error );

    struct Err( String );
    impl Error for Err {}
    impl fmt::Display for Err {
        fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
            self.0.fmt( fmt )
        }
    }

    impl fmt::Debug for Err {
        fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
            self.0.fmt( fmt )
        }
    }

    Box::new( Err( message ) )
}

pub trait PyResultExt< T > {
    fn unwrap_py( self, py: Python ) -> T;
}

impl< T > PyResultExt< T > for Result< T, PyErr > {
    fn unwrap_py( self, py: Python ) -> T {
        self.map_err( |err| py_err( py, err ) ).unwrap()
    }
}
