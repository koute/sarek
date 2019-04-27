use {
    std::{
        fmt,
        marker::{
            PhantomData
        },
        ops::{
            Deref,
            DerefMut
        },
        slice
    },
    pyo3::{
        prelude::*,
        types::{
            PyDict,
            PyTuple
        },
        AsPyPointer
    },
    crate::{
        backend::{
            keras::{
                ffi,
                py_utils::{
                    py_err,
                    py_type_name
                }
            }
        },
        core::{
            data_type::{
                DataType,
                Type
            },
            raw_array_source::{
                RawArraySource
            },
            shape::{
                Shape
            },
            type_cast_error::{
                TypeCastError
            }
        }
    }
};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArrayOrder {
    RowMajor,
    #[allow(dead_code)]
    ColumnMajor
}

struct ArrayInit< 'a > {
    order: ArrayOrder,
    shape: Shape,
    kind: &'a str
}

fn dtype( py: Python, obj: &PyObject ) -> String {
    // https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.dtype.html
    obj
        .getattr( py, "dtype" ).map_err( |err| py_err( py, err ) ).unwrap()
        .getattr( py, "name" ).unwrap()
        .extract( py ).unwrap()
}

pub struct PyArray {
    obj: PyObject,
    shape: Shape,
    ty: Type
}

unsafe fn as_array_object( obj: &PyObject ) -> &ffi::PyArrayObject {
    &*(obj.as_ptr() as *const ffi::PyArrayObject)
}

unsafe fn as_array_object_mut( obj: &mut PyObject ) -> &mut ffi::PyArrayObject {
    &mut *(obj.as_ptr() as *mut ffi::PyArrayObject)
}

impl PyArray {
    pub(crate) unsafe fn from_object_unchecked( py: Python, obj: PyObject ) -> Self {
        let ty = match dtype( py, &obj ).as_str() {
            "float32" => Type::F32,
            "uint32" => Type::U32,
            "uint16" => Type::U16,
            "uint8" => Type::U8,
            "int32" => Type::I32,
            "int16" => Type::I16,
            "int8" => Type::I8,
            ty => unimplemented!( "Unhandled array type: {}", ty )
        };

        let internal = as_array_object( &obj );
        let slice = slice::from_raw_parts( internal.dims, internal.nd as usize );
        let shape = slice.into_iter().cloned().map( |size| size as usize ).collect();

        PyArray { obj, shape, ty }
    }

    fn as_array_object( &self ) -> &ffi::PyArrayObject {
        unsafe { as_array_object( &self.obj ) }
    }

    fn as_array_object_mut( &mut self ) -> &mut ffi::PyArrayObject {
        unsafe { as_array_object_mut( &mut self.obj ) }
    }

    pub(crate) fn new( py: Python, shape: Shape, ty: Type ) -> PyArray {
        PyArray::new_internal( py, ArrayInit {
            order: ArrayOrder::RowMajor,
            shape,
            kind: py_type_name( ty )
        })
    }

    fn new_internal( py: Python, init: ArrayInit ) -> PyArray {
        let np = py.import( "numpy" ).unwrap();

        let kwargs = PyDict::new( py );
        let shape = PyTuple::new( py, &init.shape );
        kwargs.set_item( "shape", shape ).unwrap();

        let order = match init.order {
            ArrayOrder::RowMajor => "C",
            ArrayOrder::ColumnMajor => "F"
        };

        kwargs.set_item( "order", order ).unwrap();
        kwargs.set_item( "dtype", init.kind ).unwrap();
        let obj = np.get( "ndarray" ).unwrap().call( (), Some( &kwargs ) ).unwrap().to_object( py );
        unsafe { PyArray::from_object_unchecked( py, obj ) }
    }

    /// Returns the number of dimensions for this array.
    pub fn dimension_count( &self ) -> usize {
        self.as_array_object().nd as _
    }

    /// Returns the shape of this array.
    pub fn shape( &self ) -> Shape {
        self.shape.clone()
    }

    pub fn reshape< S >( &self, py: Python, shape: S ) -> PyArray where S: Into< Shape > {
        let shape = shape.into();

        let current_shape = self.shape();
        assert_eq!(
            shape.product(),
            current_shape.product(),
            "Tried to reshape an PyArray from {} into {} where their products don't match ({} != {})",
            current_shape,
            shape,
            current_shape.product(),
            shape.product()
        );

        let shape = PyTuple::new( py, &shape );
        let obj = self.obj.getattr( py, "reshape" ).unwrap().call( py, (shape,), None ).unwrap().to_object( py );
        unsafe { PyArray::from_object_unchecked( py, obj ) }
    }

    /// Casts the array into a typed variant.
    pub fn into_typed< T: DataType >( self ) -> Result< TypedPyArray< T >, TypeCastError< Self > > {
        if self.ty == T::TYPE {
            Ok( TypedPyArray( self, PhantomData ) )
        } else {
            Err( TypeCastError {
                source: "an array",
                target: "a typed array",
                source_ty: self.ty.into(),
                target_ty: T::TYPE,
                obj: self
            })
        }
    }

    /// Checks whenever the array's elements are of the given type.
    pub fn data_is< T: DataType >( &self ) -> bool {
        self.ty == T::TYPE
    }

    pub fn data_type( &self ) -> Type {
        self.ty
    }

    /// Extracts a slice containing the whole array.
    pub fn as_bytes( &self ) -> &[u8] {
        unsafe {
            slice::from_raw_parts( self.as_array_object().data as *const u8, self.shape().product() * self.data_type().byte_size() )
        }
    }

    /// Extracts a mutable slice containing the whole array.
    pub fn as_bytes_mut( &mut self ) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut( self.as_array_object().data as *mut u8, self.shape().product() * self.data_type().byte_size() )
        }
    }

    pub(crate) fn as_py_obj( &self ) -> &PyObject {
        &self.obj
    }

    // This is kinda sketchy, but it works I guess.
    pub(crate) fn into_raw_array( mut self ) -> RawArraySource {
        assert!( self.dimension_count() > 1 );

        let original_shape = self.shape();
        let length = original_shape.into_iter().next().unwrap();
        let shape = original_shape.into_iter().skip( 1 ).collect();
        let data_type = self.data_type();

        let pointer;
        {
            let array_object = self.as_array_object_mut();

            // Make sure the data is actually owned.
            assert_ne!( array_object.flags & ffi::NPY_ARRAY_OWNDATA, 0 );

            // And that the items themselves have no refcounts.
            assert_eq!( unsafe { &*array_object.descr }.flags & ffi::NPY_ITEM_REFCOUNT, 0 );

            {
                // Zero out the dimensions since we're taking its data away.
                let dims = unsafe { slice::from_raw_parts_mut( array_object.dims, array_object.nd as usize ) };
                for dim in dims.iter_mut() {
                    *dim = 0;
                }
            }

            pointer = array_object.data;
            unsafe {
                ffi::PyTraceMalloc_Untrack( ffi::NPY_TRACE_DOMAIN, pointer as libc::uintptr_t );
            }

            array_object.data = 0 as _;
        }

        unsafe {
            RawArraySource::from_pointer( pointer, length, shape, data_type )
        }
    }
}

impl ToPyObject for PyArray {
    fn to_object( &self, py: Python ) -> PyObject {
        self.obj.clone_ref( py )
    }
}

pub struct TypedPyArray< T >( PyArray, PhantomData< T > );

impl< T: DataType > TypedPyArray< T > {
    pub fn new( py: Python, shape: Shape ) -> Self {
        let array = PyArray::new( py, shape, T::TYPE );
        TypedPyArray( array, PhantomData )
    }

    /// Converts this array to a `Vec`.
    pub fn to_vec( &self ) -> Vec< T > {
        self.as_slice().to_vec()
    }

    /// Extracts a slice containing the whole array.
    pub fn as_slice( &self ) -> &[T] {
        unsafe {
            slice::from_raw_parts( self.as_array_object().data as *const T, self.shape().product() )
        }
    }

    /// Extracts a mutable slice containing the whole array.
    pub fn as_slice_mut( &mut self ) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut( self.as_array_object().data as *mut T, self.shape().product() )
        }
    }
}

impl< T > Deref for TypedPyArray< T > {
    type Target = PyArray;

    #[inline]
    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl< T > DerefMut for TypedPyArray< T > {
    #[inline]
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl< T: DataType > Into< Vec< T > > for TypedPyArray< T > {
    #[inline]
    fn into( self ) -> Vec< T > {
        self.to_vec()
    }
}

impl< 'a, T: DataType > Into< Vec< T > > for &'a TypedPyArray< T > {
    #[inline]
    fn into( self ) -> Vec< T > {
        self.to_vec()
    }
}

impl< 'a, T: DataType > Into< Vec< T > > for &'a mut TypedPyArray< T > {
    #[inline]
    fn into( self ) -> Vec< T > {
        self.to_vec()
    }
}

impl< T: DataType > Into< PyArray > for TypedPyArray< T > {
    #[inline]
    fn into( self ) -> PyArray {
        self.0
    }
}

impl< T: DataType + fmt::Debug > fmt::Debug for TypedPyArray< T > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_list().entries( self.as_slice().iter() ).finish()
    }
}

impl< T > ToPyObject for TypedPyArray< T > {
    fn to_object( &self, py: Python ) -> PyObject {
        self.0.obj.clone_ref( py )
    }
}
