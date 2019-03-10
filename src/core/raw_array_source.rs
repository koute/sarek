use {
    std::{
        slice
    },
    crate::{
        core::{
            array::{
                ArrayRef,
                ArrayMut,
                ToArrayRef
            },
            data_source::{
                DataSource,
                DataSourceList
            },
            data_type::{
                DataType,
                Type,
                cast_slice,
                cast_slice_mut
            },
            indices::{
                ToIndices
            },
            shape::{
                Shape
            }
        }
    }
};

pub struct RawArraySource {
    pointer: *mut u8,
    length: usize,
    shape: Shape,
    data_type: Type
}

unsafe impl Send for RawArraySource {}
unsafe impl Sync for RawArraySource {}

impl Drop for RawArraySource {
    fn drop( &mut self ) {
        unsafe {
            libc::free( self.pointer as *mut libc::c_void );
        }
    }
}

impl RawArraySource {
    pub unsafe fn from_pointer( pointer: *mut u8, length: usize, shape: Shape, data_type: Type ) -> Self {
        RawArraySource {
            pointer,
            length,
            shape,
            data_type
        }
    }

    pub fn new_uninitialized( length: usize, shape: Shape, data_type: Type ) -> Self {
        let byte_length = length * shape.product() * data_type.byte_size();
        let pointer = unsafe {
            libc::malloc( byte_length ) as *mut u8
        };

        RawArraySource {
            pointer,
            length,
            shape,
            data_type
        }
    }

    pub(crate) fn as_bytes( &self ) -> &[u8] {
        unsafe { slice::from_raw_parts( self.pointer, self.length * self.shape.product() * self.data_type.byte_size() ) }
    }

    pub(crate) fn as_bytes_mut( &mut self ) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut( self.pointer, self.length * self.shape.product() * self.data_type.byte_size() ) }
    }

    pub(crate) fn as_slice< T >( &self ) -> Option< &[T] > where T: DataType {
        if self.data_type == T::TYPE {
            Some( cast_slice( self.as_bytes() ) )
        } else {
            None
        }
    }

    pub(crate) fn as_slice_mut< T >( &mut self ) -> Option< &mut [T] > where T: DataType {
        if self.data_type == T::TYPE {
            Some( cast_slice_mut( self.as_bytes_mut() ) )
        } else {
            None
        }
    }
}

impl DataSourceList for RawArraySource {
    fn data_source_count( &self ) -> usize { 1 }
    fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
        if index == 0 { Some( self ) } else { None }
    }
}

impl DataSource for RawArraySource {
    fn data_type( &self ) -> Type {
        self.data_type
    }

    fn shape( &self ) -> Shape {
        self.shape.clone()
    }

    fn len( &self ) -> usize {
        self.length
    }

    fn raw_gather_bytes_into( &self, indices: &dyn ToIndices, output: &mut [u8] ) {
        let input = self.as_bytes();
        let input = ArrayRef::new( self.shape(), self.data_type(), input );
        let mut output = ArrayMut::new( self.shape(), self.data_type(), output );
        output.gather_from( indices, &input );
    }
}

impl ToArrayRef for RawArraySource {
    fn to_array_ref( &self ) -> ArrayRef {
        ArrayRef::new( self.shape(), self.data_type(), self.as_bytes() )
    }
}
