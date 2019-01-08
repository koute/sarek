use {
    std::{
        ops::{
            Deref
        },
        rc::Rc,
        sync::Arc
    },
    crate::{
        core::{
            data_type::{
                DataType,
                Type,
                as_byte_slice_mut
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

/// A source of data.
pub trait DataSource: Sized {
    /// The basic type of which the data is composed of.
    fn data_type( &self ) -> Type;

    /// The shape of a single element.
    fn shape( &self ) -> Shape;

    /// The number of elements.
    fn len( &self ) -> usize;

    /// Copies data to `output` from `indices`.
    ///
    /// Will panic if the `output`'s size is not divisible by the byte size
    /// of a single element, is not properly aligned for the underlying data type
    /// or if its size doesn't exactly match the sum of the byte sizes of the elements
    /// which it is supposed to copy to the `output`.
    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices;

    /// Copies data to `output` from `indices`.
    ///
    /// Will panic if the `data_type` of this data source doesn't match
    /// the type of the `output` slice, or for any of the same reasons as `gather_bytes_into`.
    fn gather_into< T, I >( &self, indices: I, output: &mut [T] ) where I: ToIndices, T: DataType {
        assert_eq!( T::TYPE, self.data_type() );
        self.gather_bytes_into( indices, as_byte_slice_mut( output ) );
    }

    /// Returns whenever this data source is empty.
    fn is_empty( &self ) -> bool {
        self.len() == 0
    }
}

impl< 'r, S > DataSource for &'r S where S: DataSource {
    fn data_type( &self ) -> Type {
        DataSource::data_type( *self )
    }

    fn shape( &self ) -> Shape {
        DataSource::shape( *self )
    }

    fn len( &self ) -> usize {
        DataSource::len( *self )
    }

    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        DataSource::gather_bytes_into( *self, indices, output )
    }
}

impl< S > DataSource for Rc< S > where S: DataSource {
    fn data_type( &self ) -> Type {
        DataSource::data_type( self.deref() )
    }

    fn shape( &self ) -> Shape {
        DataSource::shape( self.deref() )
    }

    fn len( &self ) -> usize {
        DataSource::len( self.deref() )
    }

    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        DataSource::gather_bytes_into( self.deref(), indices, output )
    }
}

impl< S > DataSource for Arc< S > where S: DataSource {
    fn data_type( &self ) -> Type {
        DataSource::data_type( self.deref() )
    }

    fn shape( &self ) -> Shape {
        DataSource::shape( self.deref() )
    }

    fn len( &self ) -> usize {
        DataSource::len( self.deref() )
    }

    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        DataSource::gather_bytes_into( self.deref(), indices, output )
    }
}
