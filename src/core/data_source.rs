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
pub trait DataSource {
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
    fn raw_gather_bytes_into( &self, indices: &dyn ToIndices, output: &mut [u8] );

    /// Returns whenever this data source is empty.
    fn is_empty( &self ) -> bool {
        self.len() == 0
    }
}

pub trait DataSourceExt: DataSource {
    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        self.raw_gather_bytes_into( &indices, output );
    }

    /// Copies data to `output` from `indices`.
    ///
    /// Will panic if the `data_type` of this data source doesn't match
    /// the type of the `output` slice, or for any of the same reasons as `raw_gather_bytes_into`.
    fn gather_into< T, I >( &self, indices: I, output: &mut [T] ) where I: ToIndices, T: DataType {
        assert_eq!( T::TYPE, self.data_type() );
        self.gather_bytes_into( indices, as_byte_slice_mut( output ) );
    }
}

impl< T > DataSourceExt for T where T: DataSource {}

macro_rules! impl_data_source_proxy {
    (($($ty_args:tt)*) DataSource for $type:ty $(where $($where_clause:tt)*)?) => {
        impl< $($ty_args)* > DataSource for $type where $($($where_clause)*)? {
            fn data_type( &self ) -> Type {
                DataSource::data_type( self.deref() )
            }

            fn shape( &self ) -> Shape {
                DataSource::shape( self.deref() )
            }

            fn len( &self ) -> usize {
                DataSource::len( self.deref() )
            }

            fn raw_gather_bytes_into( &self, indices: &dyn ToIndices, output: &mut [u8] ) {
                DataSource::raw_gather_bytes_into( self.deref(), indices, output )
            }
        }
    }
}

impl_data_source_proxy!( ('r, S) DataSource for &'r S where S: DataSource + ?Sized );
impl_data_source_proxy!( (S) DataSource for Rc< S > where S: DataSource + ?Sized );
impl_data_source_proxy!( (S) DataSource for Arc< S > where S: DataSource + ?Sized );
impl_data_source_proxy!( () DataSource for Box< DataSource > );
