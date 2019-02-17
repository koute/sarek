use {
    std::{
        cmp::{
            min
        },
        iter::{
            FusedIterator
        },
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
            },
            split_data_source::{
                SplitDataSource
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

    /// Returns an iterator over `chunk_size` elements of the data source at a time,
    /// starting at the beginning.
    ///
    /// The chunks will not overlap. If `chunk_size` is not divisible by the size of the data source
    /// then the last chunk will not have the length of `chunk_size`.
    ///
    /// ```rust
    /// # use sarek::{Shape, DataSource, DataSourceExt, SliceSource};
    /// let data: &[f32] = &[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ];
    /// let src = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    ///
    /// assert_eq!( src.len(), 3 );
    ///
    /// let mut chunks = src.chunks( 2 );
    ///
    /// assert_eq!( chunks.next().unwrap().len(), 2 );
    /// assert_eq!( chunks.next().unwrap().len(), 1 );
    /// assert!( chunks.next().is_none() );
    /// ```
    fn chunks( &self, chunk_size: usize ) -> DataSourceChunks< Self > {
        DataSourceChunks {
            chunk_size,
            data_source: self,
            index: 0
        }
    }
}

impl< T > DataSourceExt for T where T: DataSource {}

pub struct DataSourceChunks< 'a, S > where S: DataSource + ?Sized {
    chunk_size: usize,
    data_source: &'a S,
    index: usize
}

impl< 'a, S > Iterator for DataSourceChunks< 'a, S > where S: DataSource + ?Sized {
    type Item = SplitDataSource< &'a S >;
    fn next( &mut self ) -> Option< Self::Item > {
        if self.index == self.data_source.len() {
            return None;
        }

        let next_index = min( self.index + self.chunk_size, self.data_source.len() );
        let range = self.index..next_index;
        self.index = next_index;

        let subset = SplitDataSource::new( self.data_source, range.clone() );
        Some( subset )
    }

    fn size_hint( &self ) -> (usize, Option< usize >) {
        if self.chunk_size == 0 {
            return (0, Some( 0 ));
        }

        let total_length = self.data_source.len();
        let mut remaining = total_length / self.chunk_size;
        if total_length * self.chunk_size != 0 {
            remaining += 1;
        }

        (remaining, Some( remaining ))
    }
}

impl< 'a, S > ExactSizeIterator for DataSourceChunks< 'a, S > where S: DataSource + ?Sized {}
impl< 'a, S > FusedIterator for DataSourceChunks< 'a, S > where S: DataSource + ?Sized {}

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
