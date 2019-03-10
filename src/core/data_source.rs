use {
    std::{
        cmp::{
            min
        },
        iter::{
            FusedIterator
        },
        ops::{
            Deref,
            Index
        },
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
pub trait DataSource: Send + Sync {
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

pub trait DataSourceList {
    fn data_source_count( &self ) -> usize;
    fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource >;
}

pub trait IntoDataSourceVec: DataSourceList + Sized {
    fn into_vec( self ) -> Vec< Arc< DataSource > >;
}

pub trait DataSourceListExt: DataSourceList {
    fn data_sources( &self ) -> DataSources< Self > {
        DataSources {
            list: self,
            index: 0
        }
    }
}

impl< T > DataSourceListExt for T where T: DataSourceList {}

pub struct DataSources< 'a, T > where T: DataSourceList + ?Sized {
    list: &'a T,
    index: usize
}

impl< 'a, T > Iterator for DataSources< 'a, T > where T: DataSourceList + ?Sized {
    type Item = &'a dyn DataSource;
    fn next( &mut self ) -> Option< Self::Item > {
        let output = self.list.data_source_get( self.index );
        self.index += 1;
        output
    }

    fn size_hint( &self ) -> (usize, Option< usize >) {
        let length = self.list.data_source_count() - self.index;
        (length, Some( length ))
    }
}

impl< 'a, T > ExactSizeIterator for DataSources< 'a, T > where T: DataSourceList + ?Sized {}
impl< 'a, T > FusedIterator for DataSources< 'a, T > where T: DataSourceList + ?Sized {}
impl< 'a, T > Index< usize > for DataSources< 'a, T > where T: DataSourceList + ?Sized {
    type Output = dyn DataSource + 'a;

    fn index( &self, index: usize ) -> &Self::Output {
        self.list.data_source_get( self.index + index ).unwrap()
    }
}

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

macro_rules! impl_data_source_list_proxy {
    (($($ty_args:tt)*) DataSourceList for $type:ty $(where $($where_clause:tt)*)?) => {
        impl< $($ty_args)* > DataSourceList for $type where $($($where_clause)*)? {
            fn data_source_count( &self ) -> usize {
                DataSourceList::data_source_count( self.deref() )
            }

            fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
                DataSourceList::data_source_get( self.deref(), index )
            }
        }
    }
}

macro_rules! impl_single_element_data_source_list {
    (($($ty_args:tt)*) DataSourceList for $type:ty $(where $($where_clause:tt)*)?) => {
        impl< $($ty_args)* > DataSourceList for $type where $($($where_clause)*)? {
            fn data_source_count( &self ) -> usize {
                1
            }

            fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
                if index == 0 {
                    Some( &*self )
                } else {
                    None
                }
            }
        }
    }
}

impl_data_source_proxy!( ('r, S) DataSource for &'r S where S: DataSource + ?Sized );
impl_data_source_proxy!( (S) DataSource for Arc< S > where S: DataSource + ?Sized );
impl_data_source_proxy!( () DataSource for Box< DataSource > );

impl_data_source_list_proxy!( ('r, S) DataSourceList for &'r S where S: DataSourceList + ?Sized );
impl_data_source_list_proxy!( (S) DataSourceList for Arc< S > where S: DataSourceList + Send + ?Sized );
impl_data_source_list_proxy!( () DataSourceList for Box< DataSourceList > );

impl_single_element_data_source_list!( () DataSourceList for Box< DataSource > );

impl DataSourceList for () {
    fn data_source_count( &self ) -> usize {
        0
    }

    fn data_source_get( &self, _: usize ) -> Option< &dyn DataSource > {
        None
    }
}

impl< T > DataSourceList for &[T] where T: DataSource {
    fn data_source_count( &self ) -> usize {
        self.len()
    }

    fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
        self.get( index ).map( |value| {
            let value: &dyn DataSource = value;
            value
        })
    }
}

impl< T > DataSourceList for &mut [T] where T: DataSource {
    fn data_source_count( &self ) -> usize {
        self.len()
    }

    fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
        self.get( index ).map( |value| {
            let value: &dyn DataSource = value;
            value
        })
    }
}

impl< T > DataSourceList for Vec< T > where T: DataSource {
    fn data_source_count( &self ) -> usize {
        self.len()
    }

    fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
        self.get( index ).map( |value| {
            let value: &dyn DataSource = value;
            value
        })
    }
}

impl< T > IntoDataSourceVec for T where T: DataSource + DataSourceList + 'static {
    fn into_vec( self ) -> Vec< Arc< DataSource > > {
        let boxed: Arc< DataSource > = Arc::new( self );
        vec![ boxed ]
    }
}

impl IntoDataSourceVec for () {
    fn into_vec( self ) -> Vec< Arc< DataSource > > {
        Vec::new()
    }
}

impl< T > IntoDataSourceVec for Vec< T > where T: DataSource + 'static {
    default fn into_vec( self ) -> Vec< Arc< DataSource > > {
        self.into_iter().map( |data| {
            let boxed: Arc< DataSource > = Arc::new( data );
            boxed
        }).collect()
    }
}

impl IntoDataSourceVec for Vec< Arc< DataSource > > {
    fn into_vec( self ) -> Vec< Arc< DataSource > > {
        self
    }
}

impl< T > IntoDataSourceVec for &[T] where T: DataSource + Clone + 'static {
    default fn into_vec( self ) -> Vec< Arc< DataSource > > {
        self.iter().map( |data| {
            let boxed: Arc< DataSource > = Arc::new( data.clone() );
            boxed
        }).collect()
    }
}

impl IntoDataSourceVec for &[Arc< DataSource >] {
    fn into_vec( self ) -> Vec< Arc< DataSource > > {
        self.iter().cloned().collect()
    }
}

macro_rules! impl_data_source_list {
    (@consume $token:tt) => {};
    (@impl $($type:ident)*) => {
        impl< $($type),* > DataSourceList for ($($type,)*) where $($type: DataSource),* {
            fn data_source_count( &self ) -> usize {
                let mut counter = 0;
                $(
                    impl_data_source_list!( @consume $type );
                    counter += 1;
                )*
                counter
            }

            fn data_source_get( &self, index: usize ) -> Option< &dyn DataSource > {
                let mut counter = 0;
                $(
                    if index == counter {
                        return Some( &access_tuple!( self, $type ) );
                    }
                    counter += 1;
                )*

                std::mem::drop( counter );
                None
            }
        }
    };

    (@call_1 [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_data_source_list!( @impl $($type)* );
        impl_data_source_list!( @call_1 [$($dummy_type)*] [$($type)* $lhs] );
    };

    (@call_1 [] [$($type:ident)*]) => {};

    (@call [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_data_source_list!( @call_1 [$($dummy_type)*] [$lhs $($type)*] );
    };

    () => {
        impl_data_source_list!(
            @call
                [
                    T00 T01 T02 T03 T04 T05 T06 T07 T08 T09
                    T10 T11 T12 T13 T14 T15 T16 T17 T18 T19
                    T20 T21 T22 T23 T24 T25 T26 T27 T28 T29
                ]
                []
        );
    };
}

impl_data_source_list!();
