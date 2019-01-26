use {
    std::{
        cmp::{
            min
        },
        iter::{
            FusedIterator
        }
    },
    crate::{
        core::{
            data_source::{
                DataSource
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

/// A data set used for training.
pub struct DataSet< I, O >
    where I: DataSource,
          O: DataSource
{
    input_data: I,
    expected_output_data: O
}

impl< I, O > DataSet< I, O >
    where I: DataSource,
          O: DataSource
{
    pub fn new( input_data: I, expected_output_data: O ) -> Self {
        assert_eq!(
            input_data.len(),
            expected_output_data.len(),
            "The training input data has {} samples which is not equal to the amount of samples in the expected output data where we have {} samples",
            input_data.len(),
            expected_output_data.len()
        );

        DataSet {
            input_data,
            expected_output_data
        }
    }

    pub fn len( &self ) -> usize {
        self.input_data.len()
    }

    pub fn is_empty( &self ) -> bool {
        self.len() == 0
    }

    pub fn input_shape( &self ) -> Shape {
        self.input_data.shape()
    }

    pub fn output_shape( &self ) -> Shape {
        self.expected_output_data.shape()
    }

    pub fn input_data( &self ) -> &I {
        &self.input_data
    }

    pub fn expected_output_data( &self ) -> &O {
        &self.expected_output_data
    }

    pub fn as_ref( &self ) -> DataSet< &I, &O > {
        DataSet {
            input_data: &self.input_data,
            expected_output_data: &self.expected_output_data
        }
    }

    /// Clones the data sources and splits the set into two at a given index.
    ///
    /// ```rust
    /// # use sarek::{Shape, DataSource, DataSet, SliceSource};
    /// let data: &[f32] = &[ 1.0, 2.0, 3.0, 4.0 ];
    /// let expected_data: &[u32] = &[ 10, 20 ];
    /// let src_in = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    /// let src_out = SliceSource::from( Shape::new_1d( 1 ), expected_data );
    /// let mut src = DataSet::new( src_in, src_out );
    ///
    /// assert_eq!( src.len(), 2 );
    ///
    /// let (src_train, src_test) = src.clone_and_split_at_index( 1 );
    ///
    /// assert_eq!( src_train.len(), 1 );
    /// assert_eq!( src_test.len(), 1 );
    /// ```
    pub fn clone_and_split_at_index( self, index: usize )
        -> (
            DataSet< SplitDataSource< I >, SplitDataSource< O > >,
            DataSet< SplitDataSource< I >, SplitDataSource< O > >
        )
        where I: Clone, O: Clone
    {
        assert!( index <= self.len() );

        let left_range = 0..index;
        let right_range = index..self.len();

        let right = DataSet {
            input_data: SplitDataSource::new( self.input_data.clone(), right_range.clone() ),
            expected_output_data: SplitDataSource::new( self.expected_output_data.clone(), right_range )
        };

        let left = DataSet {
            input_data: SplitDataSource::new( self.input_data, left_range.clone() ),
            expected_output_data: SplitDataSource::new( self.expected_output_data, left_range )
        };

        (left, right)
    }

    /// Clones the data sources and splits the set into two.
    ///
    /// ```rust
    /// # use sarek::{Shape, DataSource, DataSet, SliceSource};
    /// let data: &[f32] = &[ 1.0, 2.0, 3.0, 4.0 ];
    /// let expected_data: &[u32] = &[ 10, 20 ];
    /// let src_in = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    /// let src_out = SliceSource::from( Shape::new_1d( 1 ), expected_data );
    /// let mut src = DataSet::new( src_in, src_out );
    ///
    /// assert_eq!( src.len(), 2 );
    ///
    /// let (src_train, src_test) = src.clone_and_split( 0.5 );
    ///
    /// assert_eq!( src_train.len(), 1 );
    /// assert_eq!( src_test.len(), 1 );
    /// ```
    pub fn clone_and_split( self, split_at: f32 )
        -> (
            DataSet< SplitDataSource< I >, SplitDataSource< O > >,
            DataSet< SplitDataSource< I >, SplitDataSource< O > >
        )
        where I: Clone, O: Clone
    {
        assert!( split_at >= 0.0 );
        assert!( split_at <= 1.0 );

        let index = (self.len() as f32 * split_at) as usize;
        self.clone_and_split_at_index( index )
    }

    /// Returns an iterator over `chunk_size` elements of the data set at a time,
    /// starting at the beginning.
    ///
    /// The chunks will not overlap. If `chunk_size` is not divisible by the size of the data set
    /// then the last chunk will not have the length of `chunk_size`.
    ///
    /// ```rust
    /// # use sarek::{Shape, DataSource, DataSet, SliceSource};
    /// let data: &[f32] = &[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ];
    /// let expected_data: &[u32] = &[ 10, 20, 30 ];
    /// let src_in = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    /// let src_out = SliceSource::from( Shape::new_1d( 1 ), expected_data );
    /// let mut src = DataSet::new( src_in, src_out );
    ///
    /// assert_eq!( src.len(), 3 );
    ///
    /// let mut chunks = src.chunks( 2 );
    ///
    /// assert_eq!( chunks.next().unwrap().len(), 2 );
    /// assert_eq!( chunks.next().unwrap().len(), 1 );
    /// assert!( chunks.next().is_none() );
    /// ```
    pub fn chunks< 'a >( &'a self, chunk_size: usize ) ->
        impl ExactSizeIterator< Item = DataSet< impl DataSource + 'a, impl DataSource + 'a > > + FusedIterator
    {
        struct Iter< 'a, I, O > where I: DataSource, O: DataSource {
            chunk_size: usize,
            data_set: &'a DataSet< I, O >,
            index: usize
        }

        impl< 'a, I, O > Iterator for Iter< 'a, I, O > where I: DataSource, O: DataSource {
            type Item = DataSet< SplitDataSource< &'a I >, SplitDataSource< &'a O > >;
            fn next( &mut self ) -> Option< Self::Item > {
                if self.index == self.data_set.len() {
                    return None;
                }

                let next_index = min( self.index + self.chunk_size, self.data_set.len() );
                let range = self.index..next_index;
                self.index = next_index;

                let subset = DataSet {
                    input_data: SplitDataSource::new( &self.data_set.input_data, range.clone() ),
                    expected_output_data: SplitDataSource::new( &self.data_set.expected_output_data, range )
                };

                Some( subset )
            }

            fn size_hint( &self ) -> (usize, Option< usize >) {
                if self.chunk_size == 0 {
                    return (0, Some( 0 ));
                }

                let total_length = self.data_set.len();
                let mut remaining = total_length / self.chunk_size;
                if total_length * self.chunk_size != 0 {
                    remaining += 1;
                }

                (remaining, Some( remaining ))
            }
        }

        impl< 'a, I, O > ExactSizeIterator for Iter< 'a, I, O > where I: DataSource, O: DataSource {}
        impl< 'a, I, O > FusedIterator for Iter< 'a, I, O > where I: DataSource, O: DataSource {}

        Iter {
            chunk_size,
            data_set: self,
            index: 0
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::{
            core::{
                shape::{
                    Shape
                },
                slice_source::{
                    SliceSource
                }
            }
        },
        super::{
            DataSet
        }
    };

    #[test]
    fn test_data_set_basics() {
        let input_data: Vec< u32 > = vec![ 1, 2, 3, 4 ];
        let output_data: Vec< u32 > = vec![ 10, 20 ];
        let inputs = SliceSource::from( Shape::new_2d( 1, 2 ), input_data );
        let outputs = SliceSource::from( Shape::new_1d( 1 ), output_data );
        let data_set = DataSet::new( inputs, outputs );

        assert_eq!( data_set.len(), 2 );
        assert_eq!( data_set.is_empty(), false );
        assert_eq!( data_set.input_shape(), Shape::new_2d( 1, 2 ) );
    }

}
