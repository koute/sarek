use {
    std::{
        cmp::{
            min
        },
        iter::{
            FusedIterator
        },
        sync::{
            Arc
        }
    },
    crate::{
        core::{
            data_source::{
                DataSource,
                DataSourceList,
                DataSourceListExt,
                IntoDataSourceVec
            },
            split_data_source::{
                SplitDataSource
            }
        }
    }
};

/// A data set used for training.
pub struct DataSet< I, O >
    where I: DataSourceList,
          O: DataSourceList
{
    input_list: I,
    expected_output_list: O,
    length: usize
}

type SplitDataSet =
    DataSet<
        Vec< SplitDataSource< Arc< dyn DataSource > > >,
        Vec< SplitDataSource< Arc< dyn DataSource > > >
    >;

impl< I, O > DataSet< I, O >
    where I: DataSourceList,
          O: DataSourceList
{
    pub fn new( input_list: I, expected_output_list: O ) -> Self {
        assert_eq!(
            input_list.data_source_count(),
            expected_output_list.data_source_count(),
            "The training input has {} data sources which is not equal to the amount of data sources in the expected output data where we have {}",
            input_list.data_source_count(),
            expected_output_list.data_source_count()
        );

        assert!( input_list.data_source_count() > 0 );
        assert!( expected_output_list.data_source_count() > 0 );

        let length = input_list.data_source_get( 0 ).unwrap().len();
        for (index, input_data) in input_list.data_sources().enumerate() {
            assert_eq!(
                input_data.len(),
                length,
                "The input data #{} has {} samples which is not equal to the amount of samples in the input data #0 where we have {} samples",
                index,
                input_data.len(),
                length
            );
        }

        for (index, expected_output_data) in expected_output_list.data_sources().enumerate() {
            assert_eq!(
                expected_output_data.len(),
                length,
                "The expected output data #{} has {} samples which is not equal to the amount of samples in the input data #0 where we have {} samples",
                index,
                expected_output_data.len(),
                length
            );
        }

        DataSet {
            input_list,
            expected_output_list,
            length
        }
    }

    pub fn len( &self ) -> usize {
        self.length
    }

    pub fn is_empty( &self ) -> bool {
        self.len() == 0
    }

    pub fn input_list< 'a >( &'a self ) -> impl DataSourceList + 'a {
        &self.input_list
    }

    pub fn expected_output_list< 'a >( &'a self ) -> impl DataSourceList + 'a {
        &self.expected_output_list
    }

    pub fn as_ref< 'a >( &'a self ) -> DataSet< &'a I, &'a O > where &'a I: DataSourceList, &'a O: DataSourceList {
        DataSet {
            input_list: &self.input_list,
            expected_output_list: &self.expected_output_list,
            length: self.length
        }
    }

    /// Splits the data set into two at a given index.
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
    /// let (src_train, src_test) = src.split_at_index( 1 );
    ///
    /// assert_eq!( src_train.len(), 1 );
    /// assert_eq!( src_test.len(), 1 );
    /// ```
    pub fn split_at_index( self, index: usize ) -> (SplitDataSet, SplitDataSet)
        where I: IntoDataSourceVec, O: IntoDataSourceVec
    {
        assert!( index <= self.len() );

        let left_range = 0..index;
        let right_range = index..self.len();

        let input_list = self.input_list.into_vec();
        let expected_output_list = self.expected_output_list.into_vec();

        let left_input_list: Vec< _ > =
            input_list
            .iter()
            .cloned()
            .map( |data| SplitDataSource::new( data, left_range.clone() ) )
            .collect();
        let left_expected_output_list: Vec< _ > =
            expected_output_list
            .iter()
            .cloned()
            .map( |data| SplitDataSource::new( data, left_range.clone() ) )
            .collect();

        let right_input_list: Vec< _ > =
            input_list
            .into_iter()
            .map( |data| SplitDataSource::new( data, right_range.clone() ) )
            .collect();
        let right_expected_output_list: Vec< _ > =
            expected_output_list
            .into_iter()
            .map( |data| SplitDataSource::new( data, right_range.clone() ) )
            .collect();

        let right = DataSet {
            input_list: left_input_list,
            expected_output_list: left_expected_output_list,
            length: left_range.len()
        };

        let left = DataSet {
            input_list: right_input_list,
            expected_output_list: right_expected_output_list,
            length: right_range.len()
        };

        (left, right)
    }

    /// Splits the data set into two.
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
    pub fn clone_and_split( self, split_at: f32 ) -> (SplitDataSet, SplitDataSet)
        where I: IntoDataSourceVec, O: IntoDataSourceVec
    {
        assert!( split_at >= 0.0 );
        assert!( split_at <= 1.0 );

        let index = (self.len() as f32 * split_at) as usize;
        self.split_at_index( index )
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
    pub fn chunks( &'_ self, chunk_size: usize ) ->
        impl ExactSizeIterator< Item = DataSet< impl DataSourceList + '_, impl DataSourceList + '_ > > + FusedIterator
    {
        struct Iter< 'a, I, O > where I: DataSourceList, O: DataSourceList {
            chunk_size: usize,
            input_list: &'a I,
            expected_output_list: &'a O,
            length: usize,
            index: usize
        }

        impl< 'a, I, O > Iterator for Iter< 'a, I, O > where I: DataSourceList, O: DataSourceList {
            type Item = DataSet< Vec< SplitDataSource< &'a dyn DataSource > >, Vec< SplitDataSource< &'a dyn DataSource > > >;
            fn next( &mut self ) -> Option< Self::Item > {
                if self.index == self.length {
                    return None;
                }

                let next_index = min( self.index + self.chunk_size, self.length );
                let range = self.index..next_index;
                self.index = next_index;

                // TODO: This is horribly slow. Cache those `Vec`s somehow.
                let input_list: Vec< _ > =
                    self.input_list.data_sources()
                    .map( |data| SplitDataSource::new( data, range.clone() ) )
                    .collect();
                let expected_output_list: Vec< _ > =
                    self.expected_output_list.data_sources()
                    .map( |data| SplitDataSource::new( data, range.clone() ) )
                    .collect();

                Some( DataSet {
                    input_list,
                    expected_output_list,
                    length: range.end - range.start
                })
            }

            fn size_hint( &self ) -> (usize, Option< usize >) {
                if self.chunk_size == 0 {
                    return (0, Some( 0 ));
                }

                let total_length = self.length;
                let mut remaining = total_length / self.chunk_size;
                if total_length * self.chunk_size != 0 {
                    remaining += 1;
                }

                (remaining, Some( remaining ))
            }
        }

        impl< 'a, I, O > ExactSizeIterator for Iter< 'a, I, O > where I: DataSourceList, O: DataSourceList {}
        impl< 'a, I, O > FusedIterator for Iter< 'a, I, O > where I: DataSourceList, O: DataSourceList {}

        Iter {
            chunk_size,
            input_list: &self.input_list,
            expected_output_list: &self.expected_output_list,
            length: self.length,
            index: 0
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::{
            core::{
                data_source::{
                    DataSourceListExt
                },
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
        assert_eq!( data_set.input_list().data_sources().len(), 1 );
        assert_eq!( data_set.input_list().data_sources().next().unwrap().shape(), Shape::new_2d( 1, 2 ) );
    }

}
