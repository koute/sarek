use {
    std::{
        ops::{
            Range
        }
    },
    crate::{
        core::{
            data_source::{
                DataSource
            },
            indices::{
                Indices,
                ReusedIndices,
                ToIndices
            },
            data_type::{
                Type
            },
            shape::{
                Shape
            }
        }
    }
};

pub struct SplitDataSource< S > where S: DataSource {
    range: Range< usize >,
    inner: S
}

impl< S > SplitDataSource< S > where S: DataSource {
    pub(crate) fn new( data_source: S, range: Range< usize > ) -> Self {
        SplitDataSource {
            range,
            inner: data_source
        }
    }
}

impl< S > DataSource for SplitDataSource< S > where S: DataSource {
    fn data_type( &self ) -> Type {
        self.inner.data_type()
    }

    fn shape( &self ) -> Shape {
        self.inner.shape()
    }

    fn len( &self ) -> usize {
        self.range.len()
    }

    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        let extra_offset = self.range.start;
        let indices = indices.to_indices( self.len() );
        let indices = match indices {
            Indices::Continuous { range } => {
                let range = extra_offset + range.start..extra_offset + range.end;

                assert!( range.end <= self.range.end );
                Indices::Continuous { range }
            },
            Indices::Disjoint { offset, indices } => {
                let offset = extra_offset + offset;

                assert!( indices.iter().cloned().all( |index| offset + index <= self.len() ) );
                Indices::Disjoint {
                    offset,
                    indices
                }
            }
        };

        self.inner.gather_bytes_into( ReusedIndices( indices ), output )
    }
}
