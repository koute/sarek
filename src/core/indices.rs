pub use {
    std::{
        ops::{
            Range,
            RangeFrom,
            RangeFull,
            RangeInclusive,
            RangeTo,
            RangeToInclusive
        },
        iter::{
            self,
            FusedIterator
        },
        slice,
        usize
    },
    crate::{
        core::{
            into_range::{
                IntoRange
            }
        }
    }
};

#[derive(Clone, Debug)]
pub enum Indices< 'a > {
    Continuous {
        range: Range< usize >
    },
    Disjoint {
        offset: usize,
        indices: &'a [usize]
    }
}

impl< 'a > Indices< 'a > {
    pub fn empty() -> Self {
        Indices::Continuous { range: 0..0 }
    }

    pub fn get( &self, nth: usize ) -> usize {
        match *self {
            Indices::Continuous { ref range } => {
                assert!( nth < range.len() );
                range.start + nth
            },
            Indices::Disjoint { offset, indices } => offset + indices[ nth ]
        }
    }

    pub fn len( &self ) -> usize {
        match *self {
            Indices::Continuous { ref range } => range.len(),
            Indices::Disjoint { indices, .. } => indices.len()
        }
    }
}

impl< 'a > Iterator for Indices< 'a > {
    type Item = usize;
    fn next( &mut self ) -> Option< Self::Item > {
        match *self {
            Indices::Continuous { ref mut range } => range.next(),
            Indices::Disjoint { offset, ref mut indices } => {
                if indices.is_empty() {
                    return None;
                }

                let index = indices[ 0 ];
                *indices = &indices[ 1.. ];

                Some( offset + index )
            }
        }
    }

    fn size_hint( &self ) -> (usize, Option< usize >) {
        let length = self.len();
        (length, Some( length ))
    }
}

impl< 'a > ExactSizeIterator for Indices< 'a > {}
impl< 'a > FusedIterator for Indices< 'a > {}

pub trait ToIndices {
    fn to_indices( &self, container_length: usize ) -> Indices;
}

impl< 'a > ToIndices for &'a Vec< usize > {
    fn to_indices( &self, _: usize ) -> Indices {
        Indices::Disjoint { offset: 0, indices: &self }
    }
}

impl< 'a > ToIndices for &'a [usize] {
    fn to_indices( &self, _: usize ) -> Indices {
        Indices::Disjoint { offset: 0, indices: &self }
    }
}

macro_rules! impl_for {
    ($($target_ty:ty),*) => {
        $(
            impl ToIndices for $target_ty {
                fn to_indices( &self, container_length: usize ) -> Indices {
                    Indices::Continuous { range: self.clone().into_range( container_length ) }
                }
            }
        )*
    };
}

impl_for! {
    usize,
    Range< usize >,
    RangeFrom< usize >,
    RangeFull,
    RangeInclusive< usize >,
    RangeTo< usize >,
    RangeToInclusive< usize >
}

macro_rules! impl_for_slice {
    ($($size:expr),*) => {
        $(
            impl ToIndices for [usize; $size] {
                fn to_indices( &self, _: usize ) -> Indices {
                    Indices::Disjoint { offset: 0, indices: &self[..] }
                }
            }
        )*
    };
}

impl_for_slice! {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64
}

pub(crate) struct ReusedIndices< 'a >( pub Indices< 'a > );

impl< 'a > ToIndices for ReusedIndices< 'a > {
    fn to_indices( &self, _: usize ) -> Indices {
        self.0.clone()
    }
}

#[cfg(test)]
fn check< T >( indices: T, expected: &[usize] ) where T: ToIndices {
    use std::panic;

    let container_length = 5;
    let indices = indices.to_indices( container_length );

    assert_eq!( indices.len(), expected.len() );

    {
        let mut indices = indices.clone();
        indices.next().unwrap();
        assert_eq!( indices.len(), expected.len() - 1 );
    }

    let collected: Vec< _ > = indices.clone().collect();
    assert_eq!( collected, expected );

    for (nth, expected_index) in expected.iter().cloned().enumerate() {
        assert_eq!( indices.get( nth ), expected_index );
    }

    let result = panic::catch_unwind( panic::AssertUnwindSafe( || {
        indices.get( expected.len() );
    }));

    assert!( result.is_err() );
}

#[test]
fn test_indices() {
    check( 3, &[3] );
    check( &[1, 2, 3][..], &[1, 2, 3] );
    check( &vec![1, 2, 3], &[1, 2, 3] );
    check( 1..4, &[1, 2, 3] );
    check( 1..=3, &[1, 2, 3] );
    check( ..4, &[0, 1, 2, 3] );
    check( ..=3, &[0, 1, 2, 3] );
    check( 1.., &[1, 2, 3, 4] );
    check( .., &[0, 1, 2, 3, 4] );
}
