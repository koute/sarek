use {
    std::{
        iter::{
            FusedIterator
        },
        ops::{
            Index
        },
        mem
    }
};

pub enum SmallIter< T > {
    Empty,
    One( T ),
    Two( T, T )
}

impl< T > Iterator for SmallIter< T > {
    type Item = T;
    fn next( &mut self ) -> Option< Self::Item > {
        let mut tmp = SmallIter::Empty;
        mem::swap( self, &mut tmp );
        match tmp {
            SmallIter::Empty => None,
            SmallIter::One( value ) => Some( value ),
            SmallIter::Two( value, remaining ) => {
                *self = SmallIter::One( remaining );
                Some( value )
            }
        }
    }

    fn size_hint( &self ) -> (usize, Option< usize >) {
        let length = match *self {
            SmallIter::Empty => 0,
            SmallIter::One( .. ) => 1,
            SmallIter::Two( .. ) => 2
        };

        (length, Some( length ))
    }
}

impl< T > ExactSizeIterator for SmallIter< T > {}
impl< T > FusedIterator for SmallIter< T > {}

impl< T > Index< usize > for SmallIter< T > {
    type Output = T;
    fn index( &self, index: usize ) -> &Self::Output {
        match *self {
            SmallIter::Empty => panic!(),
            SmallIter::One( ref value ) => {
                if index == 0 {
                    value
                } else {
                    panic!( "index out of bounds: the len is 1 but the index is {}", index );
                }
            },
            SmallIter::Two( ref value_1, ref value_2 ) => {
                if index == 0 {
                    value_1
                } else if index == 1 {
                    value_2
                } else {
                    panic!( "index out of bounds: the len is 2 but the index is {}", index );
                }
            }
        }
    }
}
