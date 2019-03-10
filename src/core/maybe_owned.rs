use {
    std::{
        ops::{
            Deref
        }
    }
};

pub(crate) enum MaybeOwned< 'a, T > {
    Owned( T ),
    Borrowed( &'a T )
}

impl< 'a, T > Deref for MaybeOwned< 'a, T > {
    type Target = T;
    fn deref( &self ) -> &Self::Target {
        match *self {
            MaybeOwned::Owned( ref value ) => value,
            MaybeOwned::Borrowed( value ) => value
        }
    }
}
