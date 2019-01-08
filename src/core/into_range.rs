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
        usize
    }
};

pub trait IntoRange {
    fn into_range( self, container_length: usize ) -> Range< usize >;
}

impl IntoRange for usize {
    #[allow(clippy::range_plus_one)]
    fn into_range( self, _: usize ) -> Range< usize > {
        assert_ne!( self, usize::MAX );
        self..self + 1
    }
}

impl IntoRange for Range< usize > {
    fn into_range( self, _: usize ) -> Range< usize > {
        self
    }
}

impl IntoRange for RangeFrom< usize > {
    fn into_range( self, container_length: usize ) -> Range< usize > {
        assert!( self.start <= container_length );
        self.start..container_length
    }
}

impl IntoRange for RangeFull {
    fn into_range( self, container_length: usize ) -> Range< usize > {
        0..container_length
    }
}

impl IntoRange for RangeInclusive< usize > {
    #[allow(clippy::range_plus_one)]
    fn into_range( self, _: usize ) -> Range< usize > {
        assert_ne!( *self.end(), usize::MAX );
        *self.start()..*self.end() + 1
    }
}

impl IntoRange for RangeTo< usize > {
    fn into_range( self, _: usize ) -> Range< usize > {
        0..self.end
    }
}

impl IntoRange for RangeToInclusive< usize > {
    #[allow(clippy::range_plus_one)]
    fn into_range( self, _: usize ) -> Range< usize > {
        assert_ne!( self.end, usize::MAX );
        0..self.end + 1
    }
}
