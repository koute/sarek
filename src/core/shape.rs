use {
    std::{
        fmt,
        iter::{
            self,
            FromIterator,
            FusedIterator
        },
        slice
    },
    smallvec::SmallVec
};

/// A structure which represents an n-dimensional shape.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape( SmallVec< [usize; 4] > );

impl Shape {
    /// Constructs a new empty shape.
    #[inline]
    pub fn empty() -> Self {
        [].into_iter().cloned().collect()
    }

    /// Constructs a new one dimensional shape.
    #[inline]
    pub fn new_1d( x: usize ) -> Self {
        [x].into_iter().cloned().collect()
    }

    /// Constructs a new two dimensional shape.
    #[inline]
    pub fn new_2d( x: usize, y: usize ) -> Self {
        [x, y].into_iter().cloned().collect()
    }

    /// Constructs a new three dimensional shape.
    #[inline]
    pub fn new_3d( x: usize, y: usize, z: usize ) -> Self {
        [x, y, z].into_iter().cloned().collect()
    }

    /// Constructs a new four dimensional shape.
    #[inline]
    pub fn new_4d( x: usize, y: usize, z: usize, w: usize ) -> Self {
        [x, y, z, w].into_iter().cloned().collect()
    }

    /// Checks whenever the product of all of the dimensions is zero.
    #[inline]
    pub fn is_zero( &self ) -> bool {
        self.product() == 0
    }

    /// Returns the 1st dimension; will return `0` if the number of dimensions is zero.
    #[inline]
    pub fn x( &self ) -> usize {
        if self.dimension_count() == 0 {
            return 0;
        }

        self.0.get( 0 ).cloned().unwrap_or( 0 )
    }

    /// Returns the 2nd dimension; will return `0` if the number of dimensions is zero, `1` if it has less than two dimensions.
    #[inline]
    pub fn y( &self ) -> usize {
        if self.dimension_count() == 0 {
            return 0;
        }

        self.0.get( 1 ).cloned().unwrap_or( 1 )
    }

    /// Returns the 3rd dimension; will return `0` if the number of dimensions is zero, `1` if it has less than three dimensions.
    #[inline]
    pub fn z( &self ) -> usize {
        if self.dimension_count() == 0 {
            return 0;
        }

        self.0.get( 2 ).cloned().unwrap_or( 1 )
    }

    /// Returns the 4rd dimension; will return `0` if the number of dimensions is zero, `1` if it has less than four dimensions.
    #[inline]
    pub fn w( &self ) -> usize {
        if self.dimension_count() == 0 {
            return 0;
        }

        self.0.get( 3 ).cloned().unwrap_or( 1 )
    }

    /// Multiplies every dimension with each other and returns the result.
    ///
    /// ```rust
    /// # use sarek::Shape;
    /// let shape = Shape::new_2d( 2, 3 );
    /// assert_eq!( shape.product(), 6 );
    /// ```
    #[inline]
    pub fn product( &self ) -> usize {
        if self.dimension_count() == 0 {
            return 0;
        }

        self.iter().product()
    }

    /// Returns the number of dimensions.
    ///
    /// ```rust
    /// # use sarek::Shape;
    /// let shape = Shape::new_2d( 2, 3 );
    /// assert_eq!( shape.dimension_count(), 2 );
    /// ```
    #[inline]
    pub fn dimension_count( &self ) -> usize {
        self.iter().len()
    }

    /// Returns an iterator over the dimensions.
    ///
    /// ```rust
    /// # use sarek::Shape;
    /// let shape = Shape::new_2d( 2, 3 );
    /// let mut iter = shape.iter();
    /// assert_eq!( iter.next(), Some( 2 ) );
    /// assert_eq!( iter.next(), Some( 3 ) );
    /// assert_eq!( iter.next(), None );
    /// ```
    #[inline]
    pub fn iter< 'a >( &'a self ) -> impl ExactSizeIterator< Item = usize > + FusedIterator + 'a {
        self.0.iter().cloned()
    }

    /// Prepends an extra dimension to the shape.
    ///
    /// ```rust
    /// # use sarek::Shape;
    /// let shape = Shape::new_2d( 2, 3 );
    /// assert_eq!( shape.prepend( 1 ), Shape::new_3d( 1, 2, 3 ) );
    /// ```
    pub fn prepend( &self, value: usize ) -> Shape {
        iter::once( value ).chain( self.iter() ).collect()
    }

    /// Appends an extra dimension to the shape.
    ///
    /// ```rust
    /// # use sarek::Shape;
    /// let shape = Shape::new_2d( 1, 2 );
    /// assert_eq!( shape.append( 3 ), Shape::new_3d( 1, 2, 3 ) );
    /// ```
    pub fn append( &self, value: usize ) -> Shape {
        self.iter().chain( iter::once( value ) ).collect()
    }
}

impl fmt::Display for Shape {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "(" )?;
        let mut iter = self.iter().peekable();
        while let Some( value ) = iter.next() {
            write!( fmt, "{}", value )?;

            let is_last = iter.peek().is_none();
            if !is_last {
                write!( fmt, ", " )?;
            }
        }
        write!( fmt, ")" )?;

        Ok(())
    }
}

impl FromIterator< usize > for Shape {
    #[inline]
    fn from_iter< T >( iter: T ) -> Self where T: IntoIterator< Item = usize > {
        Shape( iter.into_iter().collect() )
    }
}

pub struct Iter< 'a >( slice::Iter< 'a, usize > );

impl< 'a > Iterator for Iter< 'a > {
    type Item = usize;

    #[inline]
    fn next( &mut self ) -> Option< Self::Item > {
        self.0.next().cloned()
    }
}

impl< 'a > ExactSizeIterator for Iter< 'a > {
    #[inline]
    fn len( &self ) -> usize {
        self.0.len()
    }
}

impl< 'a > FusedIterator for Iter< 'a > {}

impl< 'a > IntoIterator for &'a Shape {
    type Item = usize;
    type IntoIter = Iter< 'a >;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        Iter( self.0.iter() )
    }
}

impl< 'a > IntoIterator for &'a mut Shape {
    type Item = usize;
    type IntoIter = Iter< 'a >;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        Iter( self.0.iter() )
    }
}

impl From< usize > for Shape {
    #[inline]
    fn from( x: usize ) -> Self {
        Shape::new_1d( x )
    }
}

impl From< (usize,) > for Shape {
    #[inline]
    fn from( (x,): (usize,) ) -> Self {
        Shape::new_1d( x )
    }
}

impl From< (usize, usize) > for Shape {
    #[inline]
    fn from( (x, y): (usize, usize) ) -> Self {
        Shape::new_2d( x, y )
    }
}

impl From< (usize, usize, usize) > for Shape {
    #[inline]
    fn from( (x, y, z): (usize, usize, usize) ) -> Self {
        Shape::new_3d( x, y, z )
    }
}

impl From< (usize, usize, usize, usize) > for Shape {
    #[inline]
    fn from( (x, y, z, w): (usize, usize, usize, usize) ) -> Self {
        Shape::new_4d( x, y, z, w )
    }
}

#[test]
fn test_format_shape() {
    assert_eq!(
        format!( "{}", Shape::new_1d( 123 ) ),
        "(123)"
    );

    assert_eq!(
        format!( "{}", Shape::new_2d( 123, 456 ) ),
        "(123, 456)"
    );
}
