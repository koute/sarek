use {
    std::{
        marker::PhantomData
    },
    crate::{
        core::{
            array::{
                ArrayRef,
                ToArrayRef,
                TypedArrayRef,
                TypedArrayMut
            },
            data_source::{
                DataSource
            },
            data_type::{
                DataType,
                Type,
                cast_slice_mut
            },
            indices::{
                ToIndices
            },
            shape::{
                Shape
            }
        }
    }
};

/// A slice converted into a `DataSource`.
pub struct SliceSource< T, C >
    where T: DataType,
          C: AsRef< [T] >
{
    length: usize,
    shape: Shape,
    container: C,
    phantom: PhantomData< T >
}

impl< T, C > SliceSource< T, C >
    where T: DataType,
          C: AsRef< [T] >
{
    /// Creates a new data source from a slice-like source.
    ///
    /// ```rust
    /// # use sarek::{Shape, SliceSource};
    /// let data: Vec< f32 > = vec![ 1.0, 2.0, 3.0, 4.0 ];
    /// let src = SliceSource::from( Shape::new_2d( 2, 2 ), data );
    ///
    /// let data: &[f32] = &[ 1.0, 2.0, 3.0, 4.0 ];
    /// let src = SliceSource::from( Shape::new_2d( 2, 2 ), data );
    ///
    /// let data: Box< [f32] > = vec![ 1.0, 2.0, 3.0, 4.0 ].into_boxed_slice();
    /// let src = SliceSource::from( Shape::new_2d( 2, 2 ), data );
    /// ```
    pub fn from( shape: Shape, container: C ) -> SliceSource< T, C > {
        let raw_length = container.as_ref().len();
        let length = if shape.is_zero() {
            0
        } else {
            if raw_length % shape.product() != 0 {
                panic!( "The length of the passed slice is not divisible by the shape! (length = {}, shape = {})", raw_length, shape );
            }

            raw_length / shape.product()
        };

        SliceSource {
            length,
            shape,
            container,
            phantom: PhantomData
        }
    }
}

impl< T, C > Clone for SliceSource< T, C >
    where T: DataType,
          C: AsRef< [T] > + Clone
{
    fn clone( &self ) -> Self {
        SliceSource {
            length: self.length,
            shape: self.shape.clone(),
            container: self.container.clone(),
            phantom: PhantomData
        }
    }
}

impl< T, C > DataSource for SliceSource< T, C >
    where T: DataType,
          C: AsRef< [T] >
{
    fn data_type( &self ) -> Type {
        T::TYPE
    }

    fn shape( &self ) -> Shape {
        self.shape.clone()
    }

    fn len( &self ) -> usize {
        self.length
    }

    fn gather_bytes_into< I >( &self, indices: I, output: &mut [u8] ) where I: ToIndices {
        let output = cast_slice_mut::< T >( output );
        let mut output = TypedArrayMut::new( self.shape.clone(), output );

        let input = self.container.as_ref();
        let input = TypedArrayRef::new( self.shape.clone(), input );

        output.gather_from( indices, &input );
    }
}

impl< T, C > ToArrayRef for SliceSource< T, C >
    where T: DataType,
          C: AsRef< [T] >
{
    fn to_array_ref( &self ) -> ArrayRef {
        TypedArrayRef::new( self.shape(), self.container.as_ref() ).into()
    }
}

#[test]
fn test_slice_source_basics() {
    let data: Vec< u32 > = vec![ 1, 2, 3, 4 ];
    let src = SliceSource::from( Shape::new_2d( 2, 2 ), data );

    assert_eq!( src.data_type(), Type::U32 );
    assert_eq!( src.shape(), Shape::new_2d( 2, 2 ) );
    assert_eq!( src.len(), 1 );
}

#[test]
fn test_slice_source_gather_into() {
    let data: Vec< u32 > = vec![ 1, 2, 3, 4 ];
    let src = SliceSource::from( Shape::new_2d( 1, 2 ), data );

    let mut output = [0_u32; 4];
    src.gather_into( 0..2, &mut output[ 0..4 ] );
    assert_eq!( &output, &[ 1, 2, 3, 4 ] );

    let mut output = [0_u32; 4];
    src.gather_into( 0..1, &mut output[ 0..2 ] );
    assert_eq!( &output, &[ 1, 2, 0, 0 ] );

    let mut output = [0_u32; 4];
    src.gather_into( 1..2, &mut output[ 0..2 ] );
    assert_eq!( &output, &[ 3, 4, 0, 0 ] );

    let mut output = [0_u32; 4];
    src.gather_into( &[0, 1][..], &mut output[ 0..4 ] );
    assert_eq!( &output, &[ 1, 2, 3, 4 ] );

    let mut output = [0_u32; 4];
    src.gather_into( &[1, 0][..], &mut output[ 0..4 ] );
    assert_eq!( &output, &[ 3, 4, 1, 2 ] );

    let mut output = [0_u32; 4];
    src.gather_into( &[0, 0][..], &mut output[ 0..4 ] );
    assert_eq!( &output, &[ 1, 2, 1, 2 ] );
}

#[test]
fn test_slice_source_panic_on_gather_bytes_into_buffer_with_size_indivisible_by_data_type_size() {
    use crate::core::data_type::as_byte_slice_mut;
    use crate::core::utils::assert_panic;

    let data: Vec< u32 > = vec![ 1, 2, 3, 4 ];
    let src = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    let mut output: [u32; 4] = [0; 4];
    let output = as_byte_slice_mut( &mut output );

    assert_panic( "The byte size of the slice (= 1) is not divisible by the byte size of u32 (= 4)", || {
        src.gather_bytes_into( 0..1, &mut output[ 0..1 ] );
    });
}

#[test]
fn test_slice_source_panic_on_gather_bytes_into_buffer_with_size_indivisible_by_minimum_alignment() {
    use crate::core::data_type::as_byte_slice_mut;
    use crate::core::utils::assert_panic;

    let data: Vec< u32 > = vec![ 1, 2, 3, 4 ];
    let src = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    let mut output: [u32; 4] = [0; 4];
    let output = as_byte_slice_mut( &mut output );

    assert_panic( "The slice's address is not divisibly by the minimum alignment of u32 (= 4)", || {
        src.gather_bytes_into( 0..1, &mut output[ 1..9 ] );
    });
}

#[test]
fn test_slice_source_panic_on_gather_bytes_into_buffer_with_size_indivisible_by_shape() {
    use crate::core::utils::assert_panic;

    let data: Vec< u8 > = vec![ 1, 2, 3, 4 ];
    let src = SliceSource::from( Shape::new_2d( 1, 2 ), data );
    let mut output: [u8; 4] = [0; 4];

    assert_panic( "Size of the slice (= 1) is not divisible by the shape (1, 2) of one element (= 2)", || {
        src.gather_bytes_into( 0..1, &mut output[ 0..1 ] );
    });
}
