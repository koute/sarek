use {
    std::{
        fmt,
        mem,
        ops::{
            Deref,
            Index,
            IndexMut,
            Range,
            RangeFrom,
            RangeFull,
            RangeInclusive,
            RangeTo,
            RangeToInclusive
        }
    },
    crate::{
        core::{
            data_type::{
                DataType,
                Type,
                as_byte_slice,
                as_byte_slice_mut,
                cast_slice,
                cast_slice_mut
            },
            indices::{
                Indices,
                ToIndices
            },
            into_range::{
                IntoRange
            },
            shape::{
                Shape
            },
            type_cast_error::{
                TypeCastError
            },
            utils::{
                assert_can_be_upcast
            }
        }
    }
};

fn assert_length_is_divisible_by_shape( shape: &Shape, slice_length: usize ) {
    assert_eq!(
        slice_length % shape.product(),
        0,
        "Size of the slice (= {}) is not divisible by the shape {} of one element (= {})",
        slice_length,
        shape,
        shape.product()
    );
}

fn assert_index_count_is_equal_to_slice_capacity( indices: &Indices, element_count: usize ) {
    assert_eq!(
        element_count,
        indices.len(),
        "Tried to gather {} element(s) into an array which is supposed to hold {}",
        indices.len(),
        element_count
    );
}

fn assert_shapes_are_equal( input_shape: &Shape, output_shape: &Shape ) {
    assert_eq!(
        input_shape,
        output_shape,
        "Tried to gather elements from an array with a shape of {} into an array with a shape of {}",
        input_shape,
        output_shape
    );
}

pub trait ToArrayRef {
    /// Acquires an `ArrayRef` from this object.
    fn to_array_ref( &self ) -> ArrayRef;

    /// A convenience method which calls `.to_array_ref().to_typed()`.
    fn to_typed_array_ref< T >( &self ) -> Result< TypedArrayRef< T >, TypeCastError< () > > where T: DataType {
        self.to_array_ref().to_typed()
    }

    /// A convenience method to cast the object into a typed slice.
    fn to_slice< T >( &self ) -> Result< &[T], TypeCastError< () > > where T: DataType {
        let array = self.to_array_ref().to_typed::< T >()?;
        Ok( array.as_slice() )
    }
}

impl< 'a, T > ToArrayRef for &'a T where &'a T: Into< ArrayRef< 'a > > + 'a {
    #[inline]
    fn to_array_ref( &self ) -> ArrayRef {
        (*self).into()
    }
}

/// An immutable multidimensional byte slice.
#[repr(C)]
#[derive(Clone)]
pub struct ArrayRef< 'a > {
    slice: &'a [u8],
    shape: Shape,
    length: usize,
    data_type: Type
}

/// A mutable multidimensional byte slice.
#[repr(C)]
pub struct ArrayMut< 'a > {
    slice: &'a mut [u8],
    shape: Shape,
    length: usize,
    data_type: Type
}

impl< 'a > Deref for ArrayMut< 'a > {
    type Target = ArrayRef< 'a >;
    fn deref( &self ) -> &Self::Target {
        assert_eq!( mem::size_of::< ArrayMut >(), mem::size_of::< ArrayRef >() );
        let slice: &ArrayMut = self;
        let slice: &ArrayRef = unsafe { &*(slice as *const ArrayMut as *const ArrayRef) };
        slice
    }
}

struct AsDisplay< T: fmt::Display >( T );

impl< T > fmt::Display for AsDisplay< T > where T: fmt::Display {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        self.0.fmt( fmt )
    }
}

impl< T > fmt::Debug for AsDisplay< T > where T: fmt::Display {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        self.0.fmt( fmt )
    }
}

impl< 'a > fmt::Debug for ArrayRef< 'a > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "ArrayRef" )
            .field( "shape", &AsDisplay( &self.shape ) )
            .field( "length", &self.length )
            .field( "data_type", &AsDisplay( self.data_type ) )
            .finish()
    }
}

impl< 'a > ArrayRef< 'a > {
    /// Creates a new shaped byte slice.
    ///
    /// The raw `slice` passed here **must** fulfil the following
    /// requirements or else the code will panic:
    ///    * `slice`'s length must be divisible by the size of the underlying data type
    ///    * `slice`'s length must be divisible by the passed shape
    ///    * `slice` must be properly aligned for the underlying data type
    pub fn new( shape: Shape, data_type: Type, slice: &'a [u8] ) -> Self {
        assert_can_be_upcast( data_type, slice );
        assert_length_is_divisible_by_shape( &shape, slice.len() / data_type.byte_size() );

        let element_size = shape.product() * data_type.byte_size();
        let length = slice.len() / element_size;
        ArrayRef { slice, shape, length, data_type }
    }

    pub fn data_type( &self ) -> Type {
        self.data_type
    }

    pub fn len( &self ) -> usize {
        self.length
    }

    pub fn is_empty( &self ) -> bool {
        self.len() == 0
    }

    pub fn as_slice( &self ) -> &'a [u8] {
        self.slice
    }

    pub fn to_typed< T >( &self ) -> Result< TypedArrayRef< 'a, T >, TypeCastError< () > > where T: DataType {
        if self.data_type() != T::TYPE {
            return Err( TypeCastError {
                source: "an array",
                target: "a statically typed array",
                source_ty: self.data_type().into(),
                target_ty: T::TYPE,
                obj: ()
            });
        }

        let slice = cast_slice( self.slice );
        Ok( TypedArrayRef {
            slice,
            shape: self.shape.clone(),
            length: self.length
        })
    }

    fn element_size( &self ) -> usize {
        self.shape.product() * self.data_type.byte_size()
    }
}

impl< 'a > ArrayMut< 'a > {
    /// Creates a new mutable shaped byte slice.
    ///
    /// This has the same requirements as `ArrayRef::new` as far
    /// as panicking goes.
    pub fn new( shape: Shape, data_type: Type, slice: &'a mut [u8] ) -> Self {
        assert_can_be_upcast( data_type, slice );
        assert_length_is_divisible_by_shape( &shape, slice.len() / data_type.byte_size() );

        let element_size = shape.product() * data_type.byte_size();
        let length = slice.len() / element_size;
        ArrayMut { slice, shape, length, data_type }
    }

    pub fn as_mut_slice( &mut self ) -> &mut [u8] {
        self.slice
    }

    pub fn gather_from< I >( &mut self, indices: I, input: &ArrayRef ) where I: ToIndices {
        assert_eq!(
            self.data_type(),
            input.data_type(),
            "Slices have different data types (input = {}, output = {})",
            input.data_type(),
            self.data_type()
        );

        let indices = indices.to_indices( input.length );
        assert_index_count_is_equal_to_slice_capacity( &indices, self.len() );
        assert_shapes_are_equal( &input.shape, &self.shape );

        match indices {
            Indices::Continuous { range } => {
                self.as_mut_slice().copy_from_slice( &input[ range ] );
            },
            Indices::Disjoint { offset, indices } => {
                for (out_index, index) in indices.into_iter().cloned().map( |index| offset + index ).enumerate() {
                    self[ out_index ].copy_from_slice( &input[ index ] );
                }
            }
        }
    }

    pub fn into_typed_mut< T >( self ) -> Result< TypedArrayMut< 'a, T >, TypeCastError< Self > > where T: DataType {
        if self.data_type() != T::TYPE {
            return Err( TypeCastError {
                source: "an array",
                target: "a statically typed array",
                source_ty: self.data_type().into(),
                target_ty: T::TYPE,
                obj: self
            });
        }

        let slice = cast_slice_mut( self.slice );
        Ok( TypedArrayMut {
            slice,
            shape: self.shape,
            length: self.length
        })
    }
}

/// An immutable multidimensional slice.
#[repr(C)] // Since transmuting structs without #[repr(C)] is UB.
#[derive(Clone)]
pub struct TypedArrayRef< 'a, T > where T: DataType {
    slice: &'a [T],
    shape: Shape,
    length: usize
}

/// A mutable multidimensional slice.
#[repr(C)]
pub struct TypedArrayMut< 'a, T > where T: DataType {
    slice: &'a mut [T],
    shape: Shape,
    length: usize
}

impl< 'a, T > Deref for TypedArrayMut< 'a, T > where T: DataType {
    type Target = TypedArrayRef< 'a, T >;
    fn deref( &self ) -> &Self::Target {
        assert_eq!( mem::size_of::< TypedArrayMut< T > >(), mem::size_of::< TypedArrayRef< T > >() );
        let slice: &TypedArrayMut< T > = self;
        let slice: &TypedArrayRef< T > = unsafe { &*(slice as *const TypedArrayMut<T> as *const TypedArrayRef<T>) };
        slice
    }
}

impl< 'a, T > TypedArrayRef< 'a, T > where T: DataType {
    pub fn new( shape: Shape, slice: &'a [T] ) -> Self {
        assert_length_is_divisible_by_shape( &shape, slice.len() );

        let element_size = shape.product();
        let length = slice.len() / element_size;
        TypedArrayRef { slice, shape, length }
    }

    pub fn data_type( &self ) -> Type {
        T::TYPE
    }

    pub fn len( &self ) -> usize {
        self.length
    }

    pub fn is_empty( &self ) -> bool {
        self.len() == 0
    }

    pub fn as_slice( &self ) -> &'a [T] {
        self.slice
    }

    fn element_size( &self ) -> usize {
        self.shape.product()
    }
}

impl< 'a, T > TypedArrayMut< 'a, T > where T: DataType {
    pub fn new( shape: Shape, slice: &'a mut [T] ) -> Self {
        assert_length_is_divisible_by_shape( &shape, slice.len() );

        let element_size = shape.product();
        let length = slice.len() / element_size;
        TypedArrayMut { slice, shape, length }
    }

    pub fn as_mut_slice( &mut self ) -> &mut [T] {
        self.slice
    }

    pub fn gather_from< I >( &mut self, indices: I, input: &TypedArrayRef< T > ) where I: ToIndices {
        let indices = indices.to_indices( input.length );
        assert_index_count_is_equal_to_slice_capacity( &indices, self.len() );
        assert_shapes_are_equal( &input.shape, &self.shape );

        match indices {
            Indices::Continuous { range } => {
                self.as_mut_slice().copy_from_slice( &input[ range ] );
            },
            Indices::Disjoint { offset, indices } => {
                for (out_index, index) in indices.into_iter().cloned().map( |index| offset + index ).enumerate() {
                    self[ out_index ].copy_from_slice( &input[ index ] );
                }
            }
        }
    }

    pub fn into_array_mut( self ) -> ArrayMut< 'a > {
        self.into()
    }
}

impl< 'a > From< ArrayMut< 'a > > for ArrayRef< 'a > {
    fn from( array: ArrayMut< 'a > ) -> Self {;
        array.deref().clone()
    }
}

impl< 'a, T > From< TypedArrayRef< 'a, T > > for ArrayRef< 'a > where T: DataType {
    fn from( array: TypedArrayRef< 'a, T > ) -> Self {
        ArrayRef {
            slice: as_byte_slice( array.slice ),
            shape: array.shape.clone(),
            length: array.length,
            data_type: array.data_type()
        }
    }
}

impl< 'a, T > From< TypedArrayMut< 'a, T > > for ArrayRef< 'a > where T: DataType {
    fn from( array: TypedArrayMut< 'a, T > ) -> Self {
        array.deref().clone().into()
    }
}

impl< 'a, T > From< TypedArrayMut< 'a, T > > for ArrayMut< 'a > where T: DataType {
    fn from( array: TypedArrayMut< 'a, T > ) -> Self {
        ArrayMut {
            data_type: array.data_type(),
            slice: as_byte_slice_mut( array.slice ),
            shape: array.shape.clone(),
            length: array.length
        }
    }
}

impl< 'a > From< &'a ArrayRef< 'a > > for ArrayRef< 'a > {
    fn from( array: &'a ArrayRef< 'a > ) -> Self {;
        (&*array).clone()
    }
}

impl< 'a > From< &'a ArrayMut< 'a > > for ArrayRef< 'a > {
    fn from( array: &'a ArrayMut< 'a > ) -> Self {;
        array.deref().into()
    }
}

impl< 'a, T > From< &'a TypedArrayRef< 'a, T > > for ArrayRef< 'a > where T: DataType {
    fn from( array: &'a TypedArrayRef< 'a, T > ) -> Self {
        array.clone().into()
    }
}

impl< 'a, T > From< &'a TypedArrayMut< 'a, T > > for ArrayRef< 'a > where T: DataType {
    fn from( array: &'a TypedArrayMut< 'a, T > ) -> Self {
        array.deref().into()
    }
}

impl< 'a > ToArrayRef for ArrayRef< 'a > {
    fn to_array_ref( &self ) -> ArrayRef {
        self.into()
    }
}

impl< 'a > ToArrayRef for ArrayMut< 'a > {
    fn to_array_ref( &self ) -> ArrayRef {
        self.into()
    }
}

impl< 'a, T > ToArrayRef for TypedArrayRef< 'a, T > where T: DataType {
    fn to_array_ref( &self ) -> ArrayRef {
        self.into()
    }
}

impl< 'a, T > ToArrayRef for TypedArrayMut< 'a, T > where T: DataType {
    fn to_array_ref( &self ) -> ArrayRef {
        self.into()
    }
}

macro_rules! each_range_ty {
    ($($macro:ident! { $($arg:tt)* })*) => {
        $(
            $macro! { $($arg)* usize }
            $macro! { $($arg)* Range< usize > }
            $macro! { $($arg)* RangeFrom< usize > }
            $macro! { $($arg)* RangeFull }
            $macro! { $($arg)* RangeInclusive< usize > }
            $macro! { $($arg)* RangeTo< usize > }
            $macro! { $($arg)* RangeToInclusive< usize > }
        )*
    }
}

macro_rules! impl_index {
    (($($impl_arg:tt)*) ($target_ty:ident) ($self_ty:ty) $index_ty:ty) => {
        impl< $($impl_arg)* > Index< $index_ty > for $self_ty {
            type Output = [$target_ty];
            fn index( &self, index: $index_ty ) -> &Self::Output {
                let index = index.into_range( self.length );
                let element_size = self.element_size();
                let range = index.start * element_size..index.end * element_size;
                &self.as_slice()[ range ]
            }
        }
    }
}

macro_rules! impl_index_mut {
    (($($impl_arg:tt)*) ($target_ty:ident) ($self_ty:ty) $index_ty:ty) => {
        impl_index! {
            ($($impl_arg)*) ($target_ty) ($self_ty) $index_ty
        }

        impl< $($impl_arg)* > IndexMut< $index_ty > for $self_ty {
            fn index_mut( &mut self, index: $index_ty ) -> &mut Self::Output {
                let index = index.into_range( self.length );
                let element_size = self.element_size();
                let range = index.start * element_size..index.end * element_size;
                &mut self.as_mut_slice()[ range ]
            }
        }
    }
}

each_range_ty! {
    impl_index! { ('a, T: DataType) (T) (TypedArrayRef< 'a, T >) }
    impl_index! { ('a) (u8) (ArrayRef< 'a >) }
    impl_index_mut! { ('a, T: DataType) (T) (TypedArrayMut< 'a, T >) }
    impl_index_mut! { ('a) (u8) (ArrayMut< 'a >) }
}

#[test]
fn test_typed_array_ref_panic_with_buffer_size_indivisible_by_shape() {
    use crate::core::utils::assert_panic;
    assert_panic( "Size of the slice (= 1) is not divisible by the shape (1, 2) of one element (= 2)", || {
        TypedArrayRef::new( Shape::new_2d( 1, 2 ), &[0_u32] );
    });
}

#[test]
fn test_typed_array_ref_mut_panic_with_buffer_size_indivisible_by_shape() {
    use crate::core::utils::assert_panic;
    assert_panic( "Size of the slice (= 1) is not divisible by the shape (1, 2) of one element (= 2)", || {
        TypedArrayMut::new( Shape::new_2d( 1, 2 ), &mut [0_u32] );
    });
}

#[test]
fn test_typed_array_ref_gather_from() {
    let input = TypedArrayRef::new( Shape::new_2d( 1, 2 ), &[ 1, 2, 3, 4 ] );
    assert_eq!( input.len(), 2 );

    let mut output = [0; 4];
    let mut output = TypedArrayMut::new( Shape::new_2d( 1, 2 ), &mut output );
    assert_eq!( output.len(), 2 );

    output.gather_from( &[1, 0][..], &input );
    assert_eq!( output.as_slice(), &[ 3, 4, 1, 2 ] );
}

#[test]
fn test_typed_array_ref_gather_from_differently_sized_array() {
    use crate::core::utils::assert_panic;

    let input = TypedArrayRef::new( Shape::new_2d( 1, 2 ), &[ 1, 2, 3, 4 ] );
    let mut output = [0; 4];
    let mut output = TypedArrayMut::new( Shape::new_2d( 1, 2 ), &mut output );
    assert_panic( "Tried to gather 1 element(s) into an array which is supposed to hold 2", || {
        output.gather_from( 0, &input );
    });
}

#[test]
fn test_typed_array_ref_gather_from_differently_shaped_array() {
    use crate::core::utils::assert_panic;

    let input = TypedArrayRef::new( Shape::new_2d( 1, 2 ), &[ 1, 2, 3, 4 ] );
    let mut output = [0; 4];
    let mut output = TypedArrayMut::new( Shape::new_2d( 2, 1 ), &mut output );
    assert_panic( "Tried to gather elements from an array with a shape of (1, 2) into an array with a shape of (2, 1)", || {
        output.gather_from( 0..2, &input );
    });
}

#[test]
fn test_typed_array_ref_to_array_ref() {
    let mut buffer: [u32; 4] = [1, 2, 3, 4];
    let slice = TypedArrayMut::new( Shape::new_2d( 1, 2 ), &mut buffer );
    assert_eq!( slice.to_array_ref().data_type(), Type::U32 );
    assert_eq!( slice.to_array_ref().len(), 2 );
    assert_eq!( cast_slice::< u32 >( slice.to_array_ref().as_slice() ), slice.as_slice() );

    let slice = slice.into_array_mut();
    assert_eq!( slice.data_type(), Type::U32 );
    assert_eq!( slice.len(), 2 );
}

#[test]
fn test_array_ref_panic_with_buffer_size_indivisible_by_shape() {
    use crate::core::utils::assert_panic;
    assert_panic( "Size of the slice (= 1) is not divisible by the shape (1, 2) of one element (= 2)", || {
        ArrayRef::new( Shape::new_2d( 1, 2 ), Type::U8, &[0_u8] );
    });
}

#[test]
fn test_array_ref_mut_panic_with_buffer_size_indivisible_by_shape() {
    use crate::core::utils::assert_panic;
    assert_panic( "Size of the slice (= 1) is not divisible by the shape (1, 2) of one element (= 2)", || {
        ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U8, &mut [0_u8] );
    });
}

#[test]
fn test_array_ref_gather_from() {
    use crate::core::data_type::as_byte_slice;

    let input: &[u32] = &[ 1, 2, 3, 4 ];
    let input = as_byte_slice( input );
    let input = ArrayRef::new( Shape::new_2d( 1, 2 ), Type::U32, input );
    assert_eq!( input.len(), 2 );

    let mut output = [0; 16];
    let mut output = ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U32, &mut output );
    assert_eq!( output.len(), 2 );

    output.gather_from( &[1, 0][..], &input );
    let output: &[u32] = cast_slice( output.as_slice() );
    assert_eq!( output, &[ 3, 4, 1, 2 ] );
}

#[test]
fn test_array_ref_gather_from_with_an_array_of_different_type() {
    use crate::core::data_type::as_byte_slice;
    use crate::core::utils::assert_panic;

    let input: &[u32] = &[ 1, 2, 3, 4 ];
    let input = as_byte_slice( input );
    let input = ArrayRef::new( Shape::new_2d( 1, 2 ), Type::U32, input );
    let mut output = [0; 16];
    let mut output = ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U8, &mut output );
    assert_panic( "Slices have different data types (input = u32, output = u8)", || {
        output.gather_from( &[1, 0][..], &input );
    });
}

#[test]
fn test_array_ref_gather_from_differently_sized_array() {
    use crate::core::utils::assert_panic;

    let input = ArrayRef::new( Shape::new_2d( 1, 2 ), Type::U8, &[ 1, 2, 3, 4 ] );
    let mut output = [0; 4];
    let mut output = ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U8, &mut output );
    assert_panic( "Tried to gather 1 element(s) into an array which is supposed to hold 2", || {
        output.gather_from( 0, &input );
    });
}

#[test]
fn test_array_ref_gather_from_differently_shaped_array() {
    use crate::core::utils::assert_panic;

    let input = ArrayRef::new( Shape::new_2d( 1, 2 ), Type::U8, &[ 1, 2, 3, 4 ] );
    let mut output = [0; 4];
    let mut output = ArrayMut::new( Shape::new_2d( 2, 1 ), Type::U8, &mut output );
    assert_panic( "Tried to gather elements from an array with a shape of (1, 2) into an array with a shape of (2, 1)", || {
        output.gather_from( 0..2, &input );
    });
}

#[test]
fn test_array_ref_to_typed_array_of_different_type() {
    let mut slice = [ 1, 2, 3, 4 ];
    let slice = ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U8, &mut slice );

    assert_eq!(
        slice.to_typed::< u32 >().err().unwrap().to_string(),
        "tried to cast an array of type 'u8' into a statically typed array of type 'u32'"
    );

    assert_eq!(
        slice.into_typed_mut::< u32 >().err().unwrap().to_string(),
        "tried to cast an array of type 'u8' into a statically typed array of type 'u32'"
    );
}

#[test]
fn test_array_ref_to_typed() {
    let mut slice = [ 1, 2, 3, 4 ];
    let slice = ArrayMut::new( Shape::new_2d( 1, 2 ), Type::U8, &mut slice );
    assert_eq!( slice.to_typed::< u8 >().unwrap().data_type(), Type::U8 );
    assert_eq!( slice.to_typed::< u8 >().unwrap().len(), 2 );

    let slice = slice.into_typed_mut::< u8 >().unwrap();
    assert_eq!( slice.data_type(), Type::U8 );
    assert_eq!( slice.len(), 2 );
}
