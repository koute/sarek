use {
    std::{
        fmt,
        mem,
        slice
    },
    crate::{
        core::{
            utils::{
                assert_can_be_upcast
            }
        }
    }
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[non_exhaustive]
#[repr(C)]
pub enum Type {
    F32,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8
}

impl Type {
    #[inline]
    pub(crate) fn byte_size( self ) -> usize {
        match self {
            Type::F32 => mem::size_of::< f32 >(),
            Type::I32 => mem::size_of::< i32 >(),
            Type::I16 => mem::size_of::< i16 >(),
            Type::I8 => mem::size_of::< i8 >(),
            Type::U32 => mem::size_of::< u32 >(),
            Type::U16 => mem::size_of::< u16 >(),
            Type::U8 => mem::size_of::< u8 >()
        }
    }

    #[inline]
    pub(crate) fn align_of( self ) -> usize {
        match self {
            Type::F32 => mem::align_of::< f32 >(),
            Type::I32 => mem::align_of::< i32 >(),
            Type::I16 => mem::align_of::< i16 >(),
            Type::I8 => mem::align_of::< i8 >(),
            Type::U32 => mem::align_of::< u32 >(),
            Type::U16 => mem::align_of::< u16 >(),
            Type::U8 => mem::align_of::< u8 >()
        }
    }
}

impl fmt::Display for Type {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        let name = match self {
            Type::F32 => "f32",
            Type::I32 => "i32",
            Type::I16 => "i16",
            Type::I8 => "i8",
            Type::U32 => "u32",
            Type::U16 => "u16",
            Type::U8 => "u8"
        };
        write!( fmt, "{}", name )
    }
}

pub trait DataType: Copy + Clone + Default + Sync + Send {
    const TYPE: Type;
}

impl DataType for f32 {
    const TYPE: Type = Type::F32;
}

impl DataType for i32 {
    const TYPE: Type = Type::I32;
}

impl DataType for i16 {
    const TYPE: Type = Type::I16;
}

impl DataType for i8 {
    const TYPE: Type = Type::I8;
}

impl DataType for u32 {
    const TYPE: Type = Type::U32;
}

impl DataType for u16 {
    const TYPE: Type = Type::U16;
}

impl DataType for u8 {
    const TYPE: Type = Type::U8;
}

/// Safely casts a slice into a byte array.
///
/// ```ignore
/// let slice_u32: &[u32] = &[0xFFFFFFFF, 0xAAAAAAAA];
/// let slice_u8: &[u8] = as_byte_slice( slice_u32 );
///
/// assert_eq!( slice_u8.len(), 8 );
/// assert_eq!( slice_u8, &[ 0xFF, 0xFF, 0xFF, 0xFF, 0xAA, 0xAA, 0xAA, 0xAA ] );
/// assert_eq!( slice_u8.as_ptr() as usize, slice_u32.as_ptr() as usize );
/// ```
pub fn as_byte_slice< T: DataType >( slice: &[T] ) -> &[u8] {
    assert!( mem::align_of::< u8 >() <= mem::align_of::< T >() );
    unsafe {
        slice::from_raw_parts( slice.as_ptr() as *const u8, slice.len() * mem::size_of::< T >() )
    }
}

pub fn as_byte_slice_mut< T: DataType >( slice: &mut [T] ) -> &mut [u8] {
    assert!( mem::align_of::< u8 >() <= mem::align_of::< T >() );
    unsafe {
        slice::from_raw_parts_mut( slice.as_mut_ptr() as *mut u8, slice.len() * mem::size_of::< T >() )
    }
}

/// Safely casts a byte slice into a slice of a wider type.
///
/// Will panic if the slice's length is not a multiple of the size of `T`
/// or if it doesn't meet the minimum alignment requirements of `T`.
pub fn cast_slice< T: DataType >( slice: &[u8] ) -> &[T] {
    assert_can_be_upcast( T::TYPE, slice );
    unsafe {
        slice::from_raw_parts( slice.as_ptr() as *const T, slice.len() / mem::size_of::< T >() )
    }
}

/// Safely casts a mutable byte slice into a mutable slice of a wider type.
///
/// Will panic if the slice's length is not a multiple of the size of `T`
/// or if it doesn't meet the minimum alignment requirements of `T`.
pub fn cast_slice_mut< T: DataType >( slice: &mut [u8] ) -> &mut [T] {
    assert_can_be_upcast( T::TYPE, slice );
    unsafe {
        slice::from_raw_parts_mut( slice.as_mut_ptr() as *mut T, slice.len() / mem::size_of::< T >() )
    }
}

#[test]
fn test_cast_slice() {
    let original_slice: &[u32] = &[0xFFFFFFFF, 0xAAAAAAAA];
    let slice: &[u8] = as_byte_slice( original_slice );
    let slice: &[u32] = cast_slice( slice );

    assert_eq!( slice.len(), 2 );
    assert_eq!( slice, original_slice );
}

#[test]
#[should_panic]
fn test_cast_slice_panics_on_slice_with_non_divisible_length() {
    let original_slice: &[u32] = &[0xFFFFFFFF, 0xAAAAAAAA];
    let slice: &[u8] = as_byte_slice( original_slice );
    let _: &[u32] = cast_slice( &slice[ 0..slice.len() - 1 ] );
}

#[test]
#[should_panic]
fn test_cast_slice_panics_on_non_aligned_slice() {
    let slice: &[u8] = &[0, 0, 0, 0, 0];
    let slice = &slice[ 1.. ];
    let _: &[u32] = cast_slice( slice );
}
