use {
    std::{
        fmt,
        intrinsics::{
            type_name
        }
    },
    crate::{
        core::{
            data_type::{
                Type
            }
        }
    }
};

#[cfg(test)]
pub fn assert_panic< F >( expected_message_chunk: &str, callback: F ) where F: FnOnce() {
    use std::panic;

    let result = panic::catch_unwind( panic::AssertUnwindSafe( move || {
        callback();
    }));

    match result {
        Ok(()) => {
            panic!( "Expected a panic!" );
        },
        Err( error ) => {
            let message = if let Some( message ) = error.downcast_ref::< &str >() {
                message
            } else if let Some( message ) = error.downcast_ref::< String >() {
                message.as_str()
            } else {
                panic!( "Panic with an unknown type inside!" );
            };
            assert!(
                message.contains( expected_message_chunk ),
                "Unexpected panic message: {}",
                message
            );
        }
    }
}

pub fn assert_can_be_upcast( data_type: Type, slice: &[u8] ) {
    assert_eq!(
        slice.as_ptr() as usize % data_type.align_of(),
        0,
        "The slice's address is not divisibly by the minimum alignment of {} (= {})",
        data_type,
        data_type.align_of()
    );

    assert_eq!(
        slice.len() % data_type.byte_size(),
        0,
        "The byte size of the slice (= {}) is not divisible by the byte size of {} (= {})",
        slice.len(),
        data_type,
        data_type.byte_size()
    );
}

pub struct SliceDebug< 'a, T >( pub &'a [T] );
impl< 'a, T > fmt::Debug for SliceDebug< 'a, T > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        let name = unsafe { type_name::< T >() };
        write!( fmt, "[{}; {}]", name, self.0.len() )
    }
}

#[test]
fn test_slice_debug() {
    let string = format!( "{:?}", SliceDebug( &[1.0_f32] ) );
    assert_eq!( string, "[f32; 1]" );
}
