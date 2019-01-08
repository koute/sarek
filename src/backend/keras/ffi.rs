use {
    pyo3::{
        ffi::PyObject
    }
};

#[repr(C)]
#[derive(Debug)]
pub struct PyArrayDesc {
    pub py_obj: PyObject,
    pub typeobj: *const libc::c_void,
    pub kind: libc::c_char,
    pub ty: libc::c_char,
    pub byteorder: libc::c_char,
    pub flags: libc::c_char,
    // ...this is incomplete
}

#[repr(C)]
#[derive(Debug)]
pub struct PyArrayObject {
    pub py_obj: PyObject,
    pub data: *mut u8,
    pub nd: libc::c_int,
    pub dims: *mut libc::intptr_t,
    pub strides: *mut libc::intptr_t,
    pub base: *mut PyObject,
    pub descr: *mut PyArrayDesc,
    pub flags: libc::c_int,
    pub weakreflist: *mut PyObject
}

pub const NPY_ARRAY_OWNDATA: libc::c_int = 0x0004;
pub const NPY_TRACE_DOMAIN: libc::c_uint = 389047;
pub const NPY_ITEM_REFCOUNT: libc::c_char = 0x01;

extern "C" {
    pub fn PyTraceMalloc_Untrack( domain: libc::c_uint, ptr: libc::uintptr_t ) -> libc::c_int;
}
