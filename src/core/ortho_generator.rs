use {
    rand::{
        prelude::*
    },
    std::{
        iter::{
            FusedIterator
        },
        marker::{
            PhantomData
        }
    },
    packed_simd::{
        f32x2,
        f32x4,
        shuffle
    }
};

const ENABLE_BOUND_CHECKS: bool = true;

struct SquareSlice< 'a > {
    pointer: *mut f32,
    pointer_end: *mut f32,
    size: usize,
    stride: usize,
    phantom: PhantomData< &'a mut [f32] >
}

unsafe impl< 'a > Send for SquareSlice< 'a > {}

impl< 'a > SquareSlice< 'a > {
    #[inline]
    fn new( slice: &'a mut [f32], size: usize ) -> SquareSlice< 'a > {
        assert!( size.is_power_of_two() );
        assert_eq!( slice.len(), size * size );

        unsafe {
            SquareSlice {
                pointer: slice.as_mut_ptr(),
                pointer_end: slice.as_mut_ptr().add( slice.len() ),
                size,
                stride: size,
                phantom: PhantomData
            }
        }
    }

    #[inline]
    fn size( &self ) -> usize {
        self.size
    }

    #[allow(dead_code)]
    #[inline]
    fn set( &mut self, y: usize, x: usize, value: f32 ) {
        if ENABLE_BOUND_CHECKS {
            assert!( x < self.size );
            assert!( y < self.size );
        }

        unsafe {
            let pointer = self.pointer.add( y * self.stride + x );
            if ENABLE_BOUND_CHECKS {
                assert!( pointer >= self.pointer );
                assert!( pointer < self.pointer_end );
            }

            *pointer = value;
        }
    }

    #[inline]
    fn row( &mut self, y: usize ) -> &mut [f32] {
        if ENABLE_BOUND_CHECKS {
            assert!( y < self.size );
        }

        unsafe {
            let pointer = self.pointer.add( y * self.stride );
            if ENABLE_BOUND_CHECKS {
                assert!( pointer >= self.pointer );
                assert!( pointer < self.pointer_end );
                assert!( pointer.add( self.size ) <= self.pointer_end );
            }

            std::slice::from_raw_parts_mut( pointer, self.size )
        }
    }

    #[inline]
    fn split_in_four( &mut self ) -> (SquareSlice, SquareSlice, SquareSlice, SquareSlice) {
        let stride = self.stride;
        let half_size = self.size / 2;

        unsafe {
            let ul = SquareSlice {
                pointer: self.pointer,
                pointer_end: self.pointer_end,
                size: half_size,
                stride,
                phantom: PhantomData
            };

            let ur = SquareSlice {
                pointer: self.pointer.add( half_size ),
                pointer_end: self.pointer_end,
                size: half_size,
                stride,
                phantom: PhantomData
            };

            let ll = SquareSlice {
                pointer: self.pointer.add( stride * half_size ),
                pointer_end: self.pointer_end,
                size: half_size,
                stride,
                phantom: PhantomData
            };

            let lr = SquareSlice {
                pointer: self.pointer.add( (stride + 1) * half_size ),
                pointer_end: self.pointer_end,
                size: half_size,
                stride,
                phantom: PhantomData
            };

            (ul, ur, ll, lr)
        }
    }
}

#[test]
fn test_square_slice_set() {
    let data = &mut [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ];

    let mut slice = SquareSlice::new( &mut data[..], 4 );
    slice.set( 0, 0, 0.1 );
    slice.set( 0, 1, 0.2 );
    slice.set( 0, 2, 0.3 );
    slice.set( 0, 3, 0.4 );
    slice.set( 1, 0, 0.5 );
    slice.set( 1, 1, 0.6 );
    slice.set( 1, 2, 0.7 );
    slice.set( 1, 3, 0.8 );
    slice.set( 2, 0, 0.9 );
    slice.set( 2, 1, 0.10 );
    slice.set( 2, 2, 0.11 );
    slice.set( 2, 3, 0.12 );
    slice.set( 3, 0, 0.13 );
    slice.set( 3, 1, 0.14 );
    slice.set( 3, 2, 0.15 );
    slice.set( 3, 3, 0.16 );

    assert_eq!( data, &[
        0.1,  0.2,  0.3,  0.4,
        0.5,  0.6,  0.7,  0.8,
        0.9,  0.10, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16
    ]);
}

#[test]
fn test_square_slice_split_and_set() {
    let data = &mut [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ];

    let mut slice = SquareSlice::new( &mut data[..], 4 );
    let (mut ul, mut ur, mut ll, mut lr) = slice.split_in_four();
    ul.set( 0, 0, 0.1 );
    ul.set( 0, 1, 0.2 );
    ul.set( 1, 0, 0.3 );
    ul.set( 1, 1, 0.4 );

    ur.set( 0, 0, 0.5 );
    ur.set( 0, 1, 0.6 );
    ur.set( 1, 0, 0.7 );
    ur.set( 1, 1, 0.8 );

    ll.set( 0, 0, 0.9 );
    ll.set( 0, 1, 0.10 );
    ll.set( 1, 0, 0.11 );
    ll.set( 1, 1, 0.12 );

    lr.set( 0, 0, 0.13 );
    lr.set( 0, 1, 0.14 );
    lr.set( 1, 0, 0.15 );
    lr.set( 1, 1, 0.16 );

    assert_eq!( data, &[
        0.1,  0.2,  0.5,  0.6,
        0.3,  0.4,  0.7,  0.8,
        0.9,  0.10, 0.13, 0.14,
        0.11, 0.12, 0.15, 0.16
    ]);
}

#[test]
fn test_square_slice_double_split_and_set() {
    let data = &mut [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ];

    let mut slice = SquareSlice::new( &mut data[..], 4 );
    let (mut ul, mut ur, mut ll, mut lr) = slice.split_in_four();

    let (mut a, mut b, mut c, mut d) = ul.split_in_four();
    a.set( 0, 0, 0.1 );
    b.set( 0, 0, 0.2 );
    c.set( 0, 0, 0.3 );
    d.set( 0, 0, 0.4 );

    let (mut a, mut b, mut c, mut d) = ur.split_in_four();
    a.set( 0, 0, 0.5 );
    b.set( 0, 0, 0.6 );
    c.set( 0, 0, 0.7 );
    d.set( 0, 0, 0.8 );

    let (mut a, mut b, mut c, mut d) = ll.split_in_four();
    a.set( 0, 0, 0.9 );
    b.set( 0, 0, 0.10 );
    c.set( 0, 0, 0.11 );
    d.set( 0, 0, 0.12 );

    let (mut a, mut b, mut c, mut d) = lr.split_in_four();
    a.set( 0, 0, 0.13 );
    b.set( 0, 0, 0.14 );
    c.set( 0, 0, 0.15 );
    d.set( 0, 0, 0.16 );

    assert_eq!( data, &[
        0.1,  0.2,  0.5,  0.6,
        0.3,  0.4,  0.7,  0.8,
        0.9,  0.10, 0.13, 0.14,
        0.11, 0.12, 0.15, 0.16
    ]);
}

#[test]
fn test_square_slice_row() {
    let data = &mut [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ];

    let mut slice = SquareSlice::new( &mut data[..], 4 );
    assert_eq!( slice.row( 0 ).len(), 4 );
    assert_eq!( slice.row( 1 ).len(), 4 );
    assert_eq!( slice.row( 2 ).len(), 4 );
    assert_eq!( slice.row( 3 ).len(), 4 );

    slice.row( 0 )[ 0 ] = 0.1;
    slice.row( 0 )[ 1 ] = 0.2;
    slice.row( 0 )[ 2 ] = 0.3;
    slice.row( 0 )[ 3 ] = 0.4;
    slice.row( 1 )[ 0 ] = 0.5;
    slice.row( 1 )[ 1 ] = 0.6;
    slice.row( 1 )[ 2 ] = 0.7;
    slice.row( 1 )[ 3 ] = 0.8;
    slice.row( 2 )[ 0 ] = 0.9;
    slice.row( 2 )[ 1 ] = 0.10;
    slice.row( 2 )[ 2 ] = 0.11;
    slice.row( 2 )[ 3 ] = 0.12;
    slice.row( 3 )[ 0 ] = 0.13;
    slice.row( 3 )[ 1 ] = 0.14;
    slice.row( 3 )[ 2 ] = 0.15;
    slice.row( 3 )[ 3 ] = 0.16;

    assert_eq!( data, &[
        0.1,  0.2,  0.3,  0.4,
        0.5,  0.6,  0.7,  0.8,
        0.9,  0.10, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16
    ]);
}

struct SinCos< 'a > {
    table: &'a [(f32, f32)]
}

impl< 'a > SinCos< 'a > {
    #[inline]
    fn get( &self, index: usize ) -> (f32, f32) {
        self.table[ index ]
    }
}

#[inline(always)]
fn ortho_chunk_4x4( matrix: &mut SquareSlice, ab: f32x2, sc: f32x4 ) {
    let ab_x2 = shuffle!( ab, [0, 0, 1, 1] );
    let ba_x2 = shuffle!( ab, [1, 1, 0, 0] );

    let ucos_usin: f32x4 = shuffle!( sc, [1, 0, 1, 0] );
    let usin_ucos: f32x4 = shuffle!( sc, [0, 1, 0, 1] );
    let lcos_lsin: f32x4 = shuffle!( sc, [3, 2, 3, 2] );
    let lsin_lcos: f32x4 = shuffle!( sc, [2, 3, 2, 3] );

    let r0 = ucos_usin * ba_x2 * f32x4::new( 1.0, -1.0, -1.0, 1.0 );
    r0.write_to_slice_unaligned( matrix.row( 0 ) );

    let r1 = usin_ucos * ba_x2 * f32x4::new( 1.0, 1.0, -1.0, -1.0 );
    r1.write_to_slice_unaligned( matrix.row( 1 ) );

    let r2 = lcos_lsin * ab_x2 * f32x4::new( 1.0, -1.0, 1.0, -1.0 );
    r2.write_to_slice_unaligned( matrix.row( 2 ) );

    let r3 = lsin_lcos * ab_x2;
    r3.write_to_slice_unaligned( matrix.row( 3 ) );
}

#[inline(always)]
fn ortho_chunk_2x2( matrix: &mut SquareSlice, ab: f32x2 ) {
    let upper = shuffle!( ab, [1, 0] ) * f32x2::new( 1.0, -1.0 );
    let lower = ab;

    upper.write_to_slice_unaligned( matrix.row( 0 ) );
    lower.write_to_slice_unaligned( matrix.row( 1 ) );
}

#[cfg(feature = "rayon")]
#[inline(always)]
fn maybe_join< A, B >( size: usize, a: A, b: B )
    where A: FnOnce() + Send,
          B: FnOnce() + Send
{
    if size <= 16 {
        a();
        b();
    } else {
        rayon::join( move || {
            a();
        }, move || {
            b();
        });
    }
}

#[cfg(not(feature = "rayon"))]
#[inline(always)]
fn maybe_join< A, B >( _: usize, a: A, b: B )
    where A: FnOnce() + Send,
          B: FnOnce() + Send
{
    a();
    b();
}

// This is based on the "Methods for Generating Random Orthogonal Matrices"
// paper by Alan Genz available here:
//     http://www.math.wsu.edu/faculty/genz/papers/rndorth.ps
fn ortho_chunk( sincos: &SinCos, mut matrix: SquareSlice, shift: usize, ab: f32x2 ) {
    let size = matrix.size();
    debug_assert_ne!( size, 1 );

    let n = size / 2;
    let (usin, ucos) = sincos.get( n / 2 + shift - 1 );
    let (lsin, lcos) = sincos.get( n / 2 + shift + n - 1 );
    let sc = f32x4::new( usin, ucos, lsin, lcos );

    if size == 4 {
        ortho_chunk_4x4( &mut matrix, ab, sc );
        return;
    }

    let ab_1 = sc * ab.extract( 0 ) * f32x4::new( -1.0, -1.0, 1.0, 1.0 );
    let ur_ab = shuffle!( ab_1, [0, 1] );
    let ll_ab = shuffle!( ab_1, [2, 3] );

    let ab_2 = sc * ab.extract( 1 );
    let ul_ab = shuffle!( ab_2, [0, 1] );
    let lr_ab = shuffle!( ab_2, [2, 3] );

    let (ul, ur, ll, lr) = matrix.split_in_four();
    maybe_join( size, move || {
        ortho_chunk( sincos, ul, shift, ul_ab );
        ortho_chunk( sincos, ur, shift, ur_ab );
    }, move || {
        ortho_chunk( sincos, ll, shift + n, ll_ab );
        ortho_chunk( sincos, lr, shift + n, lr_ab );
    });
}

fn ortho_into_buffer< I >( angles: I, scratch: &mut Vec< (f32, f32) >, output: &mut [f32] )
    where I: IntoIterator< Item = f32 >, <I as IntoIterator>::IntoIter: ExactSizeIterator
{
    let angles = angles.into_iter();
    let size = angles.len() + 1;

    if size == 1 {
        output[ 0 ] = 1.0;
        return;
    }

    assert!( size.is_power_of_two() );
    assert_eq!( output.len(), size * size );

    scratch.clear();
    scratch.reserve( angles.len() );
    unsafe {
        scratch.set_len( angles.len() );
    }

    for (out, x) in scratch.iter_mut().zip( angles ) {
        *out = x.sin_cos();
    }

    let sincos = SinCos { table: scratch.as_slice() };

    let (sin, cos) = sincos.get( size / 2 - 1 );
    let ab = f32x2::new( sin, cos );

    let mut slice = SquareSlice::new( output, size );
    if size == 2 {
        ortho_chunk_2x2( &mut slice, ab );
    } else {
        ortho_chunk( &sincos, slice, 0, ab );
    }

    scratch.clear();
}

fn reshape_into_rows< 'a >( size: usize, matrix: &'a [f32], width: usize, height: usize )
    -> impl Iterator< Item = &[f32] > + ExactSizeIterator + FusedIterator + 'a
{
    struct Iter< 'a > {
        matrix: &'a [f32],
        stride: usize,
        position: usize,
        target_width: usize,
        rows_remaining: usize,
    }

    impl< 'a > Iterator for Iter< 'a > {
        type Item = &'a [f32];
        fn next( &mut self ) -> Option< Self::Item > {
            if self.rows_remaining == 0 {
                return None
            };

            let slice = &self.matrix[ self.position..self.position + self.target_width ];
            self.position += self.stride;
            self.rows_remaining -= 1;

            Some( slice )
        }

        fn size_hint( &self ) -> (usize, Option< usize >) {
            (self.rows_remaining, Some( self.rows_remaining ))
        }
    }

    impl< 'a > ExactSizeIterator for Iter< 'a > {}
    impl< 'a > FusedIterator for Iter< 'a > {}

    assert!( width <= size, height <= size );
    assert_eq!( size * size, matrix.len() );

    Iter {
        matrix,
        stride: size,
        position: 0,
        target_width: width,
        rows_remaining: height
    }
}

fn to_nearest_power_of_two( width: usize, height: usize ) -> usize {
    std::cmp::max( width, height ).next_power_of_two()
}

pub struct OrthogonalGenerator {
    buffer: Vec< f32 >,
    scratch: Vec< (f32, f32) >
}

impl OrthogonalGenerator {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            scratch: Vec::new()
        }
    }

    fn generate_with_angles_into< F >( &mut self, width: usize, height: usize, get_angle: F, output: &mut Vec< f32 > )
        where F: FnMut( usize ) -> f32
    {
        let size = to_nearest_power_of_two( width, height );
        let angles = (0..size - 1).map( get_angle );

        self.buffer.clear();
        self.buffer.reserve( size * size );

        unsafe {
            self.buffer.set_len( size * size );
        }

        ortho_into_buffer( angles, &mut self.scratch, &mut self.buffer );

        for row in reshape_into_rows( size, &self.buffer, width, height ) {
            output.extend_from_slice( row );
        }

        self.buffer.clear();
        self.scratch.clear();
    }

    pub fn generate_into( &mut self, width: usize, height: usize, rng: &mut dyn RngCore, output: &mut Vec< f32 > ) {
        use std::f32::consts::PI;
        let dist = rand::distributions::Uniform::new_inclusive( PI * 0.125, PI * 0.875 );
        self.generate_with_angles_into( width, height, |_| dist.sample( rng ) as f32, output );
    }
}

#[cfg(test)]
fn ortho< I >( angles: I ) -> Vec< f32 >
    where I: IntoIterator< Item = f32 >, <I as IntoIterator>::IntoIter: ExactSizeIterator
{
    let mut angles = angles.into_iter();
    let size = angles.len() + 1;
    let mut output = Vec::new();
    let mut generator = OrthogonalGenerator::new();
    generator.generate_with_angles_into( size, size, |_| angles.next().unwrap(), &mut output );
    output
}

#[cfg(test)]
fn reshape( size: usize, matrix: &[f32], width: usize, height: usize ) -> Vec< f32 > {
    let mut output = Vec::with_capacity( width * height );
    for row in reshape_into_rows( size, matrix, width, height ) {
        output.extend_from_slice( row );
    }
    output
}

#[test]
fn test_reshape() {
    let matrix = &[
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  7.0,
         8.0,  9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0
    ];

    assert_eq!(
        reshape( 4, matrix, 1, 1 ),
        &[ 1.0 ]
    );

    assert_eq!(
        reshape( 4, matrix, 2, 1 ),
        &[ 1.0, 2.0 ]
    );

    assert_eq!(
        reshape( 4, matrix, 1, 2 ),
        &[
            1.0,
            5.0
        ]
    );

    assert_eq!(
        reshape( 4, matrix, 2, 2 ),
        &[
            1.0, 2.0,
            5.0, 6.0
        ]
    );
}

#[test]
fn test_to_nearest_power_of_two() {
    assert_eq!( to_nearest_power_of_two( 1, 1 ), 1 );
    assert_eq!( to_nearest_power_of_two( 1, 2 ), 2 );
    assert_eq!( to_nearest_power_of_two( 1, 3 ), 4 );
    assert_eq!( to_nearest_power_of_two( 1, 1 ), 1 );
    assert_eq!( to_nearest_power_of_two( 2, 1 ), 2 );
    assert_eq!( to_nearest_power_of_two( 3, 1 ), 4 );
    assert_eq!( to_nearest_power_of_two( 3, 3 ), 4 );
    assert_eq!( to_nearest_power_of_two( 4, 3 ), 4 );
    assert_eq!( to_nearest_power_of_two( 3, 4 ), 4 );
    assert_eq!( to_nearest_power_of_two( 4, 4 ), 4 );
}

#[cfg(test)]
use testutils::{
    assert_f32_eq,
    assert_f32_slice_eq
};

#[cfg(test)]
fn norm( vec: &[f32] ) -> f32 {
    vec.iter().cloned().map( |x| x * x ).sum::< f32 >().sqrt()
}

#[cfg(test)]
fn mul_matvec( matrix: &[f32], vec: &[f32] ) -> Vec< f32 > {
    let size = matrix.len() / vec.len();
    (0..size).into_iter().map( |y| {
        let mut sum = 0.0;
        for x in 0..vec.len() {
            sum += vec[ x ] * matrix[ y * size + x ];
        }
        sum
    }).collect()
}

#[test]
fn test_mul_mulvec() {
    assert_f32_slice_eq(
        &mul_matvec(
            &[
                1.0, 2.0,
                3.0, 4.0
            ],
            &[ 10.0, 20.0 ]
        ),
        &[ 50.0, 110.0 ]
    );

    assert_f32_slice_eq(
        &mul_matvec(
            &[
                1.0, 2.0
            ],
            &[ 10.0, 20.0 ]
        ),
        &[ 50.0 ]
    );

    assert_f32_slice_eq(
        &mul_matvec(
            &[
                1.0, 2.0, 3.0
            ],
            &[ 10.0, 20.0, 50.0 ]
        ),
        &[ 200.0 ]
    );
}

#[test]
fn test_ortho_1x1() {
    let ys = ortho( std::iter::empty() );
    let expected = &[
        1.0
    ];

    assert_f32_slice_eq( &ys, expected );
}

#[test]
fn test_ortho_2x2() {
    let angles: &[f32] = &[0.25];
    let ys = ortho( angles.into_iter().cloned() );

    let s: Vec< _ > = angles.iter().cloned().map( |x| x.sin() ).collect();
    let c: Vec< _ > = angles.iter().cloned().map( |x| x.cos() ).collect();

    let expected = &[
        c[0], -s[0],
        s[0],  c[0]
    ];

    assert_f32_slice_eq( &ys, expected );

    let a: Vec< _ > = (0..2).into_iter().map( |n| 0.123 * n as f32 ).collect();
    let b = mul_matvec( &ys, &a );
    assert_f32_eq( norm( &a ), norm( &b ) );
}

#[test]
fn test_ortho_4x4() {
    let angles: &[f32] = &[0.25, 0.66, 1.88];
    let ys = ortho( angles.into_iter().cloned() );

    let s: Vec< _ > = angles.iter().cloned().map( |x| x.sin() ).collect();
    let c: Vec< _ > = angles.iter().cloned().map( |x| x.cos() ).collect();

    let expected = &[
        c[0]*c[1], -s[0]*c[1], -c[0]*s[1],  s[0]*s[1],
        s[0]*c[1],  c[0]*c[1], -s[0]*s[1], -c[0]*s[1],
        c[2]*s[1], -s[2]*s[1],  c[2]*c[1], -s[2]*c[1],
        s[2]*s[1],  c[2]*s[1],  s[2]*c[1],  c[2]*c[1]
    ];

    assert_f32_slice_eq( &ys, expected );

    let a: Vec< _ > = (0..4).into_iter().map( |n| 0.123 * n as f32 ).collect();
    let b = mul_matvec( &ys, &a );
    assert_f32_eq( norm( &a ), norm( &b ) );
}

#[test]
fn test_ortho_8x8() {
    let angles: &[f32] = &[0.25, 0.66, 1.88, 2.11, 2.44, 2.88, 3.00];
    let ys = ortho( angles.into_iter().cloned() );

    let s: Vec< _ > = angles.iter().cloned().map( |x| x.sin() ).collect();
    let c: Vec< _ > = angles.iter().cloned().map( |x| x.cos() ).collect();

    let expected = &[
        c[0]*c[1]*c[3], -s[0]*c[1]*c[3], -c[0]*s[1]*c[3],  s[0]*s[1]*c[3], -c[0]*c[1]*s[3],  s[0]*c[1]*s[3],  c[0]*s[1]*s[3], -s[0]*s[1]*s[3],
        s[0]*c[1]*c[3],  c[0]*c[1]*c[3], -s[0]*s[1]*c[3], -c[0]*s[1]*c[3], -s[0]*c[1]*s[3], -c[0]*c[1]*s[3],  s[0]*s[1]*s[3],  c[0]*s[1]*s[3],
        c[2]*s[1]*c[3], -s[2]*s[1]*c[3],  c[2]*c[1]*c[3], -s[2]*c[1]*c[3], -c[2]*s[1]*s[3],  s[2]*s[1]*s[3], -c[2]*c[1]*s[3],  s[2]*c[1]*s[3],
        s[2]*s[1]*c[3],  c[2]*s[1]*c[3],  s[2]*c[1]*c[3],  c[2]*c[1]*c[3], -s[2]*s[1]*s[3], -c[2]*s[1]*s[3], -s[2]*c[1]*s[3], -c[2]*c[1]*s[3],
        c[4]*c[5]*s[3], -s[4]*c[5]*s[3], -c[4]*s[5]*s[3],  s[4]*s[5]*s[3],  c[4]*c[5]*c[3], -s[4]*c[5]*c[3], -c[4]*s[5]*c[3],  s[4]*s[5]*c[3],
        s[4]*c[5]*s[3],  c[4]*c[5]*s[3], -s[4]*s[5]*s[3], -c[4]*s[5]*s[3],  s[4]*c[5]*c[3],  c[4]*c[5]*c[3], -s[4]*s[5]*c[3], -c[4]*s[5]*c[3],
        c[6]*s[5]*s[3], -s[6]*s[5]*s[3],  c[6]*c[5]*s[3], -s[6]*c[5]*s[3],  c[6]*s[5]*c[3], -s[6]*s[5]*c[3],  c[6]*c[5]*c[3], -s[6]*c[5]*c[3],
        s[6]*s[5]*s[3],  c[6]*s[5]*s[3],  s[6]*c[5]*s[3],  c[6]*c[5]*s[3],  s[6]*s[5]*c[3],  c[6]*s[5]*c[3],  s[6]*c[5]*c[3],  c[6]*c[5]*c[3]
    ];

    assert_f32_slice_eq( &ys, expected );

    let a: Vec< _ > = (0..8).into_iter().map( |n| 0.123 * n as f32 ).collect();
    let b = mul_matvec( &ys, &a );
    assert_f32_eq( norm( &a ), norm( &b ) );
}

#[test]
fn test_ortho_32x32() {
    let size = 32;
    let angles = (0..(size - 1)).into_iter().map( |x| x as f32 / (size as f32) * 3.1415 );
    let ys = ortho( angles );

    let a: Vec< _ > = (0..32).into_iter().map( |n| 0.123 * n as f32 ).collect();
    let b = mul_matvec( &ys, &a );
    assert_f32_eq( norm( &a ), norm( &b ) );
}
