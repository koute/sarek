fn are_approximately_equal( a: f32, b: f32 ) -> bool {
    (a - b).abs() <= 0.00001
}

fn are_slices_approximately_equal< A: AsRef< [f32] >, B: AsRef< [f32] > >( a_raw: A, b_raw: B ) -> bool {
    let a = a_raw.as_ref();
    let b = b_raw.as_ref();

    if a.len() != b.len() {
        return false;
    }

    return a.len() == b.len() && a.iter().zip( b.iter() ).all( |(&a, &b)| are_approximately_equal( a, b ) );
}

pub fn assert_f32_slice_eq( lhs: &[f32], rhs: &[f32] ) {
    assert!( are_slices_approximately_equal( &lhs, &rhs ), format!( "left: {:?}, right: {:?}", lhs, rhs ) );
}

pub fn assert_f32_eq( lhs: f32, rhs: f32 ) {
    assert!( are_approximately_equal( lhs, rhs ), format!( "left: {:?}, right: {:?}", lhs, rhs ) );
}
