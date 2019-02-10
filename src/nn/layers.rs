use {
    std::{
        fmt,
        iter,
        slice,
        sync::{
            Arc
        }
    },
    decorum,
    crate::{
        core::{
            name::{
                Name
            },
            shape::{
                Shape
            },
            utils::{
                SliceDebug
            }
        },
        nn::{
            activation::{
                Activation
            }
        }
    }
};

pub trait LayerPrototype {
    fn name( &self ) -> &Name;
    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name >;
    fn output_shape( &self, input_shape: &Shape ) -> Shape;
    fn weight_count( &self, input_shape: &Shape ) -> usize;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerActivation {
    pub(crate) name: Name,
    pub(crate) activation: Activation
}

impl LayerActivation {
    pub fn new() -> LayerActivation {
        LayerActivation {
            name: Name::new_unique(),
            activation: Activation::ELU
        }
    }

    pub fn activation( &self ) -> Activation {
        self.activation.clone()
    }

    pub fn set_activation( &mut self, value: Activation ) -> &mut Self {
        self.activation = value;
        self
    }
}

impl LayerPrototype for LayerActivation {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq)]
pub struct LayerConvolution {
    pub(crate) name: Name,
    pub(crate) filter_count: usize,
    pub(crate) kernel_size: (usize, usize),
    pub(crate) weights: Option< Arc< Vec< f32 > > >
}

impl Eq for LayerConvolution {}
impl fmt::Debug for LayerConvolution {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "LayerConvolution" )
            .field( "name", &self.name )
            .field( "filter_count", &self.filter_count )
            .field( "kernel_size", &self.kernel_size )
            .field( "weights", &self.weights.as_ref().map( |slice| SliceDebug( &slice ) ) )
            .finish()
    }
}

impl LayerConvolution {
    pub fn new( filter_count: usize, kernel_size: (usize, usize) ) -> Self {
        assert_ne!( filter_count, 0 );
        assert_ne!( kernel_size.0, 0 );
        assert_ne!( kernel_size.1, 0 );

        Self {
            name: Name::new_unique(),
            filter_count,
            kernel_size,
            weights: None
        }
    }

    pub fn set_weights( &mut self, weights: Vec< f32 > ) -> &mut Self {
        assert!(
            !weights.iter().cloned().any( |value| value.is_nan() || value.is_infinite() ),
            "Weights contain either a NaN or an Inf"
        );

        self.weights = Some( Arc::new( weights ) );
        self
    }
}

impl LayerPrototype for LayerConvolution {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        let input_dimensions = input_shape.dimension_count();
        assert!( input_dimensions == 2 || input_dimensions == 3 );
        assert!( self.kernel_size.0 <= input_shape.x() );
        assert!( self.kernel_size.1 <= input_shape.y() );

        let out_x = input_shape.x() - self.kernel_size.0 + 1;
        let out_y = input_shape.y() - self.kernel_size.1 + 1;
        let out_z = self.filter_count;

        Shape::new_3d( out_x, out_y, out_z )
    }

    fn weight_count( &self, input_shape: &Shape ) -> usize {
        let input_dimensions = input_shape.dimension_count();
        assert!( input_dimensions == 2 || input_dimensions == 3 );
        self.kernel_size.0 * self.kernel_size.1 * input_shape.z() * self.filter_count + self.filter_count
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerDense {
    pub(crate) name: Name,
    pub(crate) size: usize
}

impl LayerDense {
    pub fn new( size: usize ) -> LayerDense {
        LayerDense {
            name: Name::new_unique(),
            size
        }
    }

    pub fn size( &self ) -> usize {
        self.size
    }

    pub fn set_size( &mut self, value: usize ) -> &mut Self {
        self.size = value;
        self
    }
}

impl LayerPrototype for LayerDense {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, _: &Shape ) -> Shape {
        Shape::new_1d( self.size )
    }

    fn weight_count( &self, input_shape: &Shape ) -> usize {
        (input_shape.product() + 1) * self.size
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerDropout {
    pub(crate) name: Name,
    pub(crate) rate: decorum::Ordered< f32 >
}

impl LayerDropout {
    pub fn new( rate: f32 ) -> LayerDropout {
        LayerDropout {
            name: Name::new_unique(),
            rate: rate.into()
        }
    }
}

impl LayerPrototype for LayerDropout {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

/// A layer which picks the maximum input and outputs the index of that input.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerIntoCategory {
    pub(crate) name: Name
}

impl LayerIntoCategory {
    pub fn new() -> LayerIntoCategory {
        LayerIntoCategory {
            name: Name::new_unique()
        }
    }
}

impl LayerPrototype for LayerIntoCategory {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, _: &Shape ) -> Shape {
        1.into()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerMaxPooling {
    pub(crate) name: Name,
    pub(crate) pool_size: (usize, usize)
}

impl LayerMaxPooling {
    pub fn new( pool_size: (usize, usize) ) -> Self {
        Self {
            name: Name::new_unique(),
            pool_size
        }
    }
}

impl LayerPrototype for LayerMaxPooling {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        let input_dimensions = input_shape.dimension_count();
        assert!( input_dimensions == 2 || input_dimensions == 3 );
        assert!( self.pool_size.0 <= input_shape.x() );
        assert!( self.pool_size.1 <= input_shape.y() );
        Shape::new_3d(
            (input_shape.x() as f32 / self.pool_size.0 as f32).ceil() as usize,
            (input_shape.y() as f32 / self.pool_size.1 as f32).ceil() as usize,
            input_shape.z()
        )
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq)]
pub struct LayerMultiply {
    pub(crate) name: Name,
    pub(crate) values: Arc< Vec< f32 > >
}

impl Eq for LayerMultiply {}
impl fmt::Debug for LayerMultiply {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "LayerMultiply" )
            .field( "name", &self.name )
            .field( "values", &SliceDebug( &self.values ) )
            .finish()
    }
}

impl LayerMultiply {
    pub fn new( values: Vec< f32 > ) -> Self {
        Self {
            name: Name::new_unique(),
            values: Arc::new( values )
        }
    }
}

impl LayerPrototype for LayerMultiply {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerReshape {
    pub(crate) name: Name,
    pub(crate) shape: Shape
}

impl LayerReshape {
    pub fn new( shape: Shape ) -> Self {
        Self {
            name: Name::new_unique(),
            shape
        }
    }
}

impl LayerPrototype for LayerReshape {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        assert_eq!( input_shape.product(), self.shape.product() );
        self.shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq)]
pub struct LayerShift {
    pub(crate) name: Name,
    pub(crate) values: Arc< Vec< f32 > >
}

impl Eq for LayerShift {}
impl fmt::Debug for LayerShift {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "LayerShift" )
            .field( "name", &self.name )
            .field( "values", &SliceDebug( &self.values ) )
            .finish()
    }
}

impl LayerShift {
    pub fn new( values: Vec< f32 > ) -> Self {
        Self {
            name: Name::new_unique(),
            values: Arc::new( values )
        }
    }
}

impl LayerPrototype for LayerShift {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

/// The softmax layer normalizes its inputs so that they sum to `1`.
///
/// More precisely it interprets the inputs as unnormalized log probabilities
/// and outputs normalized linear probabilities. This is particularly useful
/// in classification tasks, and is usually used as the last layer of a network.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerSoftmax {
    pub(crate) name: Name
}

impl LayerSoftmax {
    pub fn new() -> LayerSoftmax {
        LayerSoftmax {
            name: Name::new_unique()
        }
    }
}

impl LayerPrototype for LayerSoftmax {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Layer {
    Activation( LayerActivation ),
    Convolution( LayerConvolution ),
    Dense( LayerDense ),
    Dropout( LayerDropout ),
    IntoCategory( LayerIntoCategory ),
    MaxPooling( LayerMaxPooling ),
    Multiply( LayerMultiply ),
    Reshape( LayerReshape ),
    Shift( LayerShift ),
    Softmax( LayerSoftmax )
}

macro_rules! layer_boilerplate {
    ($(Layer::$variant:ident( $name:ident ))*) => {
        impl LayerPrototype for Layer {
            fn name( &self ) -> &Name {
                match *self {
                    $( Layer::$variant( ref layer ) => layer.name(), )*
                }
            }

            fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
                match *self {
                    $(
                        Layer::$variant( ref mut layer ) => {
                            layer.set_name( name );
                        },
                    )*
                }

                self
            }

            fn output_shape( &self, input_shape: &Shape ) -> Shape {
                match *self {
                    $( Layer::$variant( ref layer ) => layer.output_shape( input_shape ), )*
                }
            }

            fn weight_count( &self, input_shape: &Shape ) -> usize {
                match *self {
                    $( Layer::$variant( ref layer ) => layer.weight_count( input_shape ), )*
                }
            }
        }

        $(
            impl From< $name > for Layer {
                #[inline]
                fn from( layer: $name ) -> Self {
                    Layer::$variant( layer )
                }
            }

            impl< 'a > From< &'a $name > for Layer {
                #[inline]
                fn from( layer: &'a $name ) -> Self {
                    layer.clone().into()
                }
            }

            impl< 'a > From< &'a mut $name > for Layer {
                #[inline]
                fn from( layer: &'a mut $name ) -> Self {
                    layer.clone().into()
                }
            }
        )*
    }
}

impl< 'a > From< &'a Layer > for Layer {
    #[inline]
    fn from( layer: &'a Layer ) -> Self {
        layer.clone()
    }
}

impl< 'a > From< &'a mut Layer > for Layer {
    #[inline]
    fn from( layer: &'a mut Layer ) -> Self {
        layer.clone()
    }
}

layer_boilerplate!(
    Layer::Activation( LayerActivation )
    Layer::Convolution( LayerConvolution )
    Layer::Dense( LayerDense )
    Layer::Dropout( LayerDropout )
    Layer::IntoCategory( LayerIntoCategory )
    Layer::MaxPooling( LayerMaxPooling )
    Layer::Multiply( LayerMultiply )
    Layer::Reshape( LayerReshape )
    Layer::Shift( LayerShift )
    Layer::Softmax( LayerSoftmax )
);

pub trait IntoLayerIter {
    type Iter: Iterator< Item = Layer >;
    fn into_layer_iter( self ) -> Self::Iter;
}

pub struct LayerIter< 'a, T >( slice::Iter< 'a, T > );

impl< 'a, T > Iterator for LayerIter< 'a, T > where &'a T: Into< Layer > {
    type Item = Layer;
    fn next( &mut self ) -> Option< Self::Item > {
        self.0.next().map( |layer| layer.into() )
    }
}

impl< 'a, T > IntoLayerIter for &'a [T] where &'a T: Into< Layer > {
    type Iter = LayerIter< 'a, T >;
    fn into_layer_iter( self ) -> Self::Iter {
        LayerIter( self.into_iter() )
    }
}

macro_rules! impl_into_layer_iter {
    (@access $this:expr, L00) => { $this.0 };
    (@access $this:expr, L01) => { $this.1 };
    (@access $this:expr, L02) => { $this.2 };
    (@access $this:expr, L03) => { $this.3 };
    (@access $this:expr, L04) => { $this.4 };
    (@access $this:expr, L05) => { $this.5 };
    (@access $this:expr, L06) => { $this.6 };
    (@access $this:expr, L07) => { $this.7 };
    (@access $this:expr, L08) => { $this.8 };
    (@access $this:expr, L09) => { $this.9 };
    (@access $this:expr, L10) => { $this.10 };
    (@access $this:expr, L11) => { $this.11 };
    (@access $this:expr, L12) => { $this.12 };
    (@access $this:expr, L13) => { $this.13 };
    (@access $this:expr, L14) => { $this.14 };
    (@access $this:expr, L15) => { $this.15 };
    (@access $this:expr, L16) => { $this.16 };
    (@access $this:expr, L17) => { $this.17 };
    (@access $this:expr, L18) => { $this.18 };
    (@access $this:expr, L19) => { $this.19 };
    (@access $this:expr, L20) => { $this.20 };
    (@access $this:expr, L21) => { $this.21 };
    (@access $this:expr, L22) => { $this.22 };
    (@access $this:expr, L23) => { $this.23 };
    (@access $this:expr, L24) => { $this.24 };
    (@access $this:expr, L25) => { $this.25 };
    (@access $this:expr, L26) => { $this.26 };
    (@access $this:expr, L27) => { $this.27 };
    (@access $this:expr, L28) => { $this.28 };
    (@access $this:expr, L29) => { $this.29 };

    (@body $this:expr, $initial_type:ident $($type:ident)*) => {
        iter::once( $this.0.into() )
        $(
            .chain( iter::once(
                impl_into_layer_iter!( @access $this, $type ).into()
            ))
        )*
    };

    (@iter_type $type:ident) => {
        iter::Once< Layer >
    };

    (@iter_type $lhs:ident $($rhs:ident)*) => {
        iter::Chain< impl_into_layer_iter!( @iter_type $($rhs)* ), iter::Once< Layer > >
    };

    (@impl $($type:ident)*) => {
        impl< $($type),* > IntoLayerIter for ($($type,)*) where $($type: Into< Layer >),* {
            type Iter = impl_into_layer_iter!( @iter_type $($type)* );
            fn into_layer_iter( self ) -> Self::Iter {
                impl_into_layer_iter!( @body self, $($type)* )
            }
        }
    };

    (@call_1 [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_into_layer_iter!( @impl $($type)* );
        impl_into_layer_iter!( @call_1 [$($dummy_type)*] [$($type)* $lhs] );
    };

    (@call_1 [] [$($type:ident)*]) => {};

    (@call [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_into_layer_iter!( @call_1 [$($dummy_type)*] [$lhs $($type)*] );
    };

    () => {
        impl_into_layer_iter!(
            @call
                [
                    L00 L01 L02 L03 L04 L05 L06 L07 L08 L09
                    L10 L11 L12 L13 L14 L15 L16 L17 L18 L19
                    L20 L21 L22 L23 L24 L25 L26 L27 L28 L29
                ]
                []
        );
    };
}

impl< A > IntoLayerIter for A where A: Into< Layer > {
    type Iter = iter::Once< Layer >;
    fn into_layer_iter( self ) -> Self::Iter {
        iter::once( self.into() )
    }
}

impl IntoLayerIter for () {
    type Iter = iter::Empty< Layer >;
    fn into_layer_iter( self ) -> Self::Iter {
        iter::empty()
    }
}

/*
    This is what this generates:

        impl< A, B > IntoLayerIter for (A, B) where A: Into< Layer >, B: Into< Layer > {
            type Iter = iter::Chain< iter::Once< Layer >, iter::Once< Layer > >;
            fn into_layer_iter( self ) -> Self::Iter {
                iter::once( self.0.into() )
                    .chain( iter::once( self.1.into() ) )
            }
        }

        impl< A, B, C > IntoLayerIter for (A, B, C) where A: Into< Layer >, B: Into< Layer >, C: Into< Layer > {
            type Iter = iter::Chain< iter::Chain< iter::Once< Layer >, iter::Once< Layer > >, iter::Once< Layer > >;
            fn into_layer_iter( self ) -> Self::Iter {
                iter::once( self.0.into() )
                    .chain( iter::once( self.1.into() ) )
                    .chain( iter::once( self.2.into() ) )
            }
        }
*/

impl_into_layer_iter!();

#[test]
fn test_layer_convolution_prototype() {
    struct T {
        input: Shape,
        filter_count: usize,
        kernel_size: (usize, usize),
        expected_output: Shape,
        expected_weight_count: usize
    }

    fn test( data: T ) {
        let layer = LayerConvolution::new( data.filter_count, data.kernel_size );
        let output = layer.output_shape( &data.input );
        let weight_count = layer.weight_count( &data.input );
        assert_eq!( output, data.expected_output );
        assert_eq!( weight_count, data.expected_weight_count );
    }

    test( T {
        input: Shape::new_2d( 4, 4 ),
        filter_count: 1,
        kernel_size: (1, 1),
        expected_output: Shape::new_3d( 4, 4, 1 ),
        expected_weight_count: 1 + 1
    });

    test( T {
        input: Shape::new_2d( 4, 4 ),
        filter_count: 1,
        kernel_size: (2, 2),
        expected_output: Shape::new_3d( 3, 3, 1 ),
        expected_weight_count: 4 + 1
    });

    test( T {
        input: Shape::new_2d( 4, 4 ),
        filter_count: 1,
        kernel_size: (3, 3),
        expected_output: Shape::new_3d( 2, 2, 1 ),
        expected_weight_count: 9 + 1
    });

    test( T {
        input: Shape::new_2d( 4, 4 ),
        filter_count: 1,
        kernel_size: (4, 4),
        expected_output: Shape::new_3d( 1, 1, 1 ),
        expected_weight_count: 16 + 1
    });

    test( T {
        input: Shape::new_2d( 4, 4 ),
        filter_count: 2,
        kernel_size: (3, 3),
        expected_output: Shape::new_3d( 2, 2, 2 ),
        expected_weight_count: 18 + 2
    });

    test( T {
        input: Shape::new_3d( 4, 4, 1 ),
        filter_count: 2,
        kernel_size: (3, 3),
        expected_output: Shape::new_3d( 2, 2, 2 ),
        expected_weight_count: 18 + 2
    });

    test( T {
        input: Shape::new_3d( 4, 4, 2 ),
        filter_count: 2,
        kernel_size: (3, 3),
        expected_output: Shape::new_3d( 2, 2, 2 ),
        expected_weight_count: 36 + 2
    });

    test( T {
        input: Shape::new_3d( 1, 2, 2 ),
        filter_count: 1,
        kernel_size: (1, 2),
        expected_output: Shape::new_3d( 1, 1, 1 ),
        expected_weight_count: 5
    });

    test( T {
        input: Shape::new_3d( 2, 2, 1 ),
        filter_count: 1,
        kernel_size: (2, 1),
        expected_output: Shape::new_3d( 1, 2, 1 ),
        expected_weight_count: 3
    });
}
