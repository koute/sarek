use {
    std::{
        fmt,
        ops::{
            Deref
        },
        sync::{
            Arc
        }
    },
    decorum,
    crate::{
        core::{
            data_source::{
                DataSource
            },
            data_type::{
                DataType,
                Type
            },
            name::{
                Name
            },
            shape::{
                Shape
            },
            slice_source::{
                SliceSource
            },
            utils::{
                SliceDebug
            }
        },
        nn::{
            activation::{
                Activation
            },
            model::{
                BinaryLayer,
                NullaryLayer,
                UnaryLayer
            }
        }
    }
};

fn check_for_invalid_weights( weights: &[f32] ) {
    assert!(
        !weights.iter().cloned().any( |value| value.is_nan() || value.is_infinite() ),
        "Weights contain either a NaN or an Inf"
    );
}

#[derive(Clone, PartialEq)]
pub struct Weights( Arc< Vec< f32 > > );

impl Deref for Weights {
    type Target = [f32];
    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl Eq for Weights {}
impl fmt::Debug for Weights {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "{:?}", SliceDebug( &self.0 ) )
    }
}

impl From< Vec< f32 > > for Weights {
    fn from( weights: Vec< f32 > ) -> Self {
        check_for_invalid_weights( &weights );
        Weights( Arc::new( weights ) )
    }
}

impl< 'a > From< &'a [f32] > for Weights {
    fn from( weights: &'a [f32] ) -> Self {
        check_for_invalid_weights( &weights );
        Weights( Arc::new( weights.into() ) )
    }
}

impl Weights {
    pub(crate) fn new() -> Self {
        Weights( Arc::new( Vec::new() ) )
    }

    pub(crate) fn get_mut( &mut self ) -> &mut Vec< f32 > {
        Arc::get_mut( &mut self.0 ).unwrap()
    }
}

pub trait LayerPrototype: Sized {
    fn name( &self ) -> &Name;
    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name >;
    fn with_name< T >( mut self, name: T ) -> Self where T: Into< Name > {
        self.set_name( name );
        self
    }

    fn set_weights( &mut self, weights: Weights ) -> &mut Self {
        if weights.is_empty() {
            return self;
        }

        panic!( "Weights passed to a layer which has no weights" );
    }

    fn with_weights( mut self, weights: Weights ) -> Self {
        self.set_weights( weights );
        self
    }

    fn take_weights( &mut self ) -> Option< Weights > {
        None
    }

    fn type_name( &self ) -> &'static str {
        let mut name = unsafe {
            std::intrinsics::type_name::< Self >()
        };

        if let Some( index ) = name.rfind( "::" ) {
            name = &name[ index + 2.. ];
        }

        if name.starts_with( "Layer" ) {
            name = &name[ 5.. ];
        }

        name
    }
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

    pub fn with_activation( mut self, value: Activation ) -> Self {
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
}

impl UnaryLayer for LayerActivation {
    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerAdd {
    pub(crate) name: Name
}

impl LayerAdd {
    pub fn new() -> Self {
        LayerAdd {
            name: Name::new_unique()
        }
    }
}

impl LayerPrototype for LayerAdd {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }
}

impl BinaryLayer for LayerAdd {
    fn output_shape( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> Shape {
        assert_eq!( input_shape_1, input_shape_2 );
        input_shape_1.clone()
    }

    fn weight_count( &self, _: &Shape, _: &Shape ) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct LayerConstant {
    pub(crate) name: Name,
    pub(crate) data: Arc< dyn DataSource >
}

struct DataSourceDebug< 'a >( &'a dyn DataSource );
impl< 'a > fmt::Debug for DataSourceDebug< 'a > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "DataSource" )
            .field( "len", &self.0.len() )
            .field( "shape", &self.0.shape() )
            .field( "data_type", &self.0.data_type() )
            .finish()
    }
}

impl PartialEq for LayerConstant {
    fn eq( &self, rhs: &LayerConstant ) -> bool {
        self.name == rhs.name &&
        Arc::ptr_eq( &self.data, &rhs.data )
    }
}
impl Eq for LayerConstant {}
impl fmt::Debug for LayerConstant {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt.debug_struct( "LayerConstant" )
            .field( "name", &self.name )
            .field( "data", &DataSourceDebug( &*self.data ) )
            .finish()
    }
}

impl LayerConstant {
    pub fn new< S >( data: S ) -> Self where S: DataSource + 'static {
        LayerConstant {
            name: Name::new_unique(),
            data: Arc::new( data )
        }
    }

    pub fn from_slice< T >( shape: Shape, slice: &[T] ) -> Self where T: DataType + 'static {
        let data: Vec< _ > = slice.to_vec();
        Self::new( SliceSource::from( shape, data ) )
    }
}

impl LayerPrototype for LayerConstant {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }
}

impl NullaryLayer for LayerConstant {
    fn output_shape( &self ) -> Shape {
        self.data.shape()
    }

    fn output_type( &self ) -> Type {
        self.data.data_type()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerConvolution {
    pub(crate) name: Name,
    pub(crate) filter_count: usize,
    pub(crate) kernel_size: (usize, usize),
    pub(crate) weights: Option< Weights >
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
}

impl LayerPrototype for LayerConvolution {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }

    fn set_weights( &mut self, weights: Weights ) -> &mut Self {
        self.weights = Some( weights );
        self
    }

    fn take_weights( &mut self ) -> Option< Weights > {
        self.weights.take()
    }
}

impl UnaryLayer for LayerConvolution {
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
    pub(crate) size: usize,
    pub(crate) weights: Option< Weights >
}

impl LayerDense {
    pub fn new( size: usize ) -> LayerDense {
        LayerDense {
            name: Name::new_unique(),
            size,
            weights: None
        }
    }

    pub fn size( &self ) -> usize {
        self.size
    }

    pub fn with_size( mut self, value: usize ) -> Self {
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

    fn set_weights( &mut self, weights: Weights ) -> &mut Self {
        self.weights = Some( weights );
        self
    }

    fn take_weights( &mut self ) -> Option< Weights > {
        self.weights.take()
    }
}

impl UnaryLayer for LayerDense {
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
}

impl UnaryLayer for LayerDropout {
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
}

impl UnaryLayer for LayerIntoCategory {
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
}

impl UnaryLayer for LayerMaxPooling {
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayerMul {
    pub(crate) name: Name
}

impl LayerMul {
    pub fn new() -> Self {
        LayerMul {
            name: Name::new_unique()
        }
    }
}

impl LayerPrototype for LayerMul {
    fn name( &self ) -> &Name {
        &self.name
    }

    fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
        self.name = name.into();
        self
    }
}

impl BinaryLayer for LayerMul {
    fn output_shape( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> Shape {
        assert_eq!( input_shape_1, input_shape_2 );
        input_shape_1.clone()
    }

    fn weight_count( &self, _: &Shape, _: &Shape ) -> usize {
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
}

impl UnaryLayer for LayerReshape {
    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        assert_eq!( input_shape.product(), self.shape.product() );
        self.shape.clone()
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
}

impl UnaryLayer for LayerSoftmax {
    fn output_shape( &self, input_shape: &Shape ) -> Shape {
        input_shape.clone()
    }

    fn weight_count( &self, _: &Shape ) -> usize {
        0
    }
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum AnyNullaryLayer {
    Constant( LayerConstant )
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum AnyUnaryLayer {
    Activation( LayerActivation ),
    Convolution( LayerConvolution ),
    Dense( LayerDense ),
    Dropout( LayerDropout ),
    IntoCategory( LayerIntoCategory ),
    MaxPooling( LayerMaxPooling ),
    Reshape( LayerReshape ),
    Softmax( LayerSoftmax )
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum AnyBinaryLayer {
    Add( LayerAdd ),
    Mul( LayerMul )
}

macro_rules! layer_boilerplate {
    (@impl_layer_prototype $enum_ty:ident $($variant:ident( $name:ident ))*) => {
        impl LayerPrototype for $enum_ty {
            fn name( &self ) -> &Name {
                match *self {
                    $( $enum_ty::$variant( ref layer ) => layer.name(), )*
                }
            }

            fn set_name< T >( &mut self, name: T ) -> &mut Self where T: Into< Name > {
                match *self {
                    $(
                        $enum_ty::$variant( ref mut layer ) => {
                            layer.set_name( name );
                        },
                    )*
                }

                self
            }

            fn take_weights( &mut self ) -> Option< Weights > {
                match *self {
                    $( $enum_ty::$variant( ref mut layer ) => layer.take_weights(), )*
                }
            }

            fn set_weights( &mut self, weights: Weights ) -> &mut Self {
                match *self {
                    $(
                        $enum_ty::$variant( ref mut layer ) => {
                            layer.set_weights( weights );
                        },
                    )*
                }

                self
            }

            fn type_name( &self ) -> &'static str {
                match *self {
                    $( $enum_ty::$variant( ref layer ) => layer.type_name(), )*
                }
            }
        }
    };

    ($(AnyNullaryLayer::$variant:ident( $name:ident ))*) => {
        layer_boilerplate!( @impl_layer_prototype AnyNullaryLayer $($variant( $name ))* );

        impl NullaryLayer for AnyNullaryLayer {
            fn output_shape( &self ) -> Shape {
                match *self {
                    $( AnyNullaryLayer::$variant( ref layer ) => layer.output_shape(), )*
                }
            }

            fn output_type( &self ) -> Type {
                match *self {
                    $( AnyNullaryLayer::$variant( ref layer ) => layer.output_type(), )*
                }
            }
        }

        $(
            impl From< $name > for AnyNullaryLayer {
                #[inline]
                fn from( layer: $name ) -> Self {
                    AnyNullaryLayer::$variant( layer )
                }
            }
        )*
    };

    ($(AnyUnaryLayer::$variant:ident( $name:ident ))*) => {
        layer_boilerplate!( @impl_layer_prototype AnyUnaryLayer $($variant( $name ))* );

        impl UnaryLayer for AnyUnaryLayer {
            fn output_shape( &self, input_shape: &Shape ) -> Shape {
                match *self {
                    $( AnyUnaryLayer::$variant( ref layer ) => layer.output_shape( input_shape ), )*
                }
            }

            fn weight_count( &self, input_shape: &Shape ) -> usize {
                match *self {
                    $( AnyUnaryLayer::$variant( ref layer ) => layer.weight_count( input_shape ), )*
                }
            }
        }

        $(
            impl From< $name > for AnyUnaryLayer {
                #[inline]
                fn from( layer: $name ) -> Self {
                    AnyUnaryLayer::$variant( layer )
                }
            }
        )*
    };

    ($(AnyBinaryLayer::$variant:ident( $name:ident ))*) => {
        layer_boilerplate!( @impl_layer_prototype AnyBinaryLayer $($variant( $name ))* );

        impl BinaryLayer for AnyBinaryLayer {
            fn output_shape( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> Shape {
                match *self {
                    $( AnyBinaryLayer::$variant( ref layer ) => layer.output_shape( input_shape_1, input_shape_2 ), )*
                }
            }

            fn weight_count( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> usize {
                match *self {
                    $( AnyBinaryLayer::$variant( ref layer ) => layer.weight_count( input_shape_1, input_shape_2 ), )*
                }
            }
        }

        $(
            impl From< $name > for AnyBinaryLayer {
                #[inline]
                fn from( layer: $name ) -> Self {
                    AnyBinaryLayer::$variant( layer )
                }
            }
        )*
    };
}

layer_boilerplate!(
    AnyNullaryLayer::Constant( LayerConstant )
);

layer_boilerplate!(
    AnyUnaryLayer::Activation( LayerActivation )
    AnyUnaryLayer::Convolution( LayerConvolution )
    AnyUnaryLayer::Dense( LayerDense )
    AnyUnaryLayer::Dropout( LayerDropout )
    AnyUnaryLayer::IntoCategory( LayerIntoCategory )
    AnyUnaryLayer::MaxPooling( LayerMaxPooling )
    AnyUnaryLayer::Reshape( LayerReshape )
    AnyUnaryLayer::Softmax( LayerSoftmax )
);

layer_boilerplate!(
    AnyBinaryLayer::Add( LayerAdd )
    AnyBinaryLayer::Mul( LayerMul )
);

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
