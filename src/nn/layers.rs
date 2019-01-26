use {
    std::{
        iter,
        slice
    },
    decorum,
    crate::{
        core::{
            name::{
                Name
            },
            shape::{
                Shape
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
            activation: Activation::ReLU
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
    Dense( LayerDense ),
    Dropout( LayerDropout ),
    IntoCategory( LayerIntoCategory ),
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

layer_boilerplate!(
    Layer::Activation( LayerActivation )
    Layer::Dense( LayerDense )
    Layer::Dropout( LayerDropout )
    Layer::IntoCategory( LayerIntoCategory )
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
    (@access $this:expr, A) => { $this.0 };
    (@access $this:expr, B) => { $this.1 };
    (@access $this:expr, C) => { $this.2 };
    (@access $this:expr, D) => { $this.3 };
    (@access $this:expr, E) => { $this.4 };
    (@access $this:expr, F) => { $this.5 };
    (@access $this:expr, G) => { $this.6 };
    (@access $this:expr, H) => { $this.7 };
    (@access $this:expr, I) => { $this.8 };
    (@access $this:expr, J) => { $this.9 };
    (@access $this:expr, K) => { $this.10 };
    (@access $this:expr, L) => { $this.11 };
    (@access $this:expr, M) => { $this.12 };
    (@access $this:expr, N) => { $this.13 };
    (@access $this:expr, O) => { $this.14 };
    (@access $this:expr, P) => { $this.15 };

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
                [A B C D E F G H I J K L M N O P]
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
