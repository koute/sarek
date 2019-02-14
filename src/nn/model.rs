use {
    crate::{
        core::{
            data_type::{
                Type
            },
            name::{
                Name
            },
            shape::{
                Shape
            }
        },
        nn::{
            layers::{
                IntoLayerIter,
                Layer,
                LayerPrototype,
                Weights
            }
        }
    },
    std::{
        collections::{
            HashMap
        }
    }
};

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum InvalidModelError {
    #[display(fmt =
        "layer #{} ({}) has the same name as layer #{}: '{}'",
        "layer_index_1",
        "layer_kind_1",
        "layer_index_2",
        "layer_name"
    )]
    DuplicateName {
        layer_index_1: usize,
        layer_kind_1: &'static str,
        layer_index_2: usize,
        layer_name: Name
    },

    #[display(fmt =
        "layer #{} (Reshape) '{}' has an output shape of {} ({}) which is incompatible with its input shape of {} ({})",
        "layer_index",
        "layer_name",
        "output_shape",
        "output_shape.product()",
        "input_shape",
        "input_shape.product()"
    )]
    InvalidReshape {
        layer_index: usize,
        layer_name: Name,
        input_shape: Shape,
        output_shape: Shape
    },

    #[display(fmt =
        "layer #{} ({}) '{}' expects an input which has {} values while its input has shape of {} hence contains {} values",
        "layer_index",
        "layer_kind",
        "layer_name",
        "expected_value_count",
        "input_shape",
        "input_shape.product()"
    )]
    InputProductMismatch {
        layer_index: usize,
        layer_kind: &'static str,
        layer_name: Name,
        input_shape: Shape,
        expected_value_count: usize
    },

    #[display(fmt =
        "layer #{} ({}) '{}' is only supported when it's the last layer in the model",
        "layer_index",
        "layer_kind",
        "layer_name"
    )]
    LayerShouldBeTheLastLayer {
        layer_index: usize,
        layer_kind: &'static str,
        layer_name: Name
    },

    #[display(fmt =
        "layer #{} ({}) '{}' is missing weights",
        "layer_index",
        "layer_kind",
        "layer_name"
    )]
    InvalidWeightCount {
        layer_index: usize,
        layer_kind: &'static str,
        layer_name: Name,
        weight_count: usize,
        expected_weight_count: usize
    },

    #[display(fmt =
        "layer #{} ({}) '{}' is missing weights",
        "layer_index",
        "layer_kind",
        "layer_name"
    )]
    MissingWeights {
        layer_index: usize,
        layer_kind: &'static str,
        layer_name: Name
    },

    #[display(fmt =
        "layer #{} ({}) '{}' was assigned weights which contain either a NaN or an Inf",
        "layer_index",
        "layer_kind",
        "layer_name"
    )]
    InvalidWeights{
        layer_index: usize,
        layer_kind: &'static str,
        layer_name: Name
    }
}

/// A neural network model.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) layers: Vec< Layer >,
    input_shape: Shape
}

impl Model {
    pub fn new_sequential< S, I >( input_shape: S, layers: I ) -> Model
        where S: Into< Shape >,
              I: IntoLayerIter
    {
        Model {
            layers: layers.into_layer_iter().collect(),
            input_shape: input_shape.into()
        }
    }

    pub fn input_shape( &self ) -> Shape {
        self.input_shape.clone()
    }

    pub fn input_data_type( &self ) -> Type {
        Type::F32
    }

    pub fn output_data_type( &self ) -> Type {
        match self.layers.last() {
            Some( Layer::IntoCategory( .. ) ) => Type::U32,
            _ => Type::F32
        }
    }

    pub(crate) fn get_layer_and_input_shape( &self, layer_name: &Name ) -> Option< (&Layer, Shape) > {
        let mut input_shape = self.input_shape();
        let mut target_layer = None;
        for layer in &self.layers {
            if *layer.name() == *layer_name {
                target_layer = Some( layer );
                break;
            }

            input_shape = layer.output_shape( &input_shape );
        }

        target_layer.map( |layer| (layer, input_shape) )
    }

    pub fn output_shape( &self ) -> Shape {
        let mut input_shape = self.input_shape();
        for layer in &self.layers {
            input_shape = layer.output_shape( &input_shape );
        }
        input_shape
    }

    pub(crate) fn validate( &self ) -> Result< (), InvalidModelError > {
        fn check_weights(
            layer_index: usize,
            layer_kind: &'static str,
            name: &Name,
            weights: &Option< Weights >,
            expected_weight_count: usize
        ) -> Result< (), InvalidModelError >
        {
            let weights = weights.as_ref()
                .ok_or_else( || InvalidModelError::MissingWeights {
                    layer_index, layer_kind, layer_name: name.clone()
                })?;

            if weights.iter().cloned().any( |value| value.is_nan() || value.is_infinite() ) {
                return Err( InvalidModelError::InvalidWeights { layer_index, layer_kind, layer_name: name.clone() } )
            }

            if weights.len() != expected_weight_count {
                return Err( InvalidModelError::InvalidWeightCount {
                    layer_index,
                    layer_kind,
                    layer_name: name.clone(),
                    weight_count: weights.len(),
                    expected_weight_count
                });
            }

            Ok(())
        }

        fn check_arithmetic_layer(
            layer_index: usize,
            layer_kind: &'static str,
            layer_name: &Name,
            input_shape: &Shape,
            expected_value_count: usize
        ) -> Result< (), InvalidModelError >
        {
            if expected_value_count!= input_shape.product() {
                return Err( InvalidModelError::InputProductMismatch {
                    layer_index,
                    layer_kind,
                    layer_name: layer_name.clone(),
                    input_shape: input_shape.clone(),
                    expected_value_count
                });
            }

            Ok(())
        }

        let mut name_to_index = HashMap::new();
        let mut input_shape = self.input_shape();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let is_last = layer_index + 1 == self.layers.len();
            let weight_count = layer.weight_count( &input_shape );
            let output_shape = layer.output_shape( &input_shape );
            if let Some( &other_layer_index ) = name_to_index.get( layer.name() ) {
                return Err( InvalidModelError::DuplicateName {
                    layer_index_1: layer_index,
                    layer_kind_1: layer.type_name(),
                    layer_index_2: other_layer_index,
                    layer_name: layer.name().clone()
                });
            }
            name_to_index.insert( layer.name(), layer_index );

            let layer_kind = layer.type_name();
            match layer {
                Layer::Dense( layer ) => {
                    check_weights( layer_index, layer_kind, &layer.name, &layer.weights, weight_count )?;
                },
                Layer::Convolution( layer ) => {
                    check_weights( layer_index, layer_kind, &layer.name, &layer.weights, weight_count )?;
                },
                Layer::Multiply( layer ) => {
                    check_arithmetic_layer( layer_index, layer_kind, &layer.name, &input_shape, layer.values.len() )?;
                },
                Layer::Reshape( layer ) => {
                    if layer.shape.product() != input_shape.product() {
                        return Err( InvalidModelError::InvalidReshape {
                            layer_index,
                            layer_name: layer.name().clone(),
                            input_shape,
                            output_shape: layer.shape.clone()
                        });
                    }
                },
                Layer::Shift( layer ) => {
                    check_arithmetic_layer( layer_index, layer_kind, &layer.name, &input_shape, layer.values.len() )?;
                },
                Layer::IntoCategory( _ ) => {
                    if !is_last {
                        return Err( InvalidModelError::LayerShouldBeTheLastLayer {
                            layer_index,
                            layer_kind,
                            layer_name: layer.name().clone()
                        });
                    }
                },
                _ => {}
            }
            input_shape = output_shape;
        }

        Ok(())
    }
}
