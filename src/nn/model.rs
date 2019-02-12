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
                LayerPrototype
            }
        }
    }
};

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
}
