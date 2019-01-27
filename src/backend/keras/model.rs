use {
    pyo3::{
        prelude::*,
        types::{
            PyDict,
            PyList,
            PyTuple
        }
    },
    crate::{
        backend::{
            keras::{
                context::{
                    Context
                },
                loss::{
                    LossKind
                },
                py_array::{
                    PyArray,
                    TypedPyArray
                },
                py_utils::{
                    py_err
                }
            }
        },
        core::{
            array::{
                ToArrayRef
            },
            data_source::{
                DataSource
            },
            data_type::{
                Type,
                cast_slice_mut
            },
            name::{
                Name
            },
            raw_array_source::{
                RawArraySource
            },
            shape::{
                Shape
            },
            slice_source::{
                SliceSource
            }
        },
        nn::{
            activation::{
                Activation
            },
            layers::{
                Layer,
                LayerActivation,
                LayerDense,
                LayerDropout,
                LayerPrototype,
                LayerSoftmax
            },
            model::{
                Model
            },
            optimizers::{
                Optimizer,
                OptimizerAdam,
                OptimizerSGD
            },
            training_opts::{
                TrainingOpts
            }
        }
    }
};

enum OutputKind {
    Regression,
    SparseCategory
}

/// A compiled `Model`.
pub struct ModelInstance {
    _ctx: Context,
    obj: PyObject,
    model: Model,
    output_kind: OutputKind
}

#[derive(Debug)]
pub struct ModelCompilationError(());

#[derive(Debug)]
pub struct SetWeightsError(());

impl ModelInstance {
    pub fn new( ctx: &Context, model: Model ) -> Result< ModelInstance, ModelCompilationError > {
        Self::compile( ctx, model, None )
    }

    pub(crate) fn compile( ctx: &Context, model: Model, training_opts: Option< TrainingOpts > ) -> Result< ModelInstance, ModelCompilationError > {
        Context::gil( move |py| {
            let tf_ns = py.import( "tensorflow" ).unwrap();
            let keras_ns = tf_ns.getattr( "keras" ).unwrap();
            let layers_ns = keras_ns.getattr( "layers" ).unwrap();
            let models_ns = keras_ns.getattr( "models" ).unwrap();
            let optimizers_ns = keras_ns.getattr( "optimizers" ).unwrap();
            let is_trainable = training_opts.is_some();
            let mut output_kind = OutputKind::Regression;
            let mut is_first = true;

            let mut layers = Vec::with_capacity( model.layers.len() );
            for (layer_index, layer) in model.layers.iter().enumerate() {
                let mut kwargs = PyDict::new( py );

                if is_first {
                    let shape = PyTuple::new( py, &model.input_shape() );
                    kwargs.set_item( "input_shape", shape ).unwrap();
                    is_first = false;
                }

                let is_last = layer_index + 1 == model.layers.len();

                match layer {
                    Layer::Activation( LayerActivation { name, activation } ) => {
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();

                        let layer = match activation {
                            Activation::ELU =>
                                layers_ns.getattr( "Activation" ).unwrap().call( ("elu",), Some( kwargs ) ).unwrap(),
                            Activation::LeakyReLU => {
                                kwargs.set_item( "negative_slope", 0.01 ).unwrap();
                                layers_ns.getattr( "ReLU" ).unwrap().call( (), Some( kwargs ) ).unwrap()
                            },
                            Activation::Logistic =>
                                layers_ns.getattr( "Activation" ).unwrap().call( ("sigmoid",), Some( kwargs ) ).unwrap(),
                            Activation::ReLU =>
                                layers_ns.getattr( "Activation" ).unwrap().call( ("relu",), Some( kwargs ) ).unwrap(),
                            Activation::TanH =>
                                layers_ns.getattr( "Activation" ).unwrap().call( ("tanh",), Some( kwargs ) ).unwrap()
                        };
                        layers.push( layer );
                    },
                    Layer::Dense( LayerDense { name, size } ) => {
                        {
                            // Dense layers expect a one dimensional input.
                            let layer = layers_ns.getattr( "Flatten" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            layers.push( layer );

                            kwargs = PyDict::new( py );
                        }

                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Dense" ).unwrap().call( (*size,), Some( kwargs ) ).unwrap();
                        layers.push( layer );
                    },
                    Layer::Dropout( LayerDropout { name, rate } ) => {
                        let rate: f32 = rate.clone().into();
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Dropout" ).unwrap().call( (rate,), Some( kwargs ) ).unwrap();
                        layers.push( layer );
                    },
                    Layer::IntoCategory( _ ) => {
                        assert!( is_last, "The `LayerIntoCategory` is only supported as the last layer of the network" );
                        if layers.is_empty() {
                            let target_shape = PyTuple::new( py, &model.input_shape() );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();

                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            layers.push( layer );
                        }

                        output_kind = OutputKind::SparseCategory;
                    },
                    Layer::Softmax( LayerSoftmax { name } ) => {
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Activation" ).unwrap().call( ("softmax",), Some( kwargs ) ).unwrap();
                        layers.push( layer );
                    }
                }
            }

            if layers.is_empty() {
                let kwargs = PyDict::new( py );
                let shape = PyTuple::new( py, &model.input_shape() );
                kwargs.set_item( "input_shape", shape ).unwrap();
                kwargs.set_item( "trainable", is_trainable ).unwrap();

                let layer = layers_ns.getattr( "Flatten" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                layers.push( layer );
            }

            let layers = PyList::new( py, &layers );
            let model_obj = models_ns.getattr( "Sequential" ).unwrap().call( (layers,), None )
                .map_err( |err| py_err( py, err ) ).unwrap();

            if is_trainable {
                let compile_kwargs = PyDict::new( py );
                if let Some( training_opts ) = training_opts {
                    let optimizer = match training_opts.optimizer {
                        Optimizer::SGD( OptimizerSGD { learning_rate } ) => {
                            let kwargs = PyDict::new( py );
                            let learning_rate: f32 = learning_rate.into();
                            kwargs.set_item( "lr", learning_rate ).unwrap();
                            optimizers_ns.getattr( "SGD" ).unwrap().call( (), Some( kwargs ) ).unwrap()
                        },
                        Optimizer::Adam( OptimizerAdam { learning_rate } ) => {
                            let kwargs = PyDict::new( py );
                            let learning_rate: f32 = learning_rate.into();
                            kwargs.set_item( "lr", learning_rate ).unwrap();
                            optimizers_ns.getattr( "Adam" ).unwrap().call( (), Some( kwargs ) ).unwrap()
                        }
                    };
                    compile_kwargs.set_item( "optimizer", optimizer ).unwrap();
                }

                let loss = match output_kind {
                    OutputKind::Regression => LossKind::MeanSquaredError,
                    OutputKind::SparseCategory => LossKind::SparseCategoricalCrossEntropy
                };

                let loss = match loss {
                    LossKind::SparseCategoricalCrossEntropy => "sparse_categorical_crossentropy",
                    LossKind::CategoricalCrossEntropy => "categorical_crossentropy",
                    LossKind::MeanSquaredError => "mean_squared_error"
                };

                let metrics = PyList::new( py, &["accuracy"] );

                compile_kwargs.set_item( "loss", loss ).unwrap();
                compile_kwargs.set_item( "metrics", metrics ).unwrap();

                model_obj.getattr( "compile" ).unwrap().call( (), Some( compile_kwargs ) )
                    .map_err( |err| py_err( py, err ) ).unwrap();
            }

            let instance = ModelInstance {
                _ctx: ctx.clone(),
                obj: model_obj.to_object( py ),
                model,
                output_kind
            };

            Ok( instance )
        })
    }

    pub fn input_shape( &self ) -> Shape {
        self.model.input_shape()
    }

    pub fn output_shape( &self ) -> Shape {
        self.model.output_shape()
    }

    pub fn set_weights< N >( &mut self, layer_name: N, weights: &[f32] ) -> Result< (), SetWeightsError > where N: Into< Name > {
        let layer_name = layer_name.into();
        let (layer, input_shape) = match self.model.get_layer_and_input_shape( &layer_name ) {
            Some( result ) => result,
            None => {
                panic!( "Model has no layer named '{}'", layer_name );
            }
        };

        let weight_count = layer.weight_count( &input_shape );
        assert_eq!(
            weights.len(),
            weight_count,
            "Layer '{}' has {} weights, yet {} weights were passed", layer_name, weight_count, weights.len()
        );

        if weight_count == 0 {
            return Ok(());
        }

        Context::gil( |py| {
            let weights = match layer {
                Layer::Dense( layer ) => {
                    let bias_count = layer.size;

                    let list = PyList::empty( py );
                    let mut weight_array = TypedPyArray::< f32 >::new( py, Shape::new_2d( input_shape.product(), layer.size ) );
                    let mut bias_array = TypedPyArray::< f32 >::new( py, Shape::new_1d( bias_count ) );
                    weight_array.as_slice_mut().copy_from_slice( &weights[ bias_count.. ] );
                    bias_array.as_slice_mut().copy_from_slice( &weights[ ..bias_count ] );
                    list.append( weight_array ).unwrap();
                    list.append( bias_array ).unwrap();
                    list
                },
                Layer::Activation( _ ) |
                Layer::Dropout( _ ) |
                Layer::IntoCategory( _ ) |
                Layer::Softmax( _ )
                    => unreachable!()
            };

            let layer_name = layer_name.to_string();
            let layer = self.obj.getattr( py, "get_layer" ).unwrap().call( py, (layer_name,), None ).unwrap();
            layer.getattr( py, "set_weights" ).unwrap().call( py, (weights,), None ).map_err( |err| py_err( py, err ) ).unwrap();
            Ok(())
        })
    }

    pub fn get_weights< N >( &self, layer_name: N ) -> impl ToArrayRef + DataSource where N: Into< Name > {
        let layer_name = layer_name.into();
        let (layer, input_shape) = match self.model.get_layer_and_input_shape( &layer_name ) {
            Some( result ) => result,
            None => {
                panic!( "Model has no layer named '{}'", layer_name );
            }
        };

        let weight_count = layer.weight_count( &input_shape );
        let output = Context::gil( move |py| {
            let layer_name = layer_name.to_string();
            let layer = self.obj.getattr( py, "get_layer" ).unwrap().call( py, (layer_name,), None ).unwrap();
            let weights_list = layer.getattr( py, "get_weights" ).unwrap().call( py, (), None ).map_err( |err| py_err( py, err ) ).unwrap();
            let weights_list: &PyList = py.checked_cast_as( weights_list ).unwrap();
            let mut output: Vec< f32 > = Vec::with_capacity( weight_count );
            let weights_list: Vec< _ > = weights_list.iter().collect();
            for weights in weights_list.into_iter().rev() {
                let weights = weights.to_object( py );
                let weights = unsafe { PyArray::from_object_unchecked( py, weights ) };
                let weights = weights.into_typed::< f32 >().unwrap();
                output.extend( weights.as_slice() );
            }

            output
        });

        SliceSource::from( Shape::new_1d( weight_count ), output )
    }

    pub(crate) fn train_on_batch( &mut self, py: Python, inputs: &PyArray, outputs: &PyArray ) -> f32 {
        debug_assert_eq!( inputs.shape().x(), outputs.shape().x() );
        let batch_size = inputs.shape().x();

        let inputs = inputs.as_py_obj();
        let outputs = outputs.as_py_obj();
        let loss = self.obj.getattr( py, "train_on_batch" ).unwrap()
            .call( py, (inputs, outputs), None )
            .map_err( |err| py_err( py, err ) ).unwrap();

        debug_assert_eq!(
            crate::backend::keras::py_utils::py_to_string(
                py,
                self.obj.getattr( py, "metrics_names" ).unwrap().cast_as::< PyList >( py ).unwrap().get_item( 0 )
            ),
            "loss"
        );

        let loss: f32 = loss.cast_as::< PyList >( py ).unwrap()
            .get_item( 0 )
            .extract()
            .map_err( |err| py_err( py, err ) )
            .unwrap();

        // Tensorflow averages the loss it returns over the batch size for some reason;
        // we reverse it so that the losses are more comparable when changing the batch size.
        loss * (batch_size as f32)
    }

    fn predict_raw< I >( &mut self, input_data: &I ) -> RawArraySource where I: DataSource + Sync {
        let input_shape = input_data.shape();
        assert_eq!(
            self.input_shape(),
            input_shape,
            "The input data's shape is {}; expected it to be equal to the input shape of the model, which is {}",
            input_shape,
            self.input_shape()
        );

        Context::gil( move |py| {
            let mut inputs = PyArray::new( py, input_shape.prepend( input_data.len() ), input_data.data_type() );
            input_data.gather_bytes_into( .., inputs.as_bytes_mut() );

            let result = self.obj.getattr( py, "predict" ).unwrap().call( py, (inputs.as_py_obj(),), None ).map_err( |err| py_err( py, err ) ).unwrap();
            let result = unsafe { PyArray::from_object_unchecked( py, result ) };
            result.into_raw_array()
        })
    }

    pub fn predict< I >( &mut self, input_data: &I ) -> impl ToArrayRef + DataSource where I: DataSource + Sync {
        let result = self.predict_raw( input_data );
        match self.output_kind {
            OutputKind::Regression => {
                debug_assert_eq!(
                    result.shape(),
                    self.model.output_shape(),
                    "Internal error: expected the output of the network to have a shape of {}; instead it has a shape of {}",
                    self.model.output_shape(),
                    result.shape()
                );

                result
            },
            OutputKind::SparseCategory => {
                let count = result.len();
                let mut categories = RawArraySource::new_uninitialized( count, 1.into(), Type::U32 );
                let categories_slice = cast_slice_mut::< u32 >( categories.as_bytes_mut() );
                let result = result.to_typed_array_ref::< f32 >().expect( "internal error: unhandled array type" );
                for index in 0..count {
                    let category = result[ index ]
                        .iter()
                        .enumerate()
                        .max_by_key( |(_, &value)| decorum::Finite::from( value ) )
                        .unwrap()
                        .0;
                    categories_slice[ index ] = category as u32;
                }

                categories
            }
        }
    }
}
