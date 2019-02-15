use {
    pyo3::{
        prelude::*,
        types::{
            PyDict,
            PyList,
            PyObjectRef,
            PyTuple
        }
    },
    crate::{
        backend::{
            GetWeightsError,
            ModelCompilationError,
            SetWeightsError,
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
                    PyResultExt,
                    py_err
                }
            },
            model::{
                ModelInstanceState,
                OutputKind
            }
        },
        core::{
            array::{
                ToArrayRef
            },
            data_source::{
                DataSource
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
                LayerConvolution,
                LayerDense,
                LayerDropout,
                LayerMultiply,
                LayerPrototype,
                LayerReshape,
                LayerShift,
                LayerSoftmax
            },
            optimizers::{
                Optimizer,
                OptimizerSGD,
                OptimizerNadam
            },
            training_opts::{
                TrainingOpts
            }
        }
    }
};

/// A compiled `Model`.
pub struct ModelInstance {
    _ctx: Context,
    obj: PyObject
}

fn constant_layer< 'a >( py: Python< 'a >, input_shape: &Shape, values: &[f32] ) -> &'a PyObjectRef {
    let mut array = TypedPyArray::< f32 >::new( py, input_shape.prepend( 1 ) );
    array.as_slice_mut().copy_from_slice( values );

    let tf_ns = py.import( "tensorflow" ).unwrap_py( py );
    let keras_ns = tf_ns.getattr( "keras" ).unwrap_py( py );
    let layers_ns = keras_ns.getattr( "layers" ).unwrap_py( py );
    let tensor = tf_ns.getattr( "constant" ).unwrap_py( py ).call( (array.as_py_obj(),), None ).unwrap_py( py );
    let kwargs = PyDict::new( py );
    kwargs.set_item( "tensor", tensor ).unwrap_py( py );
    layers_ns.getattr( "Input" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py )
}

impl ModelInstance {
    pub(crate) fn compile(
        ctx: &Context,
        state: &mut ModelInstanceState,
        training_opts: Option< TrainingOpts >
    ) -> Result< ModelInstance, ModelCompilationError >
    {
        Context::gil( move |py| {
            let tf_ns = py.import( "tensorflow" ).unwrap();
            let keras_ns = tf_ns.getattr( "keras" ).unwrap();
            let layers_ns = keras_ns.getattr( "layers" ).unwrap();
            let optimizers_ns = keras_ns.getattr( "optimizers" ).unwrap();
            let initializers_ns = keras_ns.getattr( "initializers" ).unwrap();
            let constant = initializers_ns.getattr( "constant" ).unwrap();
            let is_trainable = training_opts.is_some();
            let mut input_shape = state.model.input_shape();

            let initial_layer = {
                let kwargs = PyDict::new( py );
                kwargs.set_item( "shape", PyTuple::new( py, &input_shape ) ).unwrap();
                keras_ns.getattr( "Input" ).unwrap().call( (), Some( kwargs ) ).unwrap()
            };

            let model_inputs = PyList::new( py, &[initial_layer.clone()] );
            let mut last_layer = initial_layer.clone();

            let model_layer_count = state.model.layers.len();
            for (layer_index, layer) in state.model.layers.iter_mut().enumerate() {
                let mut kwargs = PyDict::new( py );

                let is_last = layer_index + 1 == model_layer_count;
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
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    },
                    Layer::Convolution( layer ) => {
                        if input_shape.dimension_count() == 2 {
                            let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            kwargs.set_item( "trainable", is_trainable ).unwrap();
                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            last_layer = layer.call( (last_layer,), None ).unwrap();

                            kwargs = PyDict::new( py );
                        }

                        kwargs.set_item( "name", layer.name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        kwargs.set_item( "filters", layer.filter_count ).unwrap();
                        kwargs.set_item( "kernel_size", (layer.kernel_size.1, layer.kernel_size.0) ).unwrap();

                        if let Some( weights ) = layer.weights.take() {
                            let (bias_array, weight_array) =
                                Self::weights_into_arrays_for_convolutional_layer( py, &input_shape, &layer, &weights );

                            let bias_array = constant.call( (bias_array.as_py_obj(),), None ).unwrap();
                            let weight_array = constant.call( (weight_array.as_py_obj(),), None ).unwrap();

                            kwargs.set_item( "bias_initializer", bias_array ).unwrap();
                            kwargs.set_item( "kernel_initializer", weight_array ).unwrap();
                        }

                        let layer_obj = layers_ns.getattr( "Conv2D" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                        last_layer = layer_obj.call( (last_layer,), None ).unwrap();
                        {
                            let target_shape = layer.output_shape( &input_shape );
                            let target_shape = PyTuple::new( py, &target_shape );
                            kwargs = PyDict::new( py );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            let layer_obj = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                            last_layer = layer_obj.call( (last_layer,), None ).unwrap();
                        }
                    },
                    Layer::Dense( layer ) => {
                        {
                            // Dense layers expect a one dimensional input.
                            let layer = layers_ns.getattr( "Flatten" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            last_layer = layer.call( (last_layer,), None ).unwrap();

                            kwargs = PyDict::new( py );
                        }

                        if let Some( weights ) = layer.weights.take() {
                            let (bias_array, weight_array) =
                                Self::weights_into_arrays_for_dense_layer( py, &input_shape, &layer, &weights );

                            let bias_array = constant.call( (bias_array.as_py_obj(),), None ).unwrap();
                            let weight_array = constant.call( (weight_array.as_py_obj(),), None ).unwrap();

                            kwargs.set_item( "bias_initializer", bias_array ).unwrap();
                            kwargs.set_item( "kernel_initializer", weight_array ).unwrap();
                        }

                        kwargs.set_item( "name", layer.name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Dense" ).unwrap().call( (layer.size,), Some( kwargs ) ).unwrap();
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    },
                    Layer::Dropout( LayerDropout { name, rate } ) => {
                        let rate: f32 = rate.clone().into();
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Dropout" ).unwrap().call( (rate,), Some( kwargs ) ).unwrap();
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    },
                    Layer::IntoCategory( _ ) => {
                        assert!( is_last, "The `LayerIntoCategory` is only supported as the last layer of the network" );
                        let layer = layers_ns.getattr( "Flatten" ).unwrap().call( (), None ).unwrap();
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    },
                    Layer::MaxPooling( layer ) => {
                        if input_shape.dimension_count() == 2 {
                            let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            kwargs.set_item( "trainable", is_trainable ).unwrap();
                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            last_layer = layer.call( (last_layer,), None ).unwrap();

                            kwargs = PyDict::new( py );
                        }

                        kwargs.set_item( "pool_size", (layer.pool_size.1, layer.pool_size.0) ).unwrap();
                        kwargs.set_item( "name", layer.name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        // Padding is explained here:
                        //   https://stackoverflow.com/questions/37674306
                        kwargs.set_item( "padding", "same" ).unwrap();
                        let layer_obj = layers_ns.getattr( "MaxPool2D" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                        last_layer = layer_obj.call( (last_layer,), None ).unwrap();
                        {
                            let target_shape = layer.output_shape( &input_shape );
                            let target_shape = PyTuple::new( py, &target_shape );
                            kwargs = PyDict::new( py );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            let layer_obj = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                            last_layer = layer_obj.call( (last_layer,), None ).unwrap();
                        }
                    },
                    Layer::Reshape( LayerReshape { name, shape } ) => {
                        let target_shape = PyTuple::new( py, shape );
                        kwargs.set_item( "target_shape", target_shape ).unwrap();
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    },
                    ref layer @ Layer::Multiply( .. ) |
                    ref layer @ Layer::Shift( .. ) => {
                        let (kind, name, values) = match layer {
                            Layer::Multiply( LayerMultiply { name, values } ) => ("Multiply", name, values),
                            Layer::Shift( LayerShift { name, values } ) => ("Add", name, values),
                            _ => unreachable!()
                        };

                        let extra_layer = constant_layer( py, &input_shape, &values );
                        model_inputs.append( extra_layer ).unwrap_py( py );

                        kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                        let layer = layers_ns.getattr( kind ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                        let inputs = PyList::new( py, &[last_layer, extra_layer] );
                        last_layer = layer.call( (inputs,), None ).unwrap_py( py );
                    },
                    Layer::Softmax( LayerSoftmax { name } ) => {
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Activation" ).unwrap().call( ("softmax",), Some( kwargs ) ).unwrap();
                        last_layer = layer.call( (last_layer,), None ).unwrap();
                    }
                }

                input_shape = layer.output_shape( &input_shape );
            }

            if last_layer == initial_layer {
                let kwargs = PyDict::new( py );
                let shape = PyTuple::new( py, &input_shape );
                kwargs.set_item( "target_shape", shape ).unwrap();

                let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                last_layer = layer.call( (last_layer,), None ).unwrap_py( py );
            }

            let kwargs = PyDict::new( py );
            kwargs.set_item( "inputs", model_inputs ).unwrap();
            kwargs.set_item( "outputs", last_layer ).unwrap();
            let model_obj = keras_ns.getattr( "Model" ).unwrap().call( (), Some( kwargs ) )
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
                        Optimizer::Nadam( OptimizerNadam { learning_rate } ) => {
                            let kwargs = PyDict::new( py );
                            let learning_rate: f32 = learning_rate.into();
                            kwargs.set_item( "lr", learning_rate ).unwrap();
                            optimizers_ns.getattr( "Nadam" ).unwrap().call( (), Some( kwargs ) ).unwrap()
                        }
                    };
                    compile_kwargs.set_item( "optimizer", optimizer ).unwrap();
                }

                let loss = match state.output_kind {
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
                obj: model_obj.to_object( py )
            };

            Ok( instance )
        })
    }

    pub(crate) fn set_weights(
        &mut self,
        input_shape: &Shape,
        layer: &Layer,
        weights: &[f32]
    ) -> Result< (), SetWeightsError >
    {
        Context::gil( |py| {
            let weights = match layer {
                Layer::Convolution( layer ) => {
                    let (bias_array, weight_array) = Self::weights_into_arrays_for_convolutional_layer( py, input_shape, layer, weights );
                    let list = PyList::empty( py );
                    list.append( weight_array ).unwrap();
                    list.append( bias_array ).unwrap();
                    list
                },
                Layer::Dense( layer ) => {
                    let (bias_array, weight_array) = Self::weights_into_arrays_for_dense_layer( py, input_shape, layer, weights );
                    let list = PyList::empty( py );
                    list.append( weight_array ).unwrap();
                    list.append( bias_array ).unwrap();
                    list
                },
                Layer::Activation( _ ) |
                Layer::Dropout( _ ) |
                Layer::IntoCategory( _ ) |
                Layer::MaxPooling( _ ) |
                Layer::Multiply( _ ) |
                Layer::Reshape( _ ) |
                Layer::Shift( _ ) |
                Layer::Softmax( _ )
                    => unreachable!()
            };

            let layer_name = layer.name().to_string();
            let layer = self.obj.getattr( py, "get_layer" ).unwrap().call( py, (layer_name,), None ).unwrap();
            layer.getattr( py, "set_weights" ).unwrap().call( py, (weights,), None ).map_err( |err| py_err( py, err ) ).unwrap();
        });

        Ok(())
    }

    fn weights_into_arrays_for_convolutional_layer(
        py: Python,
        input_shape: &Shape,
        layer: &LayerConvolution,
        weights: &[f32]
    ) -> (TypedPyArray< f32 >, TypedPyArray< f32 >) {
        let bias_count = layer.filter_count;
        let weight_shape = Shape::new_4d( layer.kernel_size.1, layer.kernel_size.0, input_shape.z(), layer.filter_count );

        let mut weight_array = TypedPyArray::< f32 >::new( py, weight_shape );
        let mut bias_array = TypedPyArray::< f32 >::new( py, Shape::new_1d( bias_count ) );
        weight_array.as_slice_mut().copy_from_slice( &weights[ bias_count.. ] );
        bias_array.as_slice_mut().copy_from_slice( &weights[ ..bias_count ] );

        (bias_array, weight_array)
    }

    fn weights_into_arrays_for_dense_layer(
        py: Python,
        input_shape: &Shape,
        layer: &LayerDense,
        weights: &[f32]
    ) -> (TypedPyArray< f32 >, TypedPyArray< f32 >) {
        let bias_count = layer.size;

        let mut weight_array = TypedPyArray::< f32 >::new( py, Shape::new_2d( input_shape.product(), layer.size ) );
        let mut bias_array = TypedPyArray::< f32 >::new( py, Shape::new_1d( bias_count ) );
        weight_array.as_slice_mut().copy_from_slice( &weights[ bias_count.. ] );
        bias_array.as_slice_mut().copy_from_slice( &weights[ ..bias_count ] );

        (bias_array, weight_array)
    }

    pub(crate) fn get_weights( &self, input_shape: &Shape, layer: &Layer )
        -> Result< impl ToArrayRef + DataSource, GetWeightsError >
    {
        let weight_count = layer.weight_count( &input_shape );
        let layer_name_s = layer.name().to_string();
        let output = Context::gil( move |py| {
            let layer = self.obj.getattr( py, "get_layer" ).unwrap().call( py, (layer_name_s,), None ).unwrap();
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

        Ok( SliceSource::from( Shape::new_1d( weight_count ), output ) )
    }

    fn train_on_batch( &mut self, py: Python, inputs: &PyArray, outputs: &PyArray ) -> f32 {
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

    pub(crate) fn train_for_epoch< F >( &mut self, state: &ModelInstanceState, batch_size: usize, mut fill_data: F ) -> f32
        where F: FnMut( &mut [u8], &mut [u8] ) -> bool + Send
    {
        let input_shape = state.model.input_shape();
        let output_shape = state.model.output_shape();
        let input_data_type = state.model.input_data_type();
        let output_data_type = state.model.output_data_type();

        Context::gil( move |py| {
            let mut inputs = PyArray::new( py, input_shape.prepend( batch_size ), input_data_type );
            let mut outputs = PyArray::new( py, output_shape.prepend( batch_size ), output_data_type );

            let mut loss = 0.0;
            loop {
                let should_train = fill_data( inputs.as_bytes_mut(), outputs.as_bytes_mut() );
                if should_train {
                    loss += self.train_on_batch( py, &inputs, &outputs );
                } else {
                    break;
                }
            }

            loss
        })
    }

    pub(crate) fn predict_raw< I >( &mut self, _state: &ModelInstanceState, input_data: &I ) -> RawArraySource where I: DataSource + Sync {
        let input_shape = input_data.shape();
        Context::gil( move |py| {
            let mut inputs = PyArray::new( py, input_shape.prepend( input_data.len() ), input_data.data_type() );
            input_data.gather_bytes_into( .., inputs.as_bytes_mut() );

            let result = self.obj.getattr( py, "predict" ).unwrap().call( py, (inputs.as_py_obj(),), None ).map_err( |err| py_err( py, err ) ).unwrap();
            let result = unsafe { PyArray::from_object_unchecked( py, result ) };
            result.into_raw_array()
        })
    }
}
