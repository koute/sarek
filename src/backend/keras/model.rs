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
                    py_type_name
                }
            },
            model::{
                ModelInstanceState,
                OutputKind,
                RawBufferList
            }
        },
        core::{
            array::{
                ToArrayRef
            },
            data_source::{
                DataSource,
                DataSourceExt,
                DataSourceList,
                DataSourceListExt
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
                AnyBinaryLayer,
                AnyNullaryLayer,
                AnyUnaryLayer,
                LayerActivation,
                LayerAdd,
                LayerConstant,
                LayerConvolution,
                LayerDense,
                LayerDropout,
                LayerMul,
                LayerReshape,
                LayerSoftmax
            },
            model::{
                Node,
                Model,
                NullaryLayer,
                UnaryLayer
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

impl RawBufferList for Vec< PyArray > {
    fn get_buffer_mut( &mut self, index: usize ) -> &mut [u8] {
        self[ index ].as_bytes_mut()
    }
}

/// A compiled `Model`.
pub struct ModelInstance {
    _ctx: Context,
    obj: PyObject
}

impl ModelInstance {
    pub(crate) fn compile(
        ctx: &Context,
        state: &mut ModelInstanceState,
        training_opts: Option< TrainingOpts >
    ) -> Result< ModelInstance, ModelCompilationError >
    {
        Context::gil( move |py| {
            let tf_ns = py.import( "tensorflow" ).unwrap_py( py );
            let keras_ns = tf_ns.getattr( "keras" ).unwrap_py( py );
            let layers_ns = keras_ns.getattr( "layers" ).unwrap_py( py );
            let optimizers_ns = keras_ns.getattr( "optimizers" ).unwrap_py( py );
            let initializers_ns = keras_ns.getattr( "initializers" ).unwrap_py( py );
            let constant = initializers_ns.getattr( "constant" ).unwrap_py( py );
            let is_trainable = training_opts.is_some();
            let model_input_list = PyList::empty( py );
            let mut model_outputs = Vec::with_capacity( state.model.outputs().len() );
            for _ in 0..state.model.outputs().len() {
                model_outputs.push( None );
            }

            let mut model_extra_inputs = Vec::new();

            let _: Result< (), () > = state.model.traverse_mut( |model, inputs, node_index| {
                let input_shapes: Vec< _ > = model
                    .get_node( node_index )
                    .inputs()
                    .map( |node_index| model.get_node( node_index ).output_shape().clone() )
                    .collect();

                let output_type = model.get_node( node_index ).output_type();
                let mut input_node_shape = None;
                let layer = match *model.get_node_mut( node_index ) {
                    Node::Input { input_index, ref shape, .. } => {
                        assert_eq!( input_index, model_input_list.len() );

                        let kwargs = PyDict::new( py );
                        kwargs.set_item( "shape", PyTuple::new( py, shape ) ).unwrap_py( py );
                        kwargs.set_item( "dtype", py_type_name( output_type ) ).unwrap_py( py );

                        let layer = keras_ns.getattr( "Input" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                        model_input_list.append( layer ).unwrap_py( py );
                        input_node_shape = Some( shape.clone() );

                        layer
                    },
                    Node::NullaryNode { ref mut layer, .. } => {
                        let output_shape = layer.output_shape();
                        let output_type = layer.output_type();
                        match *layer {
                            AnyNullaryLayer::Constant( LayerConstant { ref name, ref data } ) => {
                                let mut array = PyArray::new( py, output_shape.prepend( data.len() ), output_type );
                                data.gather_bytes_into( .., array.as_bytes_mut() );

                                let tensor = tf_ns.getattr( "constant" ).unwrap_py( py ).call( (array.as_py_obj(),), None ).unwrap_py( py );
                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "tensor", tensor ).unwrap_py( py );
                                let layer_obj = layers_ns.getattr( "Input" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );

                                model_extra_inputs.push( layer_obj );
                                layer_obj
                            }
                        }
                    },
                    Node::UnaryNode { ref mut layer, .. } => {
                        let mut input = *inputs[0];
                        let input_shape = input_shapes.into_iter().next().unwrap();
                        match *layer {
                            AnyUnaryLayer::Activation( LayerActivation { ref name, ref activation } ) => {
                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );

                                let layer = match activation {
                                    Activation::ELU =>
                                        layers_ns.getattr( "Activation" ).unwrap_py( py ).call( ("elu",), Some( kwargs ) ).unwrap_py( py ),
                                    Activation::LeakyReLU => {
                                        kwargs.set_item( "negative_slope", 0.01 ).unwrap_py( py );
                                        layers_ns.getattr( "ReLU" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py )
                                    },
                                    Activation::Logistic =>
                                        layers_ns.getattr( "Activation" ).unwrap_py( py ).call( ("sigmoid",), Some( kwargs ) ).unwrap_py( py ),
                                    Activation::ReLU =>
                                        layers_ns.getattr( "Activation" ).unwrap_py( py ).call( ("relu",), Some( kwargs ) ).unwrap_py( py ),
                                    Activation::TanH =>
                                        layers_ns.getattr( "Activation" ).unwrap_py( py ).call( ("tanh",), Some( kwargs ) ).unwrap_py( py )
                                };
                                layer.call( (input,), None ).unwrap_py( py )
                            },
                            AnyUnaryLayer::Convolution( ref mut layer ) => {
                                if input_shape.dimension_count() == 2 {
                                    let kwargs = PyDict::new( py );
                                    let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                                    kwargs.set_item( "target_shape", target_shape ).unwrap_py( py );
                                    kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                    let layer = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                    input = layer.call( (input,), None ).unwrap_py( py );
                                }

                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", layer.name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                kwargs.set_item( "filters", layer.filter_count ).unwrap_py( py );
                                kwargs.set_item( "kernel_size", (layer.kernel_size.1, layer.kernel_size.0) ).unwrap_py( py );

                                if let Some( weights ) = layer.weights.take() {
                                    let (bias_array, weight_array) =
                                        Self::weights_into_arrays_for_convolutional_layer( py, &input_shape, &layer, &weights );

                                    let bias_array = constant.call( (bias_array.as_py_obj(),), None ).unwrap_py( py );
                                    let weight_array = constant.call( (weight_array.as_py_obj(),), None ).unwrap_py( py );

                                    kwargs.set_item( "bias_initializer", bias_array ).unwrap_py( py );
                                    kwargs.set_item( "kernel_initializer", weight_array ).unwrap_py( py );
                                }

                                let layer_obj = layers_ns.getattr( "Conv2D" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                input = layer_obj.call( (input,), None ).unwrap_py( py );
                                {
                                    let target_shape = layer.output_shape( &input_shape );
                                    let target_shape = PyTuple::new( py, &target_shape );
                                    let kwargs = PyDict::new( py );
                                    kwargs.set_item( "target_shape", target_shape ).unwrap_py( py );
                                    let layer_obj = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                    layer_obj.call( (input,), None ).unwrap_py( py )
                                }
                            },
                            AnyUnaryLayer::Dense( ref mut layer ) => {
                                {
                                    // Dense layers expect a one dimensional input.
                                    let layer = layers_ns.getattr( "Flatten" ).unwrap_py( py ).call( (), None ).unwrap_py( py );
                                    input = layer.call( (input,), None ).unwrap_py( py );
                                }

                                let kwargs = PyDict::new( py );
                                if let Some( weights ) = layer.weights.take() {
                                    let (bias_array, weight_array) =
                                        Self::weights_into_arrays_for_dense_layer( py, &input_shape, &layer, &weights );

                                    let bias_array = constant.call( (bias_array.as_py_obj(),), None ).unwrap_py( py );
                                    let weight_array = constant.call( (weight_array.as_py_obj(),), None ).unwrap_py( py );

                                    kwargs.set_item( "bias_initializer", bias_array ).unwrap_py( py );
                                    kwargs.set_item( "kernel_initializer", weight_array ).unwrap_py( py );
                                }

                                kwargs.set_item( "name", layer.name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                let layer = layers_ns.getattr( "Dense" ).unwrap_py( py ).call( (layer.size,), Some( kwargs ) ).unwrap_py( py );
                                layer.call( (input,), None ).unwrap_py( py )
                            },
                            AnyUnaryLayer::Dropout( LayerDropout { ref name, rate } ) => {
                                let rate: f32 = rate.into();
                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                let layer = layers_ns.getattr( "Dropout" ).unwrap_py( py ).call( (rate,), Some( kwargs ) ).unwrap_py( py );
                                layer.call( (input,), None ).unwrap_py( py )
                            },
                            AnyUnaryLayer::IntoCategory( _ ) => {
                                let layer = layers_ns.getattr( "Flatten" ).unwrap_py( py ).call( (), None ).unwrap_py( py );
                                layer.call( (input,), None ).unwrap_py( py )
                            },
                            AnyUnaryLayer::MaxPooling( ref layer ) => {
                                if input_shape.dimension_count() == 2 {
                                    let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                                    let kwargs = PyDict::new( py );
                                    kwargs.set_item( "target_shape", target_shape ).unwrap_py( py );
                                    kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                    let layer = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                    input = layer.call( (input,), None ).unwrap_py( py );
                                }

                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "pool_size", (layer.pool_size.1, layer.pool_size.0) ).unwrap_py( py );
                                kwargs.set_item( "name", layer.name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                // Padding is explained here:
                                //   https://stackoverflow.com/questions/37674306
                                kwargs.set_item( "padding", "same" ).unwrap_py( py );
                                let layer_obj = layers_ns.getattr( "MaxPool2D" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                input = layer_obj.call( (input,), None ).unwrap_py( py );
                                {
                                    let target_shape = layer.output_shape( &input_shape );
                                    let target_shape = PyTuple::new( py, &target_shape );
                                    let kwargs = PyDict::new( py );
                                    kwargs.set_item( "target_shape", target_shape ).unwrap_py( py );
                                    let layer_obj = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                    layer_obj.call( (input,), None ).unwrap_py( py )
                                }
                            },
                            AnyUnaryLayer::Reshape( LayerReshape { ref name, ref shape } ) => {
                                let target_shape = PyTuple::new( py, shape );
                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "target_shape", target_shape ).unwrap_py( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                let layer = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                layer.call( (input,), None ).unwrap_py( py )
                            },
                            AnyUnaryLayer::Softmax( LayerSoftmax { ref name } ) => {
                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                                kwargs.set_item( "trainable", is_trainable ).unwrap_py( py );
                                let layer = layers_ns.getattr( "Activation" ).unwrap_py( py ).call( ("softmax",), Some( kwargs ) ).unwrap_py( py );
                                layer.call( (input,), None ).unwrap_py( py )
                            }
                        }
                    },
                    Node::BinaryNode { ref mut layer, .. } => {
                        let input_1 = *inputs[0];
                        let input_2 = *inputs[1];
                        match *layer {
                            AnyBinaryLayer::Add( .. ) |
                            AnyBinaryLayer::Mul( .. ) => {
                                let (kind, name) = match layer {
                                    AnyBinaryLayer::Mul( LayerMul { name } ) => ("Multiply", name),
                                    AnyBinaryLayer::Add( LayerAdd { name } ) => ("Add", name)
                                };

                                let kwargs = PyDict::new( py );
                                kwargs.set_item( "name", name.to_string() ).unwrap_py( py );
                                let layer = layers_ns.getattr( kind ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                                let inputs = PyList::new( py, &[input_1, input_2] );
                                layer.call( (inputs,), None ).unwrap_py( py )
                            }
                        }
                    }
                };

                for output_index in model.output_indexes_for_node( node_index ) {
                    assert!( model_outputs[ output_index ].is_none() );

                    let layer = if let Some( ref shape ) = input_node_shape {
                        // An input node cannot be an output node,
                        // so we add a dummy reshape layer so that
                        // it can be our output node.
                        let kwargs = PyDict::new( py );
                        let shape = PyTuple::new( py, shape );
                        kwargs.set_item( "target_shape", shape ).unwrap_py( py );

                        let layer_obj = layers_ns.getattr( "Reshape" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );
                        layer_obj.call( (layer,), None ).unwrap_py( py )
                    } else {
                        layer
                    };

                    model_outputs[ output_index ] = Some( layer );
                }

                Ok( Some( layer ) )
            });

            for input in model_extra_inputs {
                model_input_list.append( input ).unwrap_py( py );
            }

            let model_output_list = PyList::empty( py );
            for model_output in model_outputs {
                let model_output = model_output.expect( "internal error: an output slot wasn't filled" );
                model_output_list.append( model_output ).unwrap_py( py );
            }

            let kwargs = PyDict::new( py );
            kwargs.set_item( "inputs", model_input_list ).unwrap_py( py );
            kwargs.set_item( "outputs", model_output_list ).unwrap_py( py );
            let model_obj = keras_ns.getattr( "Model" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py );

            if is_trainable {
                let compile_kwargs = PyDict::new( py );
                if let Some( training_opts ) = training_opts {
                    let optimizer = match training_opts.optimizer {
                        Optimizer::SGD( OptimizerSGD { learning_rate } ) => {
                            let kwargs = PyDict::new( py );
                            let learning_rate: f32 = learning_rate.into();
                            kwargs.set_item( "lr", learning_rate ).unwrap_py( py );
                            optimizers_ns.getattr( "SGD" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py )
                        },
                        Optimizer::Nadam( OptimizerNadam { learning_rate, beta_1, beta_2, epsilon, schedule_decay } ) => {
                            let kwargs = PyDict::new( py );
                            let learning_rate: f32 = learning_rate.into();
                            let beta_1: f32 = beta_1.into();
                            let beta_2: f32 = beta_2.into();
                            let epsilon: f32 = epsilon.into();
                            let schedule_decay: f32 = schedule_decay.into();
                            kwargs.set_item( "lr", learning_rate ).unwrap_py( py );
                            kwargs.set_item( "beta_1", beta_1 ).unwrap_py( py );
                            kwargs.set_item( "beta_2", beta_2 ).unwrap_py( py );
                            kwargs.set_item( "epsilon", epsilon ).unwrap_py( py );
                            kwargs.set_item( "schedule_decay", schedule_decay ).unwrap_py( py );
                            optimizers_ns.getattr( "Nadam" ).unwrap_py( py ).call( (), Some( kwargs ) ).unwrap_py( py )
                        }
                    };
                    compile_kwargs.set_item( "optimizer", optimizer ).unwrap_py( py );
                }

                let losses = PyList::empty( py );
                for &output_kind in &state.output_kinds {
                    let loss = match output_kind {
                        OutputKind::Regression => LossKind::MeanSquaredError,
                        OutputKind::SparseCategory => LossKind::SparseCategoricalCrossEntropy
                    };

                    let loss = match loss {
                        LossKind::SparseCategoricalCrossEntropy => "sparse_categorical_crossentropy",
                        LossKind::CategoricalCrossEntropy => "categorical_crossentropy",
                        LossKind::MeanSquaredError => "mean_squared_error"
                    };

                    losses.append( loss ).unwrap_py( py );
                }

                let metrics = PyList::new( py, &["accuracy"] );

                compile_kwargs.set_item( "loss", losses ).unwrap_py( py );
                compile_kwargs.set_item( "metrics", metrics ).unwrap_py( py );

                model_obj.getattr( "compile" ).unwrap_py( py ).call( (), Some( compile_kwargs ) ).unwrap_py( py );
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
        model: &Model,
        node: &Node,
        weights: &[f32]
    ) -> Result< (), SetWeightsError >
    {
        Context::gil( |py| {
            let weights = match *node {
                Node::Input { .. } => unreachable!{},
                Node::NullaryNode { .. } => unreachable!(),
                Node::BinaryNode { .. } => unreachable!(),
                Node::UnaryNode { ref layer, .. } => {
                    let input_shape = model.input_shapes_of( node ).next().unwrap();
                    match *layer {
                        AnyUnaryLayer::Convolution( ref layer ) => {
                            let (bias_array, weight_array) = Self::weights_into_arrays_for_convolutional_layer( py, &input_shape, layer, weights );
                            let list = PyList::empty( py );
                            list.append( weight_array ).unwrap_py( py );
                            list.append( bias_array ).unwrap_py( py );
                            list
                        },
                        AnyUnaryLayer::Dense( ref layer ) => {
                            let (bias_array, weight_array) = Self::weights_into_arrays_for_dense_layer( py, &input_shape, layer, weights );
                            let list = PyList::empty( py );
                            list.append( weight_array ).unwrap_py( py );
                            list.append( bias_array ).unwrap_py( py );
                            list
                        },
                        AnyUnaryLayer::Activation( _ ) |
                        AnyUnaryLayer::Dropout( _ ) |
                        AnyUnaryLayer::IntoCategory( _ ) |
                        AnyUnaryLayer::MaxPooling( _ ) |
                        AnyUnaryLayer::Reshape( _ ) |
                        AnyUnaryLayer::Softmax( _ )
                            => unreachable!()
                    }
                }
            };

            let layer_name = node.name().unwrap().to_string();
            let layer = self.obj.getattr( py, "get_layer" ).unwrap_py( py ).call( py, (layer_name,), None ).unwrap_py( py );
            layer.getattr( py, "set_weights" ).unwrap_py( py ).call( py, (weights,), None ).unwrap_py( py );
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

    pub(crate) fn get_weights( &self, model: &Model, node: &Node )
        -> Result< impl ToArrayRef + DataSource, GetWeightsError >
    {
        let weight_count = model.weight_count_of( node );
        let layer_name_s = node.name().unwrap().to_string();
        let output = Context::gil( move |py| {
            let layer = self.obj.getattr( py, "get_layer" ).unwrap_py( py ).call( py, (layer_name_s,), None ).unwrap_py( py );
            let weights_list = layer.getattr( py, "get_weights" ).unwrap_py( py ).call( py, (), None ).unwrap_py( py );
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

    fn train_on_batch( &mut self, py: Python, batch_size: usize, input_buffers: &[PyArray], output_buffers: &[PyArray] ) -> f32 {
        let inputs = PyList::empty( py );
        let outputs = PyList::empty( py );
        for buffer in input_buffers {
            inputs.append( buffer.as_py_obj() ).unwrap_py( py );
        }

        for buffer in output_buffers {
            outputs.append( buffer.as_py_obj() ).unwrap_py( py );
        }

        let loss = self.obj.getattr( py, "train_on_batch" ).unwrap_py( py )
            .call( py, (inputs, outputs), None )
            .unwrap_py( py );

        debug_assert_eq!(
            crate::backend::keras::py_utils::py_to_string(
                py,
                self.obj.getattr( py, "metrics_names" ).unwrap_py( py ).cast_as::< PyList >( py ).unwrap().get_item( 0 )
            ),
            "loss"
        );

        let loss: f32 = loss.cast_as::< PyList >( py ).unwrap()
            .get_item( 0 )
            .extract()
            .unwrap_py( py );

        // Tensorflow averages the loss it returns over the batch size for some reason;
        // we reverse it so that the losses are more comparable when changing the batch size.
        loss * (batch_size as f32)
    }

    pub(crate) fn train_for_epoch< F >( &mut self, state: &ModelInstanceState, batch_size: usize, mut fill_data: F ) -> f32
        where F: FnMut( &mut dyn RawBufferList, &mut dyn RawBufferList ) -> bool + Send
    {
        Context::gil( move |py| {
            let mut input_buffers: Vec< _ > = state.model.inputs().map( |io| {
                PyArray::new( py, io.shape.prepend( batch_size ), io.data_type )
            }).collect();

            let mut output_buffers: Vec< _ > = state.model.outputs().map( |io| {
                PyArray::new( py, io.shape.prepend( batch_size ), io.data_type )
            }).collect();

            let mut loss = 0.0;
            loop {
                let should_train = fill_data( &mut input_buffers, &mut output_buffers );
                if should_train {
                    loss += self.train_on_batch( py, batch_size, &input_buffers, &output_buffers );
                } else {
                    break;
                }
            }

            loss
        })
    }

    pub(crate) fn predict_raw< I >(
        &mut self,
        state: &ModelInstanceState,
        input_data_list: I
    ) -> Vec< RawArraySource >
        where I: DataSourceList + Send
    {
        Context::gil( move |py| {
            let input_buffers: Vec< _ > = state.model.inputs()
                .zip( input_data_list.data_sources() )
                .map( |(io, input_data)| {
                    assert_eq!( io.data_type, input_data.data_type() ); // TODO: Check this in backend-independent `predict_raw`.
                    let mut array = PyArray::new( py, io.shape.prepend( input_data.len() ), io.data_type );
                    input_data.gather_bytes_into( .., array.as_bytes_mut() );
                    array
                }).collect();

            let inputs_list = PyList::empty( py );
            for buffer in input_buffers {
                inputs_list.append( buffer.as_py_obj() ).unwrap_py( py );
            }

            let kwargs = PyDict::new( py );
            if state.model.inputs().len() == 0 {
                kwargs.set_item( "steps", 1 ).unwrap_py( py );
            }

            let results_list = self.obj.getattr( py, "predict" ).unwrap_py( py ).call( py, (inputs_list,), Some( kwargs ) ).unwrap_py( py );
            if state.model.outputs().len() == 1 {
                let results = results_list.to_object( py );
                let results = unsafe { PyArray::from_object_unchecked( py, results ) };
                vec![ results.into_raw_array() ]
            } else {
                let results_list: &PyList = py.checked_cast_as( results_list ).unwrap();
                results_list.iter().map( |results| {
                    let results = results.to_object( py );
                    let results = unsafe { PyArray::from_object_unchecked( py, results ) };
                    results.into_raw_array()
                }).collect()
            }
        })
    }
}
