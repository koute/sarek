use {
    std::{
        error::{
            Error
        },
        fmt
    },
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
                    PyResultExt,
                    py_err
                }
            }
        },
        core::{
            array::{
                ToArrayRef
            },
            data_set::{
                DataSet
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
                LayerConvolution,
                LayerDense,
                LayerDropout,
                LayerPrototype,
                LayerReshape,
                LayerSoftmax
            },
            loss::{
                Loss
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
pub enum ModelCompilationError {}

#[derive(Debug)]
pub struct SetWeightsError(());

impl fmt::Display for ModelCompilationError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "model compilation failed" )
    }
}

impl fmt::Display for SetWeightsError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "failed to set weights" )
    }
}

impl Error for ModelCompilationError {}
impl Error for SetWeightsError {}

// TODO: Replace this with pure Rust code.
pub(crate) fn ortho_weights( shape: Shape ) -> RawArraySource {
    Context::gil( move |py| {
        let code = concat!(
            "flat_shape = (shape[0], np.prod(shape[1:]))\n",
            "a = np.random.standard_normal(flat_shape)\n",
            "u, _, v = np.linalg.svd(a, full_matrices=False)\n",
            "q = u if u.shape == flat_shape else v\n",
            "q = q.reshape(shape)\n",
            "q = q.astype('float32')\n",
            "q"
        );
        let shape = PyTuple::new( py, &shape );
        let args = PyDict::new( py );
        let np = py.import( "numpy" ).unwrap();
        args.set_item( "np", np ).unwrap();
        args.set_item( "shape", shape.to_object( py ) ).unwrap();
        py.run( code, None, Some( args ) ).map_err( |err| py_err( py, err ) ).unwrap();
        let result = args.get_item( "q" ).unwrap();
        let result = unsafe { PyArray::from_object_unchecked( py, result.to_object( py ) ) };
        result.into_raw_array()
    })
}

impl ModelInstance {
    pub fn new( ctx: &Context, model: Model ) -> Result< ModelInstance, ModelCompilationError > {
        Self::compile( ctx, model, None )
    }

    pub(crate) fn compile( ctx: &Context, mut model: Model, training_opts: Option< TrainingOpts > ) -> Result< ModelInstance, ModelCompilationError > {
        Context::gil( move |py| {
            let tf_ns = py.import( "tensorflow" ).unwrap();
            let keras_ns = tf_ns.getattr( "keras" ).unwrap();
            let layers_ns = keras_ns.getattr( "layers" ).unwrap();
            let models_ns = keras_ns.getattr( "models" ).unwrap();
            let optimizers_ns = keras_ns.getattr( "optimizers" ).unwrap();
            let is_trainable = training_opts.is_some();
            let mut output_kind = OutputKind::Regression;
            let mut is_first = true;
            let mut input_shape = model.input_shape();

            let model_layer_count = model.layers.len();
            let mut layers = Vec::with_capacity( model.layers.len() );
            for (layer_index, layer) in model.layers.iter_mut().enumerate() {
                let mut kwargs = PyDict::new( py );

                if is_first {
                    let shape = PyTuple::new( py, &input_shape );
                    kwargs.set_item( "input_shape", shape ).unwrap();
                    is_first = false;
                }

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
                        layers.push( layer );
                    },
                    Layer::Convolution( layer ) => {
                        if input_shape.dimension_count() == 2 {
                            let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            kwargs.set_item( "trainable", is_trainable ).unwrap();
                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            layers.push( layer );

                            kwargs = PyDict::new( py );
                        }

                        kwargs.set_item( "name", layer.name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        kwargs.set_item( "filters", layer.filter_count ).unwrap();
                        kwargs.set_item( "kernel_size", (layer.kernel_size.1, layer.kernel_size.0) ).unwrap();

                        if let Some( weights ) = layer.weights.take() {
                            let (bias_array, weight_array) =
                                Self::weights_into_arrays_for_convolutional_layer( py, &input_shape, &layer, &weights );

                            let initializers_ns = keras_ns.getattr( "initializers" ).unwrap();
                            let constant = initializers_ns.getattr( "constant" ).unwrap();
                            let bias_array = constant.call( (bias_array.as_py_obj(),), None ).unwrap();
                            let weight_array = constant.call( (weight_array.as_py_obj(),), None ).unwrap();

                            kwargs.set_item( "bias_initializer", bias_array ).unwrap();
                            kwargs.set_item( "kernel_initializer", weight_array ).unwrap();
                        }

                        let layer_obj = layers_ns.getattr( "Conv2D" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                        layers.push( layer_obj );
                        {
                            let target_shape = layer.output_shape( &input_shape );
                            let target_shape = PyTuple::new( py, &target_shape );
                            kwargs = PyDict::new( py );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            let layer_obj = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                            layers.push( layer_obj );
                        }
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
                            let target_shape = PyTuple::new( py, &input_shape );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();

                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            layers.push( layer );
                        }

                        output_kind = OutputKind::SparseCategory;
                    },
                    Layer::MaxPooling( layer ) => {
                        if input_shape.dimension_count() == 2 {
                            let target_shape = PyTuple::new( py, &input_shape.append( 1 ) );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            kwargs.set_item( "trainable", is_trainable ).unwrap();
                            let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                            layers.push( layer );

                            kwargs = PyDict::new( py );
                        }

                        kwargs.set_item( "pool_size", (layer.pool_size.1, layer.pool_size.0) ).unwrap();
                        kwargs.set_item( "name", layer.name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        // Padding is explained here:
                        //   https://stackoverflow.com/questions/37674306
                        kwargs.set_item( "padding", "same" ).unwrap();
                        let layer_obj = layers_ns.getattr( "MaxPool2D" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                        layers.push( layer_obj );
                        {
                            let target_shape = layer.output_shape( &input_shape );
                            let target_shape = PyTuple::new( py, &target_shape );
                            kwargs = PyDict::new( py );
                            kwargs.set_item( "target_shape", target_shape ).unwrap();
                            let layer_obj = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap_py( py );
                            layers.push( layer_obj );
                        }
                    },
                    Layer::Reshape( LayerReshape { name, shape } ) => {
                        let target_shape = PyTuple::new( py, shape );
                        kwargs.set_item( "target_shape", target_shape ).unwrap();
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Reshape" ).unwrap().call( (), Some( kwargs ) ).unwrap();
                        layers.push( layer );
                    },
                    Layer::Softmax( LayerSoftmax { name } ) => {
                        kwargs.set_item( "name", name.to_string() ).unwrap();
                        kwargs.set_item( "trainable", is_trainable ).unwrap();
                        let layer = layers_ns.getattr( "Activation" ).unwrap().call( ("softmax",), Some( kwargs ) ).unwrap();
                        layers.push( layer );
                    }
                }

                input_shape = layer.output_shape( &input_shape );
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

    pub(crate) fn model( &self ) -> &Model {
        &self.model
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
            self.set_weights_for_layer( py, &input_shape, layer, weights );
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

    fn set_weights_for_layer( &self, py: Python, input_shape: &Shape, layer: &Layer, weights: &[f32] ) {
        let weights = match layer {
            Layer::Convolution( layer ) => {
                let (bias_array, weight_array) = Self::weights_into_arrays_for_convolutional_layer( py, input_shape, layer, weights );
                let list = PyList::empty( py );
                list.append( weight_array ).unwrap();
                list.append( bias_array ).unwrap();
                list
            },
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
            Layer::MaxPooling( _ ) |
            Layer::Reshape( _ ) |
            Layer::Softmax( _ )
                => unreachable!()
        };

        let layer_name = layer.name().to_string();
        let layer = self.obj.getattr( py, "get_layer" ).unwrap().call( py, (layer_name,), None ).unwrap();
        layer.getattr( py, "set_weights" ).unwrap().call( py, (weights,), None ).map_err( |err| py_err( py, err ) ).unwrap();
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
        let layer_name_s = layer_name.to_string();
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

        assert_eq!(
            weight_count,
            output.len(),
            "Internal error: expected the number of weights for layer {} to be {}; instead it is {}",
            layer_name,
            weight_count,
            output.len()
        );

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

    pub(crate) fn predict_raw< I >( &mut self, input_data: &I ) -> RawArraySource where I: DataSource + Sync {
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

    fn test_regression< I, O >( &mut self, chunk_size: usize, test_data: &DataSet< I, O > ) -> Loss
        where I: DataSource + Sync, O: DataSource + Sync
    {
        let mut total_loss = 0.0;

        let mut expected: Vec< f32 > = Vec::new();
        expected.reserve( chunk_size );
        unsafe {
            expected.set_len( chunk_size );
        }

        let element_size = test_data.expected_output_data().shape().product();
        for test_chunk in test_data.chunks( chunk_size ) {
            let chunk_size = test_chunk.len();
            let predictions = self.predict_raw( test_chunk.input_data() );
            debug_assert_eq!( predictions.len(), chunk_size );

            let expected = &mut expected[ ..chunk_size * element_size ];
            test_chunk.expected_output_data().gather_into( .., expected );

            let predictions = predictions.to_typed_array_ref::< f32 >().expect( "internal error: unhandled array type" );
            let predictions = predictions.as_slice();
            for (expected, predicted) in expected.chunks_exact( element_size ).zip( predictions.chunks_exact( element_size ) ) {
                total_loss += predicted.iter().zip( expected.iter() )
                    .map( |(output, expected_output)| (output - expected_output) * 2.0 / element_size as f32 )
                    .map( |output_error| output_error * output_error )
                    .sum::< f32 >()
                    * element_size as f32
                    / 4.0;
            }
        }

        Loss {
            loss: total_loss,
            accuracy: None
        }
    }

    fn test_classification< I, O >( &mut self, chunk_size: usize, test_data: &DataSet< I, O > ) -> Loss
        where I: DataSource + Sync, O: DataSource + Sync
    {
        let mut total_loss = 0.0;

        let mut correct_count = 0;
        let mut expected: Vec< u32 > = Vec::new();
        expected.reserve( chunk_size );
        unsafe {
            expected.set_len( chunk_size );
        }

        for test_chunk in test_data.chunks( chunk_size ) {
            let chunk_size = test_chunk.len();
            let predictions = self.predict_raw( test_chunk.input_data() );
            debug_assert_eq!( predictions.len(), chunk_size );

            let expected = &mut expected[ ..chunk_size ];
            test_chunk.expected_output_data().gather_into( .., expected );

            let predictions = predictions.to_typed_array_ref::< f32 >().expect( "internal error: unhandled array type" );
            for index in 0..chunk_size {
                let prediction = &predictions[ index ];
                let predicted = prediction
                    .iter()
                    .enumerate()
                    .max_by_key( |(_, &value)| decorum::Finite::from( value ) )
                    .unwrap()
                    .0 as u32;

                let expected = expected[ index ];
                if predicted == expected {
                    correct_count += 1;
                }

                let sum: f32 = prediction.iter().sum();
                total_loss += -(prediction[ expected as usize ] / sum).ln();
            }
        }

        let total_count = test_data.len();
        let accuracy = correct_count as f32 / total_count as f32;
        Loss {
            loss: total_loss,
            accuracy: Some( accuracy )
        }
    }

    pub fn test< I, O >( &mut self, test_data: &DataSet< I, O > ) -> Loss
        where I: DataSource + Sync, O: DataSource + Sync
    {
        // TODO: Pick the chunk size more intelligently.
        let chunk_size = 128;

        match self.output_kind {
            OutputKind::Regression => self.test_regression( chunk_size, test_data ),
            OutputKind::SparseCategory => self.test_classification( chunk_size, test_data )
        }
    }
}
