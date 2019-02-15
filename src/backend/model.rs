use {
    std::{
        error::{
            Error
        }
    },
    crate::{
        backend::{
            Context,
            ContextKind,
            keras
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
            }
        },
        nn::{
            layers::{
                Layer,
                LayerPrototype
            },
            loss::{
                Loss
            },
            model::{
                InvalidModelError,
                Model
            },
            training_opts::{
                TrainingOpts
            }
        }
    }
};

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum ModelCompilationError {
    #[display(fmt = "model compilation failed: {}", "_0")]
    InvalidModel( InvalidModelError )
}

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum SetWeightsError {
    #[display(fmt = "failed to set weights: {}", "_0")]
    LayerNotFound( LayerNotFoundError ),
    #[display(fmt =
        "failed to set weights: layer '{}' has {} weight(s), yet {} weight(s) were passed",
        "layer_name",
        "layer_weight_count",
        "passed_weight_count"
    )]
    UnexpectedWeightCount {
        layer_name: Name,
        layer_weight_count: usize,
        passed_weight_count: usize
    }
}

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum GetWeightsError {
    #[display(fmt = "failed to get weights: {}", "_0")]
    LayerNotFound( LayerNotFoundError )
}

#[derive(Debug, Display)]
#[display(fmt = "model has no layer named '{}'", "_0")]
pub struct LayerNotFoundError( pub Name );

impl Error for ModelCompilationError {}
impl Error for SetWeightsError {}
impl Error for GetWeightsError {}

enum ModelInstanceKind {
    Keras( keras::ModelInstance )
}

pub(crate) enum OutputKind {
    Regression,
    SparseCategory
}

pub(crate) struct ModelInstanceState {
    pub model: Model,
    pub output_kind: OutputKind
}

pub struct ModelInstance {
    kind: ModelInstanceKind,
    state: ModelInstanceState
}

impl ModelInstance {
    pub fn new( ctx: &Context, model: Model ) -> Result< ModelInstance, ModelCompilationError > {
        Self::compile( ctx, model, None )
    }

    pub(crate) fn compile( ctx: &Context, model: Model, training_opts: Option< TrainingOpts > ) -> Result< ModelInstance, ModelCompilationError > {
        model.validate()?;

        let output_kind =
            if let Some( Layer::IntoCategory( .. ) ) = model.layers.last() {
                OutputKind::SparseCategory
            } else {
                OutputKind::Regression
            };

        let mut state = ModelInstanceState {
            model,
            output_kind
        };

        let kind = match ctx.0 {
            ContextKind::Keras( ref ctx ) => {
                let model_instance = keras::ModelInstance::compile( &ctx, &mut state, training_opts )?;
                ModelInstanceKind::Keras( model_instance )

            }
        };

        Ok( ModelInstance {
            kind,
            state
        })
    }

    pub(crate) fn predict_raw< I >( &mut self, input_data: &I ) -> RawArraySource where I: DataSource + Sync {
        let input_shape = input_data.shape();
        assert_eq!(
            self.state.model.input_shape(),
            input_shape,
            "The input data's shape is {}; expected it to be equal to the input shape of the model, which is {}",
            input_shape,
            self.state.model.input_shape()
        );

        match self.kind {
            ModelInstanceKind::Keras( ref mut model_instance ) => {
                model_instance.predict_raw( &self.state, input_data )
            }
        }
    }

    pub(crate) fn train_for_epoch< F >( &mut self, batch_size: usize, fill_data: F ) -> f32
        where F: FnMut( &mut [u8], &mut [u8] ) -> bool + Send
    {
        match self.kind {
            ModelInstanceKind::Keras( ref mut model_instance ) => {
                model_instance.train_for_epoch( &self.state, batch_size, fill_data )
            }
        }
    }

    pub fn input_shape( &self ) -> Shape {
        self.state.model.input_shape()
    }

    pub fn output_shape( &self ) -> Shape {
        self.state.model.output_shape()
    }

    pub fn set_weights< N >( &mut self, layer_name: N, weights: &[f32] ) -> Result< (), SetWeightsError > where N: Into< Name > {
        let layer_name = layer_name.into();
        let (layer, input_shape) = match self.state.model.get_layer_and_input_shape( &layer_name ) {
            Some( result ) => result,
            None => return Err( LayerNotFoundError( layer_name ).into() )
        };

        let weight_count = layer.weight_count( &input_shape );
        if weights.len() != weight_count {
            return Err( SetWeightsError::UnexpectedWeightCount {
                layer_name,
                layer_weight_count: weight_count,
                passed_weight_count: weights.len()
            });
        }

        if weight_count == 0 {
            return Ok(());
        }

        match self.kind {
            ModelInstanceKind::Keras( ref mut model_instance ) => {
                model_instance.set_weights( &input_shape, &layer, weights )
            }
        }
    }

    pub fn get_weights< N >( &self, layer_name: N ) -> Result< impl ToArrayRef + DataSource, GetWeightsError > where N: Into< Name > {
        let layer_name = layer_name.into();
        let (layer, input_shape) = match self.state.model.get_layer_and_input_shape( &layer_name ) {
            Some( result ) => result,
            None => return Err( LayerNotFoundError( layer_name ) )?
        };

        let weight_count = layer.weight_count( &input_shape );
        let weights = match self.kind {
            ModelInstanceKind::Keras( ref model_instance ) => {
                model_instance.get_weights( &input_shape, &layer )?
            }
        };

        assert_eq!(
            weight_count,
            weights.shape().product(),
            "Internal error: expected the number of weights for layer {} to be {}; instead it is {}",
            layer_name,
            weight_count,
            weights.shape().product()
        );

        Ok( weights )
    }

    pub fn predict< I >( &mut self, input_data: &I ) -> impl ToArrayRef + DataSource where I: DataSource + Sync {
        let result = self.predict_raw( input_data );
        match self.state.output_kind {
            OutputKind::Regression => {
                debug_assert_eq!(
                    result.shape(),
                    self.state.model.output_shape(),
                    "Internal error: expected the output of the network to have a shape of {}; instead it has a shape of {}",
                    self.state.model.output_shape(),
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

        match self.state.output_kind {
            OutputKind::Regression => self.test_regression( chunk_size, test_data ),
            OutputKind::SparseCategory => self.test_classification( chunk_size, test_data )
        }
    }
}
