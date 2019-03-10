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
                DataSource,
                DataSourceExt,
                DataSourceList,
                DataSourceListExt
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
            }
        },
        nn::{
            layers::{
                AnyUnaryLayer
            },
            loss::{
                Loss
            },
            model::{
                InvalidModelError,
                Model
            },
            model::{
                Node
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

pub(crate) trait RawBufferList {
    fn get_buffer_mut( &mut self, index: usize ) -> &mut [u8];
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum OutputKind {
    Regression,
    SparseCategory
}

pub(crate) struct ModelInstanceState {
    pub model: Model,
    pub output_kinds: Vec< OutputKind >
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

        let output_kinds =
            model.outputs().map( |io| {
                match *model.get_node( io.node_index ) {
                    Node::UnaryNode { layer: AnyUnaryLayer::IntoCategory( .. ), .. } => OutputKind::SparseCategory,
                    _ => OutputKind::Regression
                }
            }).collect();

        let mut state = ModelInstanceState {
            model,
            output_kinds
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

    pub(crate) fn predict_raw< I >( &mut self, input_data_list: I ) -> Vec< RawArraySource > where I: DataSourceList + Send {
        let model_inputs = self.state.model.inputs();
        assert_eq!(
            input_data_list.data_sources().len(),
            model_inputs.len(),
            "The model expects {} inputs, got {}",
            model_inputs.len(),
            input_data_list.data_sources().len()
        );

        for (io, input_data) in model_inputs.zip( input_data_list.data_sources() ) {
            assert_eq!(
                input_data.shape(),
                io.shape,
                "The input data #{}'s shape is {}; expected it to be equal to the input shape of the model's #{} input, which is {}",
                io.index,
                input_data.shape(),
                io.index,
                io.shape
            );
        }

        match self.kind {
            ModelInstanceKind::Keras( ref mut model_instance ) => {
                model_instance.predict_raw( &self.state, input_data_list )
            }
        }
    }

    pub(crate) fn train_for_epoch< F >( &mut self, batch_size: usize, fill_data: F ) -> f32
        where F: FnMut( &mut dyn RawBufferList, &mut dyn RawBufferList ) -> bool + Send
    {
        match self.kind {
            ModelInstanceKind::Keras( ref mut model_instance ) => {
                model_instance.train_for_epoch( &self.state, batch_size, fill_data )
            }
        }
    }

    pub fn set_weights< N >( &mut self, layer_name: N, weights: &[f32] ) -> Result< (), SetWeightsError > where N: Into< Name > {
        let layer_name = layer_name.into();
        let node = match self.state.model.get_node_by_name( &layer_name ) {
            Some( node ) => node,
            None => return Err( LayerNotFoundError( layer_name ).into() )
        };

        let weight_count = self.state.model.weight_count_of( node );
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
                model_instance.set_weights( &self.state.model, node, weights )
            }
        }
    }

    pub fn get_weights< N >( &self, layer_name: N ) -> Result< impl ToArrayRef + DataSource, GetWeightsError > where N: Into< Name > {
        let layer_name = layer_name.into();
        let node = match self.state.model.get_node_by_name( &layer_name ) {
            Some( node ) => node,
            None => return Err( LayerNotFoundError( layer_name ) )?
        };

        let weight_count = self.state.model.weight_count_of( node );
        let weights = match self.kind {
            ModelInstanceKind::Keras( ref model_instance ) => {
                model_instance.get_weights( &self.state.model, node )?
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

    pub fn predict< I >( &mut self, input_data_list: &I ) -> Vec< impl ToArrayRef + DataSource > where I: DataSourceList + Sync {
        let mut results_list = self.predict_raw( input_data_list );
        for ((results, &output_kind), io) in results_list.iter_mut().zip( self.state.output_kinds.iter() ).zip( self.state.model.outputs() ) {
            match output_kind {
                OutputKind::Regression => {
                    debug_assert_eq!(
                        results.shape(),
                        io.shape,
                        "Internal error: expected the output of the network to have a shape of {}; instead it has a shape of {}",
                        io.shape,
                        results.shape()
                    );
                },
                OutputKind::SparseCategory => {
                    let count = results.len();
                    let mut categories = RawArraySource::new_uninitialized( count, 1.into(), Type::U32 );
                    let categories_slice = cast_slice_mut::< u32 >( categories.as_bytes_mut() );
                    let typed_results = results.to_typed_array_ref::< f32 >().expect( "internal error: unhandled array type" );
                    for index in 0..count {
                        let category = typed_results[ index ]
                            .iter()
                            .enumerate()
                            .max_by_key( |(_, &value)| decorum::Finite::from( value ) )
                            .unwrap()
                            .0;
                        categories_slice[ index ] = category as u32;
                    }

                    *results = categories;
                }
            }
        }

        results_list
    }

    pub fn test< I, O >( &mut self, test_data: &DataSet< I, O > ) -> Loss
        where I: DataSourceList + Sync, O: DataSourceList + Sync
    {
        for (io, output_data) in self.state.model.outputs().zip( test_data.expected_output_list().data_sources() ) {
            assert_eq!(
                output_data.shape(),
                io.shape,
                "Model's output #{} has a shape of {}, and yet the test output data's shape is {}",
                io.index,
                io.shape,
                output_data.shape()
            );

            assert_eq!(
                output_data.data_type(),
                io.data_type,
                "Model's output #{} has a data type of {}, and yet the test output data's data type is {}",
                io.index,
                io.data_type,
                output_data.data_type()
            );
        }

        // TODO: Pick the chunk size more intelligently.
        let chunk_size = 128;
        let mut total_loss = 0.0;
        let mut correct_count = 0;

        let mut buffers: Vec< _ > =
            self.state.model.outputs()
            .map( |io| RawArraySource::new_uninitialized( chunk_size, io.shape, io.data_type ) )
            .collect();

        for test_chunk in test_data.chunks( chunk_size ) {
            let chunk_size = test_chunk.len();
            let predictions = self.predict_raw( test_chunk.input_list() );
            debug_assert_eq!( predictions.len(), buffers.len() );

            for (data_source, buffer) in test_chunk.expected_output_list().data_sources().zip( buffers.iter_mut() ) {
                let element_size = buffer.shape().product() * buffer.data_type().byte_size();
                let buffer = &mut buffer.as_bytes_mut()[ ..chunk_size * element_size ];
                data_source.gather_bytes_into( .., buffer );
            }

            let iter = predictions.into_iter()
                .zip( buffers.iter_mut() )
                .zip( self.state.output_kinds.iter().cloned() );

            for ((predictions, expected), output_kind) in iter {
                debug_assert_eq!( predictions.len(), chunk_size );
                let predictions = predictions.to_typed_array_ref::< f32 >().expect( "internal error: unhandled array type" );
                match output_kind {
                    OutputKind::Regression => {
                        let element_size = expected.shape().product();
                        let expected = expected.to_typed_array_ref::< f32 >().unwrap();
                        let expected = expected.as_slice();
                        let predictions = predictions.as_slice();
                        for (expected, predicted) in expected.chunks_exact( element_size ).zip( predictions.chunks_exact( element_size ) ) {
                            total_loss += predicted.iter().zip( expected.iter() )
                                .map( |(output, expected_output)| (output - expected_output) * 2.0 / element_size as f32 )
                                .map( |output_error| output_error * output_error )
                                .sum::< f32 >()
                                * element_size as f32
                                / 4.0;
                        }
                    },
                    OutputKind::SparseCategory => {
                        let expected = expected.to_typed_array_ref::< u32 >().unwrap();
                        let expected = expected.as_slice();
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
                }
            }
        }

        let accuracy = if self.state.output_kinds.iter().cloned().any( |kind| kind == OutputKind::SparseCategory ) {
            let total_count = test_data.len();
            let accuracy = correct_count as f32 / total_count as f32;
            Some( accuracy )
        } else {
            None
        };

        Loss {
            loss: total_loss,
            accuracy
        }
    }
}
