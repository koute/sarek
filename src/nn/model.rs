use {
    crate::{
        core::{
            data_type::{
                Type
            },
            maybe_owned::{
                MaybeOwned
            },
            name::{
                Name
            },
            shape::{
                Shape
            },
            small_iter::{
                SmallIter
            }
        },
        nn::{
            layers::{
                *
            }
        }
    },
    smallvec::{
        SmallVec
    },
    std::{
        cell::{
            RefCell
        },
        collections::{
            HashMap,
            HashSet
        },
        fmt,
        iter::{
            FusedIterator
        },
        mem
    }
};

#[non_exhaustive]
#[derive(Debug, Display, From)]
pub enum InvalidModelError {
    #[display(fmt =
        "layer {} ({}) has the same name as layer {} ({}): '{}'",
        "node_index_1",
        "layer_kind_1",
        "node_index_2",
        "layer_kind_2",
        "layer_name"
    )]
    DuplicateName {
        node_index_1: NodeIndex,
        layer_kind_1: &'static str,
        node_index_2: NodeIndex,
        layer_kind_2: &'static str,
        layer_name: Name
    },

    #[display(fmt =
        "layer {} (Reshape) '{}' has an output shape of {} ({}) which is incompatible with its input shape of {} ({})",
        "node_index",
        "layer_name",
        "output_shape",
        "output_shape.product()",
        "input_shape",
        "input_shape.product()"
    )]
    InvalidReshape {
        node_index: NodeIndex,
        layer_name: Name,
        input_shape: Shape,
        output_shape: Shape
    },

    #[display(fmt =
        "layer {} ({}) '{}' expects its inputs to have the same shape where currently its first input has a shape of {} while the second one has a shape of {}",
        "node_index",
        "layer_kind",
        "layer_name",
        "input_shape_1",
        "input_shape_2"
    )]
    ExpectedEqualInputShapes {
        node_index: NodeIndex,
        layer_kind: &'static str,
        layer_name: Name,
        input_shape_1: Shape,
        input_shape_2: Shape
    },

    #[display(fmt =
        "layer {} ({}) '{}' is only supported when it's the last layer in the model",
        "node_index",
        "layer_kind",
        "layer_name"
    )]
    LayerShouldBeTheLastLayer {
        node_index: NodeIndex,
        layer_kind: &'static str,
        layer_name: Name
    },

    #[display(fmt =
        "layer {} ({}) '{}' is missing weights",
        "node_index",
        "layer_kind",
        "layer_name"
    )]
    InvalidWeightCount {
        node_index: NodeIndex,
        layer_kind: &'static str,
        layer_name: Name,
        weight_count: usize,
        expected_weight_count: usize
    },

    #[display(fmt =
        "layer {} ({}) '{}' is missing weights",
        "node_index",
        "layer_kind",
        "layer_name"
    )]
    MissingWeights {
        node_index: NodeIndex,
        layer_kind: &'static str,
        layer_name: Name
    },

    #[display(fmt =
        "layer {} ({}) '{}' was assigned weights which contain either a NaN or an Inf",
        "node_index",
        "layer_kind",
        "layer_name"
    )]
    InvalidWeights{
        node_index: NodeIndex,
        layer_kind: &'static str,
        layer_name: Name
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    nodes: Vec< Node >,
    inputs: Vec< NodeIndex >,
    outputs: Vec< NodeIndex >,
    initial_nodes: Vec< NodeIndex >
}

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeIndex( usize );

impl NodeIndex {
    fn new( raw: usize ) -> Self {
        NodeIndex( raw )
    }

    fn raw( self ) -> usize {
        self.0
    }
}

impl fmt::Display for NodeIndex {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "#{}", self.0 )
    }
}

pub trait NullaryLayer: LayerPrototype + Into< AnyNullaryLayer > {
    fn output_shape( &self ) -> Shape;
    fn output_type( &self ) -> Type;

    fn into_node( self, builder: &ModelBuilder ) -> NodeRef {
        builder.add_nullary_layer( self.into() )
    }
}

pub trait UnaryLayer: LayerPrototype + Into< AnyUnaryLayer > {
    fn output_shape( &self, input_shape: &Shape ) -> Shape;
    fn weight_count( &self, input_shape: &Shape ) -> usize;

    fn into_node( self, input: NodeRef ) -> NodeRef {
        input.chain_with_unary_layer( self.into() )
    }
}

pub trait BinaryLayer: LayerPrototype + Into< AnyBinaryLayer > {
    fn output_shape( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> Shape;
    fn weight_count( &self, input_shape_1: &Shape, input_shape_2: &Shape ) -> usize;

    fn into_node< 'a >( self, input_1: NodeRef< 'a >, input_2: NodeRef< 'a > ) -> NodeRef< 'a > {
        input_1.chain_with_binary_layer( input_2, self.into() )
    }
}

pub trait UnaryLayerList {
    fn collapse_into_node( self, input: NodeRef ) -> NodeRef;
}

impl UnaryLayerList for () {
    fn collapse_into_node( self, input: NodeRef ) -> NodeRef {
        input
    }
}

impl< T > UnaryLayerList for T where T: UnaryLayer {
    fn collapse_into_node( self, input: NodeRef ) -> NodeRef {
        self.into_node( input )
    }
}

type Outputs = SmallVec< [NodeIndex; 2] >;

#[derive(Clone, Debug)]
pub enum Node {
    Input {
        outputs: Outputs,
        input_index: usize,
        shape: Shape
    },
    NullaryNode {
        outputs: Outputs,
        layer: AnyNullaryLayer
    },
    UnaryNode {
        outputs: Outputs,
        input: NodeIndex,
        layer: AnyUnaryLayer,
        output_shape: Shape
    },
    BinaryNode {
        outputs: Outputs,
        inputs: (NodeIndex, NodeIndex),
        layer: AnyBinaryLayer,
        output_shape: Shape
    }
}

impl Node {
    pub fn outputs( &self ) -> &Outputs {
        match *self {
            Node::Input { ref outputs, .. } |
            Node::NullaryNode { ref outputs, .. } |
            Node::UnaryNode { ref outputs, .. } |
            Node::BinaryNode { ref outputs, .. } => outputs
        }
    }

    pub fn outputs_mut( &mut self ) -> &mut Outputs {
        match *self {
            Node::Input { ref mut outputs, .. } |
            Node::NullaryNode { ref mut outputs, .. } |
            Node::UnaryNode { ref mut outputs, .. } |
            Node::BinaryNode { ref mut outputs, .. } => outputs
        }
    }

    pub fn inputs( &self ) -> impl ExactSizeIterator< Item = NodeIndex > + FusedIterator {
        match *self {
            Node::Input { .. } => SmallIter::Empty,
            Node::NullaryNode { .. } => SmallIter::Empty,
            Node::UnaryNode { input, .. } => SmallIter::One( input ),
            Node::BinaryNode { inputs, .. } => SmallIter::Two( inputs.0, inputs.1 )
        }
    }

    pub fn inputs_mut( &mut self ) -> impl ExactSizeIterator< Item = &mut NodeIndex > + FusedIterator {
        match *self {
            Node::Input { .. } => SmallIter::Empty,
            Node::NullaryNode { .. } => SmallIter::Empty,
            Node::UnaryNode { ref mut input, .. } => SmallIter::One( input ),
            Node::BinaryNode { ref mut inputs, .. } => SmallIter::Two( &mut inputs.0, &mut inputs.1 )
        }
    }

    pub fn output_shape( &self ) -> Shape {
        match *self {
            Node::Input { ref shape, .. } => shape.clone(),
            Node::NullaryNode { ref layer, .. } => layer.output_shape(),
            Node::UnaryNode { ref output_shape, .. } |
            Node::BinaryNode { ref output_shape, .. } => output_shape.clone()
        }
    }

    pub fn output_type( &self ) -> Type {
        match *self {
            Node::NullaryNode { layer: AnyNullaryLayer::Constant( LayerConstant { ref data, .. } ), .. } => data.data_type(),
            Node::UnaryNode { layer: AnyUnaryLayer::IntoCategory { .. }, .. } => Type::U32,
            // TODO: this should not be hardcoded.
            _ => Type::F32
        }
    }

    pub fn name( &self ) -> Option< &Name > {
        match *self {
            Node::Input { .. } => None,
            Node::NullaryNode { ref layer, .. } => Some( layer.name() ),
            Node::UnaryNode { ref layer, .. } => Some( layer.name() ),
            Node::BinaryNode { ref layer, .. } => Some( layer.name() )
        }
    }

    pub fn type_name( &self ) -> &'static str {
        match *self {
            Node::Input { .. } => "Input",
            Node::NullaryNode { ref layer, .. } => layer.type_name(),
            Node::UnaryNode { ref layer, .. } => layer.type_name(),
            Node::BinaryNode { ref layer, .. } => layer.type_name()
        }
    }
}

#[non_exhaustive]
pub struct ModelInOut {
    pub index: usize,
    pub node_index: NodeIndex,
    pub data_type: Type,
    pub shape: Shape
}

pub struct ModelBuilder( RefCell< Model > );

impl Model {
    pub fn new_graph< F >( callback: F ) -> Model where F: FnOnce( &mut ModelBuilder ) {
        let mut builder = ModelBuilder::new();
        callback( &mut builder );
        builder.build()
    }

    pub(crate) fn modify< F >( &mut self, callback: F ) where F: FnOnce( &mut ModelBuilder ) {
        take_mut::take( self, |model| {
            let mut builder = ModelBuilder( RefCell::new( model ) );
            callback( &mut builder );
            builder.build()
        })
    }

    pub fn new_sequential< S, T >( input_shape: S, layers: T ) -> Self
        where S: Into< Shape >,
              T: UnaryLayerList
    {
        Self::new_graph( |builder| {
            builder
            .add_input( input_shape.into() )
            .chain( layers )
            .add_as_output();
        })
    }

    pub(crate) fn get_node_by_name( &self, layer_name: &Name ) -> Option< &Node > {
        self.nodes.iter().find( |node| node.name().map( |name| name == layer_name ).unwrap_or( false ) )
    }

    pub(crate) fn get_node( &self, index: NodeIndex ) -> &Node {
        &self.nodes[ index.raw() ]
    }

    pub(crate) fn get_node_mut( &mut self, index: NodeIndex ) -> &mut Node {
        &mut self.nodes[ index.raw() ]
    }

    pub(crate) fn input_shapes_of( &self, node: &Node ) -> impl ExactSizeIterator< Item = Shape > + FusedIterator {
        match *node {
            Node::Input { .. } => SmallIter::Empty,
            Node::NullaryNode { .. } => SmallIter::Empty,
            Node::UnaryNode { input, .. } => SmallIter::One( self.get_node( input ).output_shape().clone() ),
            Node::BinaryNode { inputs, .. } => {
                SmallIter::Two(
                    self.get_node( inputs.0 ).output_shape().clone(),
                    self.get_node( inputs.1 ).output_shape().clone()
                )
            }
        }
    }

    pub(crate) fn weight_count_of( &self, node: &Node ) -> usize {
        match *node {
            Node::Input { .. } => 0,
            Node::NullaryNode { .. } => 0,
            Node::UnaryNode { input, ref layer, .. } => {
                layer.weight_count( &self.get_node( input ).output_shape() )
            },
            Node::BinaryNode { inputs, ref layer, .. } => {
                let input_shape_1 = self.get_node( inputs.0 ).output_shape();
                let input_shape_2 = self.get_node( inputs.1 ).output_shape();
                layer.weight_count( &input_shape_1, &input_shape_2 )
            }
        }
    }

    pub fn inputs< 'a >( &'a self ) -> impl ExactSizeIterator< Item = ModelInOut > + FusedIterator + 'a {
        self.inputs.iter().cloned().enumerate().map( move |(index, node_index)| {
            let node = self.get_node( node_index );
            let shape = node.output_shape().clone();
            let data_type = node.output_type();

            ModelInOut {
                index,
                node_index,
                data_type,
                shape
            }
        })
    }

    pub fn outputs< 'a >( &'a self ) -> impl ExactSizeIterator< Item = ModelInOut > + FusedIterator + 'a {
        self.outputs.iter().cloned().enumerate().map( move |(index, node_index)| {
            let node = self.get_node( node_index );
            let shape = node.output_shape().clone();
            let data_type = node.output_type();

            ModelInOut {
                index,
                node_index,
                data_type,
                shape
            }
        })
    }

    pub(crate) fn initial_node_indexes< 'a >( &'a self ) -> impl Iterator< Item = NodeIndex > + FusedIterator + 'a {
        self.initial_nodes.iter().cloned()
    }

    pub(crate) fn node_indexes( &self ) -> impl ExactSizeIterator< Item = NodeIndex > + FusedIterator + 'static {
        (0..self.nodes.len()).into_iter().map( |index| NodeIndex( index ) )
    }

    pub(crate) fn output_indexes_for_node< 'a >( &'a self, node_index: NodeIndex ) -> impl Iterator< Item = usize > + FusedIterator + 'a {
        self.outputs.iter()
            .cloned()
            .enumerate()
            .filter( move |&(_, index)| index == node_index )
            .map( |(output_index, _)| output_index )
    }

    pub(crate) fn traverse_mut< T, E, F >( &mut self, mut callback: F ) -> Result< (), E >
        where F: FnMut( &mut Model, &[MaybeOwned< T >], NodeIndex ) -> Result< Option< T >, E >
    {
        fn process< T, E, F >(
            model: &mut Model,
            node_index: NodeIndex,
            result_for_node: &mut [Option< (T, usize) >],
            callback: &mut F
        ) -> Result< (), E >
            where F: FnMut( &mut Model, &[MaybeOwned< T >], NodeIndex ) -> Result< Option< T >, E >
        {
            if !model.get_node( node_index ).inputs().all( |idx| result_for_node[ idx.raw() ].is_some() ) {
                return Ok(());
            }

            let mut owned_inputs: SmallVec< [_; 2] > = SmallVec::new();
            for input_node_index in model.get_node( node_index ).inputs() {
                let result = &mut result_for_node[ input_node_index.raw() ];
                let refcount = &mut result.as_mut().unwrap().1;
                if *refcount == 1 {
                    owned_inputs.push( Some( result.take().unwrap().0 ) );
                } else {
                    *refcount -= 1;
                    owned_inputs.push( None );
                }
            }

            let inputs: SmallVec< [_; 2] > = model.get_node( node_index )
                .inputs()
                .zip( owned_inputs.into_iter() )
                .map( |(input_node_index, owned_input)| {
                    if let Some( input ) = owned_input {
                        MaybeOwned::Owned( input )
                    } else {
                        MaybeOwned::Borrowed( &result_for_node[ input_node_index.raw() ].as_ref().unwrap().0 )
                    }
                })
                .collect();

            let result = match callback( model, &inputs, node_index )? {
                Some( value ) => value,
                None => return Ok(())
            };

            mem::drop( inputs );

            let outputs = model.get_node( node_index ).outputs().clone();
            result_for_node[ node_index.raw() ] = Some( (result, outputs.len()) );

            for output_node_index in outputs {
                process( model, output_node_index, result_for_node, callback )?;
            }

            Ok(())
        }

        let mut result_for_node = Vec::with_capacity( self.nodes.len() );
        for _ in 0..self.nodes.len() {
            result_for_node.push( None );
        }

        let initial_node_count = self.initial_nodes.len();
        for index in 0..initial_node_count {
            let node_index = self.initial_nodes[ index ];
            process( self, node_index, &mut result_for_node, &mut callback )?;
        }

        Ok(())
    }

    fn assert_internal_invariants_for_node( &self, done: &mut HashSet< NodeIndex >, node_index: NodeIndex ) {
        done.insert( node_index );

        let node = self.get_node( node_index );
        for &output_node_index in node.outputs() {
            assert!( self.get_node( output_node_index ).inputs().any( |input_node_index| input_node_index == node_index ) );
            self.assert_internal_invariants_for_node( done, output_node_index );
        }

        for input_node_index in node.inputs() {
            assert!( self.get_node( input_node_index ).outputs().iter().any( |&output_node_index| output_node_index == node_index ) );
        }
    }

    fn assert_internal_invariants( &self ) {
        let initial_indexes: Vec< _ > = self.nodes.iter().enumerate().filter_map( |(node_index, node)| {
            let node_index = NodeIndex( node_index );
            match *node {
                Node::Input { .. } |
                Node::NullaryNode { .. } => Some( node_index ),
                Node::UnaryNode { .. } |
                Node::BinaryNode { .. } => None
            }
        }).collect();

        assert_eq!( initial_indexes, self.initial_nodes );

        let mut done = HashSet::new();
        for node_index in self.initial_node_indexes() {
            self.assert_internal_invariants_for_node( &mut done, node_index );
        }

        assert_eq!( done.len(), self.nodes.len() );
    }

    pub(crate) fn validate( &self ) -> Result< (), InvalidModelError > {
        fn check_weights(
            node_index: NodeIndex,
            layer_kind: &'static str,
            name: &Name,
            weights: &Option< Weights >,
            expected_weight_count: usize
        ) -> Result< (), InvalidModelError >
        {
            let weights = weights.as_ref()
                .ok_or_else( || InvalidModelError::MissingWeights {
                    node_index, layer_kind, layer_name: name.clone()
                })?;

            if weights.iter().cloned().any( |value| value.is_nan() || value.is_infinite() ) {
                return Err( InvalidModelError::InvalidWeights { node_index, layer_kind, layer_name: name.clone() } )
            }

            if weights.len() != expected_weight_count {
                return Err( InvalidModelError::InvalidWeightCount {
                    node_index,
                    layer_kind,
                    layer_name: name.clone(),
                    weight_count: weights.len(),
                    expected_weight_count
                });
            }

            Ok(())
        }

        fn check_input_shapes_are_the_same(
            node_index: NodeIndex,
            layer_kind: &'static str,
            layer_name: &Name,
            input_shape_1: &Shape,
            input_shape_2: &Shape
        ) -> Result< (), InvalidModelError >
        {
            if input_shape_1 != input_shape_2 {
                return Err( InvalidModelError::ExpectedEqualInputShapes {
                    node_index,
                    layer_kind,
                    layer_name: layer_name.clone(),
                    input_shape_1: input_shape_1.clone(),
                    input_shape_2: input_shape_2.clone()
                });
            }

            Ok(())
        }

        self.assert_internal_invariants();

        let mut name_to_index = HashMap::new();

        for node_index in self.node_indexes() {
            let node = self.get_node( node_index );
            let weight_count = self.weight_count_of( node );
            let is_last = self.outputs.iter().cloned().any( |index| node_index == index );

            if let Some( name ) = node.name() {
                if let Some( &other_node_index ) = name_to_index.get( name ) {
                    let other_node = self.get_node( other_node_index );
                    return Err( InvalidModelError::DuplicateName {
                        node_index_1: node_index,
                        layer_kind_1: node.type_name(),
                        node_index_2: other_node_index,
                        layer_kind_2: other_node.type_name(),
                        layer_name: name.clone()
                    });
                }
                name_to_index.insert( name, node_index );
            }

            let layer_kind = node.type_name();
            match node {
                Node::UnaryNode { ref layer, .. } => {
                    let input_shape = self.input_shapes_of( node ).next().unwrap();
                    match *layer {
                        AnyUnaryLayer::Dense( ref layer ) => {
                            check_weights( node_index, layer_kind, &layer.name, &layer.weights, weight_count )?;
                        },
                        AnyUnaryLayer::Convolution( ref layer ) => {
                            check_weights( node_index, layer_kind, &layer.name, &layer.weights, weight_count )?;
                        },
                        AnyUnaryLayer::Reshape( ref layer ) => {
                            if layer.shape.product() != input_shape.product() {
                                return Err( InvalidModelError::InvalidReshape {
                                    node_index,
                                    layer_name: layer.name().clone(),
                                    input_shape,
                                    output_shape: layer.shape.clone()
                                });
                            }
                        },
                        AnyUnaryLayer::IntoCategory( _ ) => {
                            if !is_last {
                                return Err( InvalidModelError::LayerShouldBeTheLastLayer {
                                    node_index,
                                    layer_kind,
                                    layer_name: layer.name().clone()
                                });
                            }
                        },
                        _ => {}
                    }
                },
                Node::BinaryNode { ref layer, .. } => {
                    let mut input_shapes = self.input_shapes_of( node );
                    let input_shape_1 = input_shapes.next().unwrap();
                    let input_shape_2 = input_shapes.next().unwrap();
                    match *layer {
                        AnyBinaryLayer::Add( ref layer ) => {
                            check_input_shapes_are_the_same( node_index, layer_kind, &layer.name, &input_shape_1, &input_shape_2 )?;
                        },
                        AnyBinaryLayer::Mul( ref layer ) => {
                            check_input_shapes_are_the_same( node_index, layer_kind, &layer.name, &input_shape_1, &input_shape_2 )?;
                        }
                    }
                },
                _ => {}
            }
        }

        Ok(())
    }
}

impl ModelBuilder {
    fn new() -> Self {
        let model = Model {
            nodes: Vec::new(),
            outputs: Vec::new(),
            inputs: Vec::new(),
            initial_nodes: Vec::new()
        };

        ModelBuilder( RefCell::new( model ) )
    }

    fn build( self ) -> Model {
        self.0.into_inner()
    }

    pub(crate) fn get_node( &self, node_index: NodeIndex ) -> NodeRef {
        NodeRef {
            builder: self,
            node_index
        }
    }

    pub fn add_input( &self, shape: Shape ) -> NodeRef {
        let mut model = self.0.borrow_mut();

        let input_index = model.inputs.len();
        let node = Node::Input {
            outputs: Default::default(),
            input_index,
            shape
        };

        let node_index = NodeIndex::new( model.nodes.len() );
        model.nodes.push( node );
        model.inputs.push( node_index );
        model.initial_nodes.push( node_index );

        NodeRef {
            builder: self,
            node_index
        }
    }

    pub fn add_output( &self, node: NodeRef ) {
        let mut model = self.0.borrow_mut();
        model.outputs.push( node.node_index );
    }

    pub(crate) fn set_output( &self, index: usize, node: NodeRef ) {
        let mut model = self.0.borrow_mut();
        model.outputs[ index ] = node.node_index;
    }

    fn add_nullary_layer( &self, layer: AnyNullaryLayer ) -> NodeRef {
        let mut model = self.0.borrow_mut();
        let new_node_index = NodeIndex::new( model.nodes.len() );

        let new_node = Node::NullaryNode {
            outputs: Default::default(),
            layer
        };

        model.nodes.push( new_node );
        model.initial_nodes.push( new_node_index );

        NodeRef {
            builder: self,
            node_index: new_node_index
        }
    }
}

#[must_use]
#[derive(Clone)]
pub struct NodeRef< 'a > {
    builder: &'a ModelBuilder,
    node_index: NodeIndex
}

impl< 'a > NodeRef< 'a > {
    pub fn add_as_output( self ) {
        self.builder.add_output( self )
    }

    pub fn chain< T >( self, layer: T ) -> Self where T: UnaryLayerList {
        layer.collapse_into_node( self )
    }

    pub fn chain_into_first_input< T >( self, second_input: NodeRef< 'a >, layer: T ) -> Self where T: BinaryLayer {
        layer.into_node( self, second_input )
    }

    fn chain_with_unary_layer( self, layer: AnyUnaryLayer ) -> NodeRef< 'a > {
        let mut model = self.builder.0.borrow_mut();
        let new_node_index = NodeIndex::new( model.nodes.len() );

        let node = model.get_node_mut( self.node_index );
        node.outputs_mut().push( new_node_index );

        let new_node = Node::UnaryNode {
            outputs: Default::default(),
            input: self.node_index,
            output_shape: layer.output_shape( &node.output_shape() ),
            layer
        };

        model.nodes.push( new_node );
        NodeRef {
            builder: self.builder,
            node_index: new_node_index
        }
    }

    fn chain_with_binary_layer(
        self,
        second_input: NodeRef< 'a >,
        layer: AnyBinaryLayer
    ) -> NodeRef< 'a >
    {
        assert_eq!( self.builder as *const _, second_input.builder as *const _ );
        let mut model = self.builder.0.borrow_mut();
        let new_node_index = NodeIndex::new( model.nodes.len() );

        let node_index_1 = self.node_index;
        let node_index_2 = second_input.node_index;

        let node_1 = model.get_node_mut( node_index_1 );
        node_1.outputs_mut().push( new_node_index );
        let input_shape_1 = node_1.output_shape();

        let node_2 = model.get_node_mut( node_index_2 );
        node_2.outputs_mut().push( new_node_index );
        let input_shape_2 = node_2.output_shape();

        let new_node = Node::BinaryNode {
            outputs: Default::default(),
            inputs: (node_index_1, node_index_2),
            output_shape: layer.output_shape( &input_shape_1, &input_shape_2 ),
            layer
        };

        model.nodes.push( new_node );
        NodeRef {
            builder: self.builder,
            node_index: new_node_index
        }
    }

    pub(crate) fn node_index( &self ) -> NodeIndex {
        self.node_index
    }

    pub(crate) fn outputs( &self ) -> Vec< Self > {
        let model = self.builder.0.borrow();
        model.get_node( self.node_index ).outputs().into_iter().map( |&node_index| {
            NodeRef {
                builder: self.builder,
                node_index
            }
        }).collect()
    }

    pub(crate) fn replace_input( &self, old: NodeRef, new: NodeRef ) {
        let mut model = self.builder.0.borrow_mut();

        assert_eq!(
            model.get_node( old.node_index ).output_shape(),
            model.get_node( new.node_index ).output_shape()
        );

        assert_eq!(
            model.get_node( old.node_index ).output_type(),
            model.get_node( new.node_index ).output_type()
        );

        if cfg!( debug_assertions ) {
            model.assert_internal_invariants();
        }

        for input_node_index in model.get_node( self.node_index ).inputs() {
            if input_node_index == old.node_index {
                let outputs = model.get_node_mut( input_node_index ).outputs_mut();
                let position = outputs.iter().position( |&output_node_index| output_node_index == self.node_index ).unwrap();
                outputs.swap_remove( position );
            }
        }

        let mut count = 0;
        for input_node_index in model.get_node_mut( self.node_index ).inputs_mut() {
            if *input_node_index == old.node_index {
                *input_node_index = new.node_index;
                count += 1;
            }
        }

        for _ in 0..count {
            model.get_node_mut( new.node_index ).outputs_mut().push( self.node_index );
        }

        if cfg!( debug_assertions ) {
            model.assert_internal_invariants();
        }
    }
}

macro_rules! impl_unary_layer_list {
    (@impl $($type:ident)*) => {
        impl< $($type),+ > UnaryLayerList for ($($type,)+) where $($type: UnaryLayerList),+ {
            fn collapse_into_node( self, mut input: NodeRef ) -> NodeRef {
                $(
                    input = access_tuple!( self, $type ).collapse_into_node( input );
                )+
                input
            }
        }
    };

    (@call_1 [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_unary_layer_list!( @impl $($type)* );
        impl_unary_layer_list!( @call_1 [$($dummy_type)*] [$($type)* $lhs] );
    };

    (@call_1 [] [$($type:ident)*]) => {};

    (@call [$lhs:ident $($dummy_type:ident)*] [$($type:ident)*]) => {
        impl_unary_layer_list!( @call_1 [$($dummy_type)*] [$lhs $($type)*] );
    };

    () => {
        impl_unary_layer_list!(
            @call
                [
                    T00 T01 T02 T03 T04 T05 T06 T07 T08 T09
                    T10 T11 T12 T13 T14 T15 T16 T17 T18 T19
                    T20 T21 T22 T23 T24 T25 T26 T27 T28 T29
                ]
                []
        );
    };
}

impl_unary_layer_list!();
