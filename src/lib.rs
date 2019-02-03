#![feature(non_exhaustive)]
#![feature(core_intrinsics)]
#![allow(clippy::into_iter_on_ref)]
#![allow(clippy::into_iter_on_array)]
#![allow(clippy::new_without_default)]
#![allow(clippy::new_without_default_derive)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::type_complexity)]

mod backend;
mod core;
mod nn;

pub mod layers {
    pub use crate::{
        nn::{
            layers::{
                LayerActivation,
                LayerConvolution,
                LayerDense,
                LayerDropout,
                LayerIntoCategory,
                LayerReshape,
                LayerSoftmax
            }
        }
    };
}

pub mod optimizers {
    pub use crate::{
        nn::{
            optimizers::{
                OptimizerAdam,
                OptimizerSGD
            }
        }
    };
}

pub use crate::{
    backend::{
        keras::{
            Context,
            ModelInstance,
            Trainer
        }
    },
    core::{
        array::{
            ArrayMut,
            ArrayRef,
            ToArrayRef,
            TypedArrayMut,
            TypedArrayRef
        },
        data_set::{
            DataSet
        },
        data_source::{
            DataSource
        },
        data_type::{
            DataType,
            Type
        },
        shape::{
            Shape
        },
        slice_source::{
            SliceSource
        },
        name::{
            Name
        }
    },
    nn::{
        activation::{
            Activation
        },
        layers::{
            Layer,
            LayerPrototype,
            IntoLayerIter
        },
        loss::{
            Loss
        },
        model::{
            Model
        },
        optimizers::{
            Optimizer
        },
        training_opts::{
            TrainingOpts
        }
    }
};
