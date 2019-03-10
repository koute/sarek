#![feature(non_exhaustive)]
#![feature(core_intrinsics)]
#![feature(specialization)]
#![allow(clippy::into_iter_on_ref)]
#![allow(clippy::into_iter_on_array)]
#![allow(clippy::new_without_default)]
#![allow(clippy::new_without_default_derive)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::type_complexity)]

#[macro_use]
extern crate derive_more;

#[macro_use]
mod macros;

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
                LayerMaxPooling,
                LayerMultiply,
                LayerReshape,
                LayerShift,
                LayerSoftmax
            }
        }
    };
}

pub mod optimizers {
    pub use crate::{
        nn::{
            optimizers::{
                OptimizerNadam,
                OptimizerSGD
            }
        }
    };
}

pub use crate::{
    backend::{
        Context,
        ModelInstance
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
            DataSource,
            DataSourceExt
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
        trainer::{
            Trainer
        },
        training_opts::{
            TrainingOpts
        }
    }
};
