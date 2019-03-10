#![feature(non_exhaustive)]
#![feature(core_intrinsics)]
#![feature(specialization)]
#![allow(clippy::into_iter_on_ref)]
#![allow(clippy::into_iter_on_array)]
#![allow(clippy::new_without_default)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::type_complexity)]
#![deny(clippy::iter_cloned_collect)]
#![deny(clippy::identity_conversion)]
#![deny(clippy::redundant_closure)]
#![deny(clippy::cast_lossless)]
#![deny(clippy::redundant_field_names)]
#![deny(clippy::clone_on_copy)]
#![deny(clippy::get_unwrap)]
#![deny(clippy::ptr_offset_with_cast)]
#![deny(clippy::needless_lifetimes)]
#![deny(bare_trait_objects)]
#![deny(unused_extern_crates)]
#![deny(ellipsis_inclusive_range_patterns)]
#![deny(explicit_outlives_requirements)]

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
                LayerAdd,
                LayerConstant,
                LayerConvolution,
                LayerDense,
                LayerDropout,
                LayerIntoCategory,
                LayerMaxPooling,
                LayerMul,
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
            DataSourceExt,
            DataSourceList,
            DataSourceListExt
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
            LayerPrototype
        },
        loss::{
            Loss
        },
        model::{
            Model
        },
        model::{
            BinaryLayer,
            ModelBuilder,
            NullaryLayer,
            UnaryLayer,
            UnaryLayerList
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
