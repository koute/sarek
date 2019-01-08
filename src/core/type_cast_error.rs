use {
    std::{
        error,
        fmt
    },
    crate::{
        core::{
            data_type::{
                Type
            }
        }
    }
};

pub(crate) enum SourceTy {
    Type( Type ),
    String( String )
}

impl fmt::Display for SourceTy {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        match *self {
            SourceTy::Type( ref ty ) => ty.fmt( fmt ),
            SourceTy::String( ref ty ) => ty.fmt( fmt )
        }
    }
}

impl From< Type > for SourceTy {
    fn from( ty: Type ) -> Self {
        SourceTy::Type( ty )
    }
}

impl From< String > for SourceTy {
    fn from( ty: String ) -> Self {
        SourceTy::String( ty )
    }
}

pub struct TypeCastError< T > {
    pub(crate) source: &'static str,
    pub(crate) target: &'static str,
    pub(crate) source_ty: SourceTy,
    pub(crate) target_ty: Type,
    pub(crate) obj: T
}

impl< T > TypeCastError< T > {
    /// Recovers the object for which the cast has failed.
    pub fn recover( self ) -> T {
        self.obj
    }
}

impl< T > fmt::Display for TypeCastError< T > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "tried to cast {} of type '{}' into {} of type '{}'", self.source, self.source_ty, self.target, self.target_ty )
    }
}

impl< T > fmt::Debug for TypeCastError< T > {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "TypeCastError {{ source_ty: '{}', target_ty: '{}' }}", self.source_ty, self.target_ty )
    }
}

impl< T > error::Error for TypeCastError< T > {}
