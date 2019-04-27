use {
    std::{
        error::{
            Error
        },
        fmt
    },
    crate::{
        backend::{
            keras
        }
    }
};

pub trait CloneableError: Error + Send + Sync {
    fn clone( &self ) -> Box< dyn CloneableError >;
}

impl< T > CloneableError for T where T: Error + Clone + Send + Sync + 'static {
    fn clone( &self ) -> Box< dyn CloneableError > {
        Box::new( self.clone() )
    }
}

#[derive(Display)]
#[display(fmt = "context creation failed: {}", "_0")]
pub struct ContextCreationError( pub(crate) Box< dyn CloneableError > );

impl Clone for ContextCreationError {
    fn clone( &self ) -> Self {
        ContextCreationError( self.0.clone() )
    }
}

impl fmt::Debug for ContextCreationError {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        fmt::Debug::fmt( &self.0, fmt )
    }
}

impl Error for ContextCreationError {}

#[derive(Clone)]
pub enum ContextKind {
    Keras( keras::Context )
}

#[derive(Clone)]
pub struct Context( pub(crate) ContextKind );

impl Context {
    pub fn new() -> Result< Self, ContextCreationError > {
        let ctx = keras::Context::new()?;
        Ok( Context( ContextKind::Keras( ctx ) ) )
    }

    #[doc(hidden)]
    pub fn is_using_tensorflow( &self ) -> bool {
        match self.0 {
            ContextKind::Keras( _ ) => true
        }
    }
}
