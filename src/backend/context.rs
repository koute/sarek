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

#[derive(Display)]
#[display(fmt = "context creation failed: {}", "_0")]
pub struct ContextCreationError( pub(crate) Box< dyn Error + Send > );

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
}
