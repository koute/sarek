use {
    std::{
        fmt
    }
};

#[derive(Clone, Debug)]
pub struct Loss {
    pub(crate) loss: f32,
    pub(crate) accuracy: Option< f32 >
}

impl fmt::Display for Loss {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        write!( fmt, "{}", self.loss )
    }
}

impl Loss {
    pub fn get( &self ) -> f32 {
        self.loss
    }

    pub fn accuracy( &self ) -> Option< f32 > {
        self.accuracy
    }
}
