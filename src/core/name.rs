use {
    std::{
        fmt,
        io::{
            Write
        },
        hash::{
            Hash,
            Hasher
        },
        str
    }
};

/// A name.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Name {
    kind: NameKind
}

impl Name {
    /// Creates a new name.
    pub fn new( name: &str ) -> Name {
        Name {
            kind: NameKind::String( name.into() )
        }
    }

    /// Creates a new unique name.
    pub fn new_unique() -> Name {
        Name {
            kind: NameKind::Uuid( uuid::Uuid::new_v4() )
        }
    }
}

#[derive(Clone, Debug)]
enum NameKind {
    String( String ),
    Uuid( uuid::Uuid )
}

impl NameKind {
    fn as_str< F, R >( &self, callback: F ) -> R where F: FnOnce( &str ) -> R {
        match self {
            NameKind::String( ref name ) => callback( name ),
            NameKind::Uuid( ref uuid ) => {
                let mut buffer: [u8; 36] = [0; 36];
                write!( &mut buffer[..], "{}", uuid ).unwrap();
                callback( str::from_utf8( &buffer ).unwrap() )
            }
        }
    }
}

impl fmt::Display for Name {
    fn fmt( &self, fmt: &mut fmt::Formatter ) -> fmt::Result {
        match self.kind {
            NameKind::String( ref name ) => write!( fmt, "{}", name ),
            NameKind::Uuid( ref name ) => write!( fmt, "{}", name )
        }
    }
}

impl PartialEq for NameKind {
    fn eq( &self, rhs: &NameKind ) -> bool {
        self.as_str( |lhs| {
            rhs.as_str( |rhs| {
                lhs == rhs
            })
        })
    }
}

impl Hash for NameKind {
    fn hash< H >( &self, state: &mut H ) where H: Hasher {
        self.as_str( |name| {
            name.hash( state );
        });
    }
}

impl Eq for NameKind {}

impl< 'a > From< &'a str > for Name {
    fn from( name: &'a str ) -> Self {
        Name::new( name )
    }
}

impl< 'a > From< &'a Name > for Name {
    fn from( name: &'a Name ) -> Self {
        name.clone()
    }
}

#[test]
fn test_names_are_compared_by_their_string_representations() {
    let name_1 = Name::new_unique();
    let name_2 = Name::new( &format!( "{}", name_1 ) );
    assert_eq!( name_1, name_2 );
}
