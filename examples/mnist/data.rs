use {
    byteorder::{
        BigEndian,
        ReadBytesExt
    },
    flate2::{
        read::{
            GzDecoder
        }
    },
    log::{
        info
    },
    sarek::{
        DataSet,
        SliceSource
    },
    std::{
        io::{
            Cursor,
            Read
        }
    }
};

static MNIST_TRAINING_IMAGES: &[u8] = include_bytes!( "data/train-images-idx3-ubyte.gz" );
static MNIST_TRAINING_LABELS: &[u8] = include_bytes!( "data/train-labels-idx1-ubyte.gz" );
static MNIST_TEST_IMAGES: &[u8] = include_bytes!( "data/t10k-images-idx3-ubyte.gz" );
static MNIST_TEST_LABELS: &[u8] = include_bytes!( "data/t10k-labels-idx1-ubyte.gz" );

fn create_reader( data: &'static [u8] ) -> Cursor< Vec< u8 > > {
    let mut decoder = GzDecoder::new( data );
    let mut data = Vec::new();
    decoder.read_to_end( &mut data ).unwrap();

    Cursor::new( data )
}

fn load_images( images: &'static [u8] ) -> (u32, u32, u32, Vec< u8 >) {
    let mut reader = create_reader( images );
    let magic = reader.read_u32::< BigEndian >().unwrap();
    assert_eq!( magic, 0x00000803 );

    let count = reader.read_u32::< BigEndian >().unwrap();
    let height = reader.read_u32::< BigEndian >().unwrap();
    let width = reader.read_u32::< BigEndian >().unwrap();

    let position = reader.position() as usize;
    let data = reader.into_inner()[ position.. ].to_vec();

    (count, width, height, data)
}

fn load_labels( labels: &'static [u8] ) -> (u32, Vec< u8 >) {
    let mut reader = create_reader( labels );
    let magic = reader.read_u32::< BigEndian >().unwrap();
    assert_eq!( magic, 0x00000801 );

    let count = reader.read_u32::< BigEndian >().unwrap();
    let position = reader.position() as usize;
    let data = reader.into_inner()[ position.. ].to_vec();

    (count, data)
}

type VecSource< T > = SliceSource< T, Vec< T > >;

fn load( raw_labels: &'static [u8], raw_images: &'static [u8] ) -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    let (label_count, labels) = load_labels( raw_labels );
    let (image_count, width, height, image_data) = load_images( raw_images );

    assert_eq!( label_count, image_count );

    let images: Vec< f32 > = image_data.into_iter().map( |value| value as f32 / 255.0 ).collect();
    let categories: Vec< u32 > = labels.iter().map( |&category| category as u32 ).collect();

    let images = SliceSource::from( (width as usize, height as usize).into(), images );
    let categories = SliceSource::from( 1.into(), categories );

    DataSet::new( images, categories )
}

pub fn load_training_data_set() -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    info!( "Starting to load the training data set..." );
    let data = load( MNIST_TRAINING_LABELS, MNIST_TRAINING_IMAGES );
    info!( "Finished loading the training data set!" );
    data
}

pub fn load_test_data_set() -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    info!( "Starting to load the test data set..." );
    let data = load( MNIST_TEST_LABELS, MNIST_TEST_IMAGES );
    info!( "Finished loading the test data set!" );
    data
}
