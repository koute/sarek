use {
    byteorder::{
        ReadBytesExt
    },
    log::{
        info
    },
    sarek::{
        DataSet,
        SliceSource
    },
    std::{
        fs::{
            File
        },
        io::{
            self,
            Read
        },
        path::{
            Path,
            PathBuf
        }
    }
};

fn load_from_stream< F >( mut fp: F, images: &mut Vec< f32 >, labels: &mut Vec< u32 > ) -> Result< (), io::Error > where F: Read {
    let width = 32;
    let height = 32;
    let channels = 3;
    let image_size = width * height * channels;
    let image_count = 10000;

    assert_eq!( labels.len(), images.len() / image_size );

    labels.reserve( image_count );
    images.reserve( image_count * image_size );

    let labels_position = labels.len();
    let images_position = images.len();
    unsafe {
        labels.set_len( labels_position + image_count );
        images.set_len( images_position + image_count * image_size );
    }

    let labels = &mut labels[ labels_position.. ];
    let images = &mut images[ images_position.. ];

    let mut buffer = [0; 3072];
    assert_eq!( buffer.len(), image_size );

    for index in 0..image_count {
        labels[ index ] = fp.read_u8()? as u32;
        fp.read_exact( &mut buffer )?;

        let image = &mut images[ index * image_size..(index + 1) * image_size ];
        for (out, value) in image.iter_mut().zip( buffer.iter().cloned() ) {
            *out = value as f32 / 255.0 - 0.5;
        }
    }

    Ok(())
}

fn load_from_file< P >( path: P, images: &mut Vec< f32 >, labels: &mut Vec< u32 > ) -> Result< (), io::Error > where P: AsRef< Path > {
    let path = path.as_ref();

    info!( "Loading data from {:?}...", path );
    let fp = File::open( path )?;
    load_from_stream( fp, images, labels )
}

fn to_data_source( images: Vec< f32 >, labels: Vec< u32 > ) -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    let width = 32;
    let height = 32;
    let channels = 3;

    let images = SliceSource::from( (width, height, channels).into(), images );
    let labels = SliceSource::from( 1.into(), labels );

    DataSet::new( images, labels )
}

type VecSource< T > = SliceSource< T, Vec< T > >;

fn data_path() -> PathBuf {
    Path::new( env!( "CARGO_MANIFEST_DIR" ) )
        .join( "examples" )
        .join( "cifar10" )
        .join( "data" )
        .join( "cifar-10-batches-bin" )
}

pub fn load_training_data() -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    info!( "Starting to load the training data set..." );

    let mut images = Vec::new();
    let mut labels = Vec::new();
    let path = data_path();
    load_from_file( path.join( "data_batch_1.bin" ), &mut images, &mut labels ).unwrap();
    load_from_file( path.join( "data_batch_2.bin" ), &mut images, &mut labels ).unwrap();
    load_from_file( path.join( "data_batch_3.bin" ), &mut images, &mut labels ).unwrap();
    load_from_file( path.join( "data_batch_4.bin" ), &mut images, &mut labels ).unwrap();
    load_from_file( path.join( "data_batch_5.bin" ), &mut images, &mut labels ).unwrap();
    let data = to_data_source( images, labels );

    info!( "Finished loading the training data set!" );
    data
}

pub fn load_test_data() -> DataSet< VecSource< f32 >, VecSource< u32 > > {
    info!( "Starting to load the test data set..." );

    let mut images = Vec::new();
    let mut labels = Vec::new();
    load_from_file( data_path().join( "test_batch.bin" ), &mut images, &mut labels ).unwrap();
    let data = to_data_source( images, labels );

    info!( "Finished loading the test data set!" );
    data
}
