use {
    std::{
        cell::{
            UnsafeCell
        },
        collections::{
            VecDeque
        },
        mem,
        panic,
        ptr,
        thread,
        sync::{
            atomic::{
                AtomicBool,
                Ordering
            },
            Arc
        }
    },
    parking_lot::{
        Mutex,
        Condvar
    }
};

trait CallOnce {
    fn call_once( &mut self );
}

impl< F: FnOnce() > CallOnce for Option< F > {
    fn call_once( &mut self ) {
        (self.take().unwrap())();
    }
}

struct Queue {
    deque: VecDeque< Box< CallOnce + Send > >,
    counter: usize
}

struct State {
    queue: Mutex< Queue >,
    condvar: Condvar,
    result_condvar: Condvar,
    running: AtomicBool
}

pub struct AsyncRunner {
    thread_handle: Option< thread::JoinHandle< () > >,
    state: Arc< State >
}

impl Drop for AsyncRunner {
    fn drop( &mut self ) {
        self.state.running.store( false, Ordering::SeqCst );
        self.state.condvar.notify_all();
        let thread_handle = self.thread_handle.take().unwrap();
        let _ = thread_handle.join();
    }
}

impl AsyncRunner {
    pub fn new() -> Self {
        let state = Arc::new( State {
            queue: Mutex::new( Queue {
                deque: VecDeque::new(),
                counter: 0
            }),
            condvar: Condvar::new(),
            result_condvar: Condvar::new(),
            running: AtomicBool::new( true )
        });

        let state_clone = state.clone();
        let thread_handle = thread::Builder::new().name( "async_runner".into() ).spawn( move || {
            let state = state_clone;

            let mut queue = state.queue.lock();
            loop {
                while let Some( mut command ) = queue.deque.pop_front() {
                    command.call_once();
                    queue.counter += 1;
                }

                state.result_condvar.notify_all();

                if !state.running.load( Ordering::SeqCst ) {
                    break;
                }

                state.condvar.wait( &mut queue );
            }
        }).unwrap();

        AsyncRunner {
            thread_handle: Some( thread_handle ),
            state
        }
    }

    pub fn execute_sync< R, F >( &self, callback: F ) -> R where F: FnOnce() -> R + Send, R: Send + 'static {
        let mut queue = self.state.queue.lock();
        let expected_counter = queue.counter + queue.deque.len() + 1;

        let cell: UnsafeCell< Option< thread::Result< R > > > = UnsafeCell::new( None );
        let return_ptr = cell.get() as usize;
        let callback: Box< CallOnce + Send > = Box::new( Some( move || {
            let return_value = panic::catch_unwind( panic::AssertUnwindSafe( move || {
                callback()
            }));

            unsafe {
                ptr::write( return_ptr as *mut Option< thread::Result< R > >, Some( return_value ) );
            }
        }));

        let callback: *mut (CallOnce + Send) = Box::into_raw( callback );
        #[allow(clippy::transmute_ptr_to_ptr)]
        let callback: *mut (CallOnce + Send + 'static) = unsafe { mem::transmute( callback ) };
        let callback: Box< CallOnce + Send + 'static > = unsafe { Box::from_raw( callback ) };

        queue.deque.push_back( callback );

        self.state.condvar.notify_all();
        while queue.counter < expected_counter {
            self.state.result_condvar.wait( &mut queue );
        }

        match cell.into_inner().unwrap() {
            Ok( value ) => value,
            Err( error ) => {
                panic::resume_unwind( error );
            }
        }
    }
}

#[test]
fn test_async_runner_works() {
    let runner = AsyncRunner::new();
    let result = runner.execute_sync( || {
        format!( "Hello world!" )
    });

    assert_eq!( result, "Hello world!" );
}

#[cfg(test)]
const PANIC_MESSAGE: &'static str = "test_async_runner -- manual panic";

#[cfg(test)]
fn set_panic_hook() {
    // This will prevent the panic messages from being printed out
    // when the test passes.
    static ONCE: parking_lot::Once = parking_lot::ONCE_INIT;
    ONCE.call_once( || {
        panic_control::chain_hook_ignoring_if( |text: &&'static str| {
            text.contains( PANIC_MESSAGE )
        });

        panic_control::chain_hook_ignoring_if( |text: &String| {
            text.contains( PANIC_MESSAGE )
        });
    });
}

#[test]
fn test_async_runner_proxies_panic_str() {
    use crate::core::utils::assert_panic;

    let runner = AsyncRunner::new();
    set_panic_hook();
    assert_panic( PANIC_MESSAGE, || {
        runner.execute_sync( || {
            panic!( PANIC_MESSAGE );
        });
    });
}

#[test]
fn test_async_runner_proxies_panic_owned_string() {
    use crate::core::utils::assert_panic;

    let runner = AsyncRunner::new();
    set_panic_hook();
    assert_panic( PANIC_MESSAGE, || {
        runner.execute_sync( || {
            panic!( PANIC_MESSAGE.to_owned() );
        });
    });
}
