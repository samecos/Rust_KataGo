//! Simple object-pooling allocator.
//!
//! Corresponds to `cpp/core/simpleallocator.h`.

use std::cell::RefCell;
use std::collections::HashMap;

/// A pool allocator that reuses buffers of the same size.
pub struct SimpleAllocator<T> {
    allocate: Box<dyn Fn(usize) -> T>,
    release: Box<dyn Fn(T)>,
    buffers: HashMap<usize, Vec<T>>,
}

impl<T> SimpleAllocator<T> {
    /// Create a new allocator with the given allocate and release callbacks.
    pub fn new<A, R>(allocate: A, release: R) -> Self
    where
        A: Fn(usize) -> T + 'static,
        R: Fn(T) + 'static,
    {
        Self {
            allocate: Box::new(allocate),
            release: Box::new(release),
            buffers: HashMap::new(),
        }
    }
}

impl<T> Drop for SimpleAllocator<T> {
    fn drop(&mut self) {
        for buf in self.buffers.values_mut().flatten() {
            // SAFETY: we take ownership of each buffer by swapping it out with
            // a value produced by the allocate callback. This avoids requiring
            // `T: Default`. The released buffer is then dropped.
            let taken = std::mem::replace(buf, (self.allocate)(0));
            (self.release)(taken);
        }
    }
}

/// An RAII buffer borrowed from a [`SimpleAllocator`].
pub struct SizedBuf<'a, T> {
    size: usize,
    buf: Option<T>,
    allocator: &'a RefCell<SimpleAllocator<T>>,
}

impl<'a, T> SizedBuf<'a, T> {
    /// Allocate or reuse a buffer of `size` elements from `allocator`.
    pub fn new(allocator: &'a RefCell<SimpleAllocator<T>>, size: usize) -> Self {
        let buf = {
            let mut alloc = allocator.borrow_mut();
            let vec = alloc.buffers.entry(size).or_default();
            if vec.is_empty() {
                (alloc.allocate)(size)
            } else {
                vec.pop().expect("SizedBuf::new: empty buffer vector")
            }
        };
        Self {
            size,
            buf: Some(buf),
            allocator,
        }
    }

    /// Return a reference to the buffer.
    pub fn buf(&self) -> &T {
        self.buf
            .as_ref()
            .expect("SizedBuf::buf: buffer already released")
    }

    /// Return a mutable reference to the buffer.
    pub fn buf_mut(&mut self) -> &mut T {
        self.buf
            .as_mut()
            .expect("SizedBuf::buf_mut: buffer already released")
    }
}

impl<T> Drop for SizedBuf<'_, T> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.allocator
                .borrow_mut()
                .buffers
                .entry(self.size)
                .or_default()
                .push(buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_reuses_buffers() {
        let allocations: std::rc::Rc<std::cell::RefCell<i32>> =
            std::rc::Rc::new(std::cell::RefCell::new(0));
        let releases: std::rc::Rc<std::cell::RefCell<i32>> =
            std::rc::Rc::new(std::cell::RefCell::new(0));

        let allocations_a = std::rc::Rc::clone(&allocations);
        let releases_a = std::rc::Rc::clone(&releases);
        let allocator = RefCell::new(SimpleAllocator::new(
            move |size| {
                *allocations_a.borrow_mut() += 1;
                vec![0i32; size]
            },
            move |_buf| {
                *releases_a.borrow_mut() += 1;
            },
        ));

        {
            let mut b1 = SizedBuf::new(&allocator, 4);
            b1.buf_mut().fill(1);
        }
        {
            let b2 = SizedBuf::new(&allocator, 4);
            // Reused buffer retains old contents because release didn't clear.
            assert_eq!(b2.buf().len(), 4);
        }

        assert_eq!(*allocations.borrow(), 1);
        assert_eq!(*releases.borrow(), 0);

        drop(allocator);
        assert_eq!(*releases.borrow(), 1);
    }

    #[test]
    fn test_allocator_different_sizes() {
        let allocator = RefCell::new(SimpleAllocator::new(|size| vec![0u8; size], |_buf| {}));

        {
            let _b1 = SizedBuf::new(&allocator, 4);
            let _b2 = SizedBuf::new(&allocator, 8);
        }

        let alloc = allocator.borrow();
        assert_eq!(alloc.buffers.get(&4).unwrap().len(), 1);
        assert_eq!(alloc.buffers.get(&8).unwrap().len(), 1);
    }
}
