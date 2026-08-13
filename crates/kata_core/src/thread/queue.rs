//! Thread-safe queue and priority queue.
//!
//! Corresponds to `cpp/core/threadsafequeue.h` and
//! `cpp/core/threadsafequeue.cpp`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Condvar, Mutex};

/// A bounded, thread-safe FIFO queue with close/read-only semantics.
pub struct ThreadSafeQueue<T> {
    mutex: Mutex<QueueInner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
}

struct QueueInner<T> {
    max_size: usize,
    closed: bool,
    read_only: bool,
    head_idx: usize,
    elts_dequeue: Vec<Option<T>>,
    elts_enqueue: Vec<Option<T>>,
    size: usize,
}

impl<T> ThreadSafeQueue<T> {
    /// Create an unbounded queue.
    pub fn new() -> Self {
        Self::with_max_size(usize::MAX)
    }

    /// Create a queue with the given maximum size.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            mutex: Mutex::new(QueueInner {
                max_size,
                closed: false,
                read_only: false,
                head_idx: 0,
                elts_dequeue: Vec::new(),
                elts_enqueue: Vec::new(),
                size: 0,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    /// Reserve capacity in the internal buffers.
    pub fn reserve(&self, sz: usize) {
        let mut inner = self.mutex.lock().unwrap();
        inner.elts_dequeue.reserve(sz);
        inner.elts_enqueue.reserve(sz);
    }

    /// Return the number of elements in the queue.
    pub fn size(&self) -> usize {
        let inner = self.mutex.lock().unwrap();
        Self::size_unsync(&inner)
    }

    /// Return whether the queue is closed.
    pub fn is_closed(&self) -> bool {
        let inner = self.mutex.lock().unwrap();
        inner.closed
    }

    /// Return whether the queue is read-only.
    pub fn is_read_only(&self) -> bool {
        let inner = self.mutex.lock().unwrap();
        inner.read_only
    }

    /// Close the queue, dropping remaining elements and unblocking all waiters.
    pub fn close(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.closed = true;
        Self::clear_unsync(&mut inner);
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    /// Set the queue to read-only, unblocking all waiters.
    pub fn set_read_only(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.read_only = true;
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    /// Clear the read-only flag.
    pub fn unset_read_only(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.read_only = false;
    }

    fn size_unsync(inner: &QueueInner<T>) -> usize {
        inner.size
    }

    fn push_unsync(inner: &mut QueueInner<T>, elt: T) {
        inner.elts_enqueue.push(Some(elt));
        inner.size += 1;
    }

    fn pop_unsync(inner: &mut QueueInner<T>) -> T {
        if inner.head_idx >= inner.elts_dequeue.len() {
            debug_assert!(!inner.elts_enqueue.is_empty());
            inner.elts_dequeue.clear();
            std::mem::swap(&mut inner.elts_dequeue, &mut inner.elts_enqueue);
            inner.head_idx = 0;
        }
        let idx = inner.head_idx;
        inner.head_idx += 1;
        inner.size -= 1;
        inner.elts_dequeue[idx]
            .take()
            .expect("pop_unsync: empty slot")
    }

    fn clear_unsync(inner: &mut QueueInner<T>) {
        inner.elts_dequeue.clear();
        inner.elts_enqueue.clear();
        inner.head_idx = 0;
        inner.size = 0;
    }

    /// Push an element, blocking until space is available or the queue is
    /// closed/read-only. Returns `true` if the push succeeded.
    pub fn wait_push(&self, elt: T) -> bool {
        let mut inner = self.mutex.lock().unwrap();
        while !inner.closed && !inner.read_only && Self::size_unsync(&inner) >= inner.max_size {
            inner = self.not_full.wait(inner).unwrap();
        }
        if inner.closed || inner.read_only {
            return false;
        }
        let was_empty = Self::size_unsync(&inner) == 0;
        Self::push_unsync(&mut inner, elt);
        if was_empty {
            self.not_empty.notify_all();
        }
        true
    }

    /// Push an element without blocking, ignoring `max_size`. Returns `true` if
    /// the push succeeded.
    pub fn force_push(&self, elt: T) -> bool {
        let mut inner = self.mutex.lock().unwrap();
        if inner.closed || inner.read_only {
            return false;
        }
        let was_empty = Self::size_unsync(&inner) == 0;
        Self::push_unsync(&mut inner, elt);
        if was_empty {
            self.not_empty.notify_all();
        }
        true
    }

    /// Try to pop an element without blocking. Returns `true` if successful.
    pub fn try_pop(&self, buf: &mut T) -> bool
    where
        T: Clone,
    {
        let mut inner = self.mutex.lock().unwrap();
        if inner.closed {
            return false;
        }
        let size = Self::size_unsync(&inner);
        if size == 0 {
            return false;
        }
        if size == inner.max_size {
            self.not_full.notify_all();
        }
        *buf = Self::pop_unsync(&mut inner);
        true
    }

    /// Wait until an element is available or the queue is closed/read-only, then
    /// pop it into `buf`. Returns `true` if successful.
    pub fn wait_pop(&self, buf: &mut T) -> bool
    where
        T: Clone,
    {
        let mut inner = self.mutex.lock().unwrap();
        while !inner.closed && !inner.read_only && Self::size_unsync(&inner) == 0 {
            inner = self.not_empty.wait(inner).unwrap();
        }
        if inner.closed {
            return false;
        }
        let size = Self::size_unsync(&inner);
        if size == 0 {
            return false;
        }
        if size == inner.max_size {
            self.not_full.notify_all();
        }
        *buf = Self::pop_unsync(&mut inner);
        true
    }

    /// Wait until elements are available or the queue is closed/read-only, then
    /// pop up to `n` elements into `buf`. Returns `true` if at least one element
    /// was popped.
    pub fn wait_pop_up_to_n(&self, buf: &mut Vec<T>, n: usize) -> bool
    where
        T: Clone,
    {
        let mut inner = self.mutex.lock().unwrap();
        while !inner.closed && !inner.read_only && Self::size_unsync(&inner) == 0 {
            inner = self.not_empty.wait(inner).unwrap();
        }
        if inner.closed {
            return false;
        }
        let size = Self::size_unsync(&inner);
        if size == 0 {
            return false;
        }
        let num_to_pop = size.min(n);
        for _ in 0..num_to_pop {
            buf.push(Self::pop_unsync(&mut inner));
        }
        if size >= inner.max_size && size < inner.max_size.saturating_add(n) {
            self.not_full.notify_all();
        }
        true
    }
}

impl<T> Default for ThreadSafeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Priority queue
// ---------------------------------------------------------------------------

/// A thread-safe max-priority queue storing `(key, value)` pairs ordered by
/// `key`. The pair with the largest key is popped first, matching C++
/// `std::priority_queue`.
pub struct ThreadSafePriorityQueue<K, V> {
    mutex: Mutex<PqInner<K, V>>,
    not_empty: Condvar,
    not_full: Condvar,
}

struct PqInner<K, V> {
    max_size: usize,
    closed: bool,
    read_only: bool,
    queue: BinaryHeap<OrdPair<K, V>>,
}

/// Wrapper to make `(K, V)` orderable by key for `BinaryHeap`.
#[derive(Debug, Clone)]
struct OrdPair<K, V>(K, V);

impl<K: PartialEq, V> PartialEq for OrdPair<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Eq, V> Eq for OrdPair<K, V> {}

impl<K: PartialOrd, V> PartialOrd for OrdPair<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: Ord, V> Ord for OrdPair<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<K, V> ThreadSafePriorityQueue<K, V> {
    /// Create an unbounded priority queue.
    pub fn new() -> Self
    where
        K: Ord,
    {
        Self::with_max_size(usize::MAX)
    }

    /// Create a priority queue with the given maximum size.
    pub fn with_max_size(max_size: usize) -> Self
    where
        K: Ord,
    {
        Self {
            mutex: Mutex::new(PqInner {
                max_size,
                closed: false,
                read_only: false,
                queue: BinaryHeap::new(),
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    /// Return the number of elements in the queue.
    pub fn size(&self) -> usize {
        let inner = self.mutex.lock().unwrap();
        inner.queue.len()
    }

    /// Return whether the queue is closed.
    pub fn is_closed(&self) -> bool {
        let inner = self.mutex.lock().unwrap();
        inner.closed
    }

    /// Return whether the queue is read-only.
    pub fn is_read_only(&self) -> bool {
        let inner = self.mutex.lock().unwrap();
        inner.read_only
    }

    /// Close the queue, dropping remaining elements and unblocking all waiters.
    pub fn close(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.closed = true;
        inner.queue.clear();
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    /// Set the queue to read-only, unblocking all waiters.
    pub fn set_read_only(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.read_only = true;
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    /// Clear the read-only flag.
    pub fn unset_read_only(&self) {
        let mut inner = self.mutex.lock().unwrap();
        inner.read_only = false;
    }

    /// Push an element, blocking until space is available or the queue is
    /// closed/read-only. Returns `true` if the push succeeded.
    pub fn wait_push(&self, key: K, value: V) -> bool
    where
        K: Ord,
    {
        let mut inner = self.mutex.lock().unwrap();
        while !inner.closed && !inner.read_only && inner.queue.len() >= inner.max_size {
            inner = self.not_full.wait(inner).unwrap();
        }
        if inner.closed || inner.read_only {
            return false;
        }
        let was_empty = inner.queue.is_empty();
        inner.queue.push(OrdPair(key, value));
        if was_empty {
            self.not_empty.notify_all();
        }
        true
    }

    /// Push an element without blocking, ignoring `max_size`. Returns `true` if
    /// the push succeeded.
    pub fn force_push(&self, key: K, value: V) -> bool
    where
        K: Ord,
    {
        let mut inner = self.mutex.lock().unwrap();
        if inner.closed || inner.read_only {
            return false;
        }
        let was_empty = inner.queue.is_empty();
        inner.queue.push(OrdPair(key, value));
        if was_empty {
            self.not_empty.notify_all();
        }
        true
    }

    /// Try to pop the highest-priority element without blocking. Returns `true`
    /// if successful.
    pub fn try_pop(&self, buf: &mut (K, V)) -> bool
    where
        K: Ord + Clone,
        V: Clone,
    {
        let mut inner = self.mutex.lock().unwrap();
        if inner.closed {
            return false;
        }
        let size = inner.queue.len();
        if size == 0 {
            return false;
        }
        if size == inner.max_size {
            self.not_full.notify_all();
        }
        let OrdPair(k, v) = inner.queue.pop().expect("try_pop: empty heap");
        *buf = (k, v);
        true
    }

    /// Wait until an element is available or the queue is closed/read-only, then
    /// pop the highest-priority element into `buf`. Returns `true` if successful.
    pub fn wait_pop(&self, buf: &mut (K, V)) -> bool
    where
        K: Ord + Clone,
        V: Clone,
    {
        let mut inner = self.mutex.lock().unwrap();
        while !inner.closed && !inner.read_only && inner.queue.is_empty() {
            inner = self.not_empty.wait(inner).unwrap();
        }
        if inner.closed {
            return false;
        }
        let size = inner.queue.len();
        if size == 0 {
            return false;
        }
        if size == inner.max_size {
            self.not_full.notify_all();
        }
        let OrdPair(k, v) = inner.queue.pop().expect("wait_pop: empty heap");
        *buf = (k, v);
        true
    }
}

impl<K, V> Default for ThreadSafePriorityQueue<K, V>
where
    K: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_queue_push_pop() {
        let q = ThreadSafeQueue::new();
        assert!(q.wait_push(1));
        assert!(q.wait_push(2));
        assert_eq!(q.size(), 2);

        let mut v = 0;
        assert!(q.wait_pop(&mut v));
        assert_eq!(v, 1);
        assert!(q.wait_pop(&mut v));
        assert_eq!(v, 2);
        assert!(!q.try_pop(&mut v));
    }

    #[test]
    fn test_queue_close() {
        let q = ThreadSafeQueue::new();
        assert!(q.wait_push(1));
        q.close();
        assert!(q.is_closed());
        assert_eq!(q.size(), 0);

        let mut v = 0;
        assert!(!q.wait_pop(&mut v));
        assert!(!q.wait_push(2));
    }

    #[test]
    fn test_queue_read_only() {
        let q = ThreadSafeQueue::new();
        assert!(q.wait_push(1));
        q.set_read_only();
        assert!(q.is_read_only());
        assert!(!q.wait_push(2));

        let mut v = 0;
        assert!(q.wait_pop(&mut v));
        assert_eq!(v, 1);
        assert!(!q.wait_pop(&mut v));
    }

    #[test]
    fn test_queue_bounded_blocks() {
        let q = Arc::new(ThreadSafeQueue::with_max_size(1));
        assert!(q.wait_push(1));

        let q2 = Arc::clone(&q);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            let mut v = 0;
            assert!(q2.wait_pop(&mut v));
            assert_eq!(v, 1);
        });

        assert!(q.wait_push(2));
        handle.join().unwrap();
    }

    #[test]
    fn test_queue_pop_up_to_n() {
        let q = ThreadSafeQueue::new();
        for i in 0..5 {
            assert!(q.wait_push(i));
        }
        let mut buf = Vec::new();
        assert!(q.wait_pop_up_to_n(&mut buf, 2));
        assert_eq!(buf, vec![0, 1]);
        assert_eq!(q.size(), 3);
    }

    #[test]
    fn test_priority_queue_order() {
        let q = ThreadSafePriorityQueue::new();
        assert!(q.wait_push(3, 'a'));
        assert!(q.wait_push(1, 'b'));
        assert!(q.wait_push(2, 'c'));

        let mut buf = (0, 'x');
        assert!(q.wait_pop(&mut buf));
        assert_eq!(buf, (3, 'a'));
        assert!(q.wait_pop(&mut buf));
        assert_eq!(buf, (2, 'c'));
        assert!(q.wait_pop(&mut buf));
        assert_eq!(buf, (1, 'b'));
        assert!(!q.try_pop(&mut buf));
    }
}
