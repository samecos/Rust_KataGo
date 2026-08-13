//! Integration tests for thread primitives.
//!
//! Corresponds to `cpp/core/threadtest.h` and `cpp/core/threadtest.cpp`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::thread;

    use crate::rng::Rand;
    use crate::thread::counter::{ThreadSafeCounter, WaitableFlag};
    use crate::thread::queue::ThreadSafeQueue;

    #[test]
    fn test_thread_coordination() {
        let flag = Arc::new(WaitableFlag::new());
        let counter0 = Arc::new(ThreadSafeCounter::new());
        let counter1 = Arc::new(ThreadSafeCounter::new());
        let counter2 = Arc::new(ThreadSafeCounter::new());
        let queue = Arc::new(ThreadSafeQueue::new());
        let queue2 = Arc::new(ThreadSafeQueue::new());
        let exit_count = Arc::new(ThreadSafeCounter::new());

        counter0.add(4);
        counter1.add(2);
        counter2.add(8);

        let flag_f = Arc::clone(&flag);
        let counter0_f = Arc::clone(&counter0);
        let counter1_f = Arc::clone(&counter1);
        let counter2_f = Arc::clone(&counter2);
        let queue_f = Arc::clone(&queue);
        let exit_count_f = Arc::clone(&exit_count);
        let f = thread::spawn(move || {
            flag_f.wait_until_false();
            thread::yield_now();
            assert!(queue_f.force_push(10));
            counter0_f.add(-4);
            counter1_f.add(-4);
            counter2_f.add(-4);

            flag_f.wait_until_true();
            thread::yield_now();
            assert!(queue_f.force_push(11));
            counter0_f.add(2);
            counter1_f.add(2);
            counter2_f.add(2);

            flag_f.wait_until_false();
            thread::yield_now();
            assert!(queue_f.force_push(12));
            counter0_f.add(-6);
            counter1_f.add(-6);
            counter2_f.add(-6);

            flag_f.wait_until_true();
            assert!(queue_f.force_push(13));
            thread::yield_now();
            assert!(queue_f.force_push(14));
            thread::yield_now();
            assert!(queue_f.force_push(15));

            flag_f.wait_until_false();
            thread::yield_now();
            queue_f.set_read_only();
            exit_count_f.add(1);
        });

        let counter0_c0 = Arc::clone(&counter0);
        let queue_c0 = Arc::clone(&queue);
        let exit_count_c0 = Arc::clone(&exit_count);
        let c0 = thread::spawn(move || {
            counter0_c0.wait_until_zero();
            assert!(queue_c0.force_push(16));
            exit_count_c0.add(1);
        });

        let counter1_c1 = Arc::clone(&counter1);
        let queue_c1 = Arc::clone(&queue);
        let exit_count_c1 = Arc::clone(&exit_count);
        let c1 = thread::spawn(move || {
            counter1_c1.wait_until_zero();
            assert!(queue_c1.force_push(17));
            exit_count_c1.add(1);
        });

        let counter2_c2 = Arc::clone(&counter2);
        let queue_c2 = Arc::clone(&queue);
        let exit_count_c2 = Arc::clone(&exit_count);
        let c2 = thread::spawn(move || {
            counter2_c2.wait_until_zero();
            assert!(queue_c2.force_push(18));
            exit_count_c2.add(1);
        });

        let flag_g = Arc::clone(&flag);
        let queue_g = Arc::clone(&queue);
        let queue2_g = Arc::clone(&queue2);
        let exit_count_g = Arc::clone(&exit_count);
        let g = thread::spawn(move || {
            let mut buf = 0;
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 10);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 16);
            flag_g.set(true);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 11);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 17);
            assert!(!queue_g.try_pop(&mut buf));
            thread::yield_now();
            assert!(!queue_g.try_pop(&mut buf));
            flag_g.set(false);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 12);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 18);
            flag_g.set(true);
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 13);
            while !queue_g.try_pop(&mut buf) {
                thread::yield_now();
            }
            assert_eq!(buf, 14);
            while !queue_g.try_pop(&mut buf) {
                thread::yield_now();
            }
            assert_eq!(buf, 15);
            queue2_g.close();
            assert!(queue_g.wait_pop(&mut buf));
            assert_eq!(buf, 19);
            flag_g.set_permanently(false);
            flag_g.set(true);
            assert!(!queue_g.wait_pop(&mut buf));
            exit_count_g.add(1);
        });

        let queue_h = Arc::clone(&queue);
        let queue2_h = Arc::clone(&queue2);
        let exit_count_h = Arc::clone(&exit_count);
        let h = thread::spawn(move || {
            let mut buf = 0;
            assert!(!queue2_h.wait_pop(&mut buf));
            assert!(queue_h.force_push(19));
            exit_count_h.add(1);
        });

        for h in [f, c0, c1, c2, g, h] {
            h.join().unwrap();
        }

        exit_count.add(-6);
        exit_count.wait_until_zero();
        flag.set(true);
        flag.wait_until_false();
        assert!(!queue.wait_push(20));
        assert!(!queue2.wait_push(21));
        assert!(!queue.is_closed());
        assert!(queue2.is_closed());
        queue.close();
        assert!(queue.is_closed());
    }

    #[test]
    fn test_queue_stress_many_writers() {
        stress_test(
            &[0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10],
            &[0.30],
            8,
        );
    }

    #[test]
    fn test_queue_stress_many_readers() {
        stress_test(
            &[0.50, 0.10],
            &[0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15],
            2,
        );
    }

    fn stress_test(writer_yields: &[f64], reader_yields: &[f64], num_writers: usize) {
        let queue = Arc::new(ThreadSafeQueue::new());
        let total = Arc::new(AtomicI64::new(0));
        let total_sq = Arc::new(AtomicI64::new(0));

        let mut writers = Vec::new();
        for &yield_prob in writer_yields.iter().take(num_writers) {
            let queue = Arc::clone(&queue);
            writers.push(thread::spawn(move || {
                let mut rand = Rand::new();
                for i in 1..=10_000 {
                    if rand.next_bool(yield_prob) {
                        thread::yield_now();
                    }
                    queue.wait_push(i);
                }
            }));
        }

        let mut readers = Vec::new();
        for &yield_prob in reader_yields {
            let queue = Arc::clone(&queue);
            let total = Arc::clone(&total);
            let total_sq = Arc::clone(&total_sq);
            readers.push(thread::spawn(move || {
                let mut rand = Rand::new();
                let mut sum = 0i64;
                let mut sum_sq = 0i64;
                let mut buf = Vec::new();
                loop {
                    if rand.next_bool(yield_prob) {
                        thread::yield_now();
                    }
                    if rand.next_bool(0.5) {
                        let mut x = 0;
                        if queue.wait_pop(&mut x) {
                            sum += x as i64;
                            sum_sq += (x as i64) * (x as i64);
                        } else {
                            break;
                        }
                    } else {
                        let n = rand.next_i32_range(1, 8);
                        if queue.wait_pop_up_to_n(&mut buf, n as usize) {
                            assert!(!buf.is_empty() && buf.len() <= n as usize);
                            for x in buf.drain(..) {
                                sum += x as i64;
                                sum_sq += (x as i64) * (x as i64);
                            }
                        } else {
                            break;
                        }
                    }
                }
                total.fetch_add(sum, Ordering::Relaxed);
                total_sq.fetch_add(sum_sq, Ordering::Relaxed);
            }));
        }

        for w in writers {
            w.join().unwrap();
        }
        queue.set_read_only();
        for r in readers {
            r.join().unwrap();
        }

        let expected_sum = num_writers as i64 * 10_000 * 10_001 / 2;
        let expected_sum_sq = num_writers as i64 * 10_000 * 10_001 * 20_001 / 6;
        assert_eq!(total.load(Ordering::Relaxed), expected_sum);
        assert_eq!(total_sq.load(Ordering::Relaxed), expected_sum_sq);
    }
}
