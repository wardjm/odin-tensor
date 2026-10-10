package tensor

// Worker threads for matrix products.
//
// The library is single-threaded until set_thread_count asks for more. Large
// products are then split into row ranges, one per thread, with the calling
// thread taking one range and waiting for the rest. Each output element is
// still computed by one thread, adding its products in the same order, so the
// thread count never changes a result.

import "base:intrinsics"
import "core:sync"
import "core:thread"

// _PARALLEL_MIN_WORK is the multiply-add count below which a product stays on
// the calling thread: splitting smaller ones costs more than it saves.
_PARALLEL_MIN_WORK :: 1 << 18

_Parallel_Body :: #type proc(data: rawptr, chunk: int)

_Parallel_Job :: struct {
	body:   _Parallel_Body,
	data:   rawptr,
	chunks: int,
	// next is the next unclaimed chunk, taken with an atomic add.
	next:   int,
	done:   sync.Wait_Group,
}

_Pool :: struct {
	// busy is held by the one caller using the workers, and by
	// set_thread_count while it changes them.
	busy:       sync.Mutex,
	threads:    [dynamic]^thread.Thread,
	count:      int,
	// mutex and wake guard job, generation and stop, which workers wait on.
	mutex:      sync.Mutex,
	wake:       sync.Cond,
	generation: int,
	job:        ^_Parallel_Job,
	stop:       bool,
	// active counts workers inside the current job, so the caller doesn't
	// return while one could still touch it.
	active:     int,
}

@(private)
_pool := _Pool{count = 1}

// set_thread_count sets how many threads matrix products may use, counting
// the calling thread; n = 1, the default, keeps everything on the caller and
// joins any workers. It starts or stops worker threads, created with
// context.allocator, and returns Invalid_Shape for n < 1. Physical cores are
// a good choice: hyperthreads rarely help these kernels. It is safe to call
// while other threads run tensor operations; a product already in progress
// finishes first.
set_thread_count :: proc(n: int) -> Tensor_Error {
	if n < 1 {
		return .Invalid_Shape
	}
	sync.guard(&_pool.busy)
	_stop_workers()
	for len(_pool.threads) < n-1 {
		worker := thread.create(_parallel_worker)
		if worker == nil {
			_stop_workers()
			return .Allocation_Failed
		}
		if _, err := append(&_pool.threads, worker); err != nil {
			thread.destroy(worker)
			_stop_workers()
			return .Allocation_Failed
		}
		thread.start(worker)
	}
	intrinsics.atomic_store(&_pool.count, n)
	return .None
}

// thread_count is the number of threads matrix products may use.
thread_count :: proc() -> int {
	return intrinsics.atomic_load(&_pool.count)
}

// _stop_workers joins and destroys every worker. The caller holds busy.
_stop_workers :: proc() {
	sync.mutex_lock(&_pool.mutex)
	_pool.stop = true
	sync.cond_broadcast(&_pool.wake)
	sync.mutex_unlock(&_pool.mutex)
	for worker in _pool.threads {
		thread.join(worker)
		thread.destroy(worker)
	}
	delete(_pool.threads)
	_pool.threads = nil
	_pool.stop = false
	intrinsics.atomic_store(&_pool.count, 1)
}

_parallel_worker :: proc(t: ^thread.Thread) {
	sync.mutex_lock(&_pool.mutex)
	seen := _pool.generation
	for {
		for _pool.generation == seen && !_pool.stop {
			sync.cond_wait(&_pool.wake, &_pool.mutex)
		}
		if _pool.stop {
			break
		}
		seen = _pool.generation
		job := _pool.job
		if job == nil {
			// The job finished before this worker woke.
			continue
		}
		intrinsics.atomic_add(&_pool.active, 1)
		sync.mutex_unlock(&_pool.mutex)
		_parallel_run(job)
		intrinsics.atomic_sub(&_pool.active, 1)
		sync.mutex_lock(&_pool.mutex)
	}
	sync.mutex_unlock(&_pool.mutex)
}

_parallel_run :: proc(job: ^_Parallel_Job) {
	for {
		chunk := intrinsics.atomic_add(&job.next, 1)
		if chunk >= job.chunks {
			return
		}
		job.body(job.data, chunk)
		sync.wait_group_done(&job.done)
	}
}

// _parallel_for runs body(data, chunk) for every chunk in 0..<chunks and
// returns when all have finished. It uses the workers when there are any and
// no other caller has them; otherwise it runs the chunks itself, in order.
_parallel_for :: proc(chunks: int, body: _Parallel_Body, data: rawptr) {
	if chunks > 1 && thread_count() > 1 && sync.mutex_try_lock(&_pool.busy) {
		defer sync.mutex_unlock(&_pool.busy)
		if len(_pool.threads) > 0 {
			job := _Parallel_Job{body = body, data = data, chunks = chunks}
			sync.wait_group_add(&job.done, chunks)
			sync.mutex_lock(&_pool.mutex)
			_pool.job = &job
			_pool.generation += 1
			sync.cond_broadcast(&_pool.wake)
			sync.mutex_unlock(&_pool.mutex)

			_parallel_run(&job)
			sync.wait_group_wait(&job.done)
			sync.mutex_lock(&_pool.mutex)
			_pool.job = nil
			sync.mutex_unlock(&_pool.mutex)
			for intrinsics.atomic_load(&_pool.active) > 0 {
				intrinsics.cpu_relax()
			}
			return
		}
	}
	for chunk in 0..<chunks {
		body(data, chunk)
	}
}

// _parallel_chunks splits count items into one range per thread, each a
// multiple of granule, and returns the range length and number of ranges.
_parallel_chunks :: proc(count, granule: int) -> (length, chunks: int) {
	threads := thread_count()
	length = max(granule, (count + threads - 1) / threads)
	length += (granule - length%granule) % granule
	return length, (count + length - 1) / length
}

_Product_Rows :: struct($T: typeid) {
	output:     []T,
	a, b:       Tensor_View(T),
	accumulate: bool,
	rows:       int,
	m, n:       int,
}

// _product_kernel computes output = a @ b, or output += a @ b when accumulate
// is set, splitting a product with enough work into row ranges across
// threads. Each range is a product of a's rows with all of b, computed by
// _product_kernel_serial into its own rows of output.
_product_kernel :: proc(output: []$T, a, b: Tensor_View(T), accumulate := false) {
	if len(a.shape) == 2 && thread_count() > 1 {
		m, k := a.shape[0], a.shape[1]
		n := b.shape[1] if len(b.shape) == 2 else 1
		if m >= 2*_PRODUCT_TILE_ROWS && m*k*n >= _PARALLEL_MIN_WORK {
			rows, chunks := _parallel_chunks(m, _PRODUCT_TILE_ROWS)
			job := _Product_Rows(T){output = output, a = a, b = b, accumulate = accumulate, rows = rows, m = m, n = n}
			_parallel_for(chunks, proc(data: rawptr, chunk: int) {
				job := (^_Product_Rows(T))(data)
				first := chunk * job.rows
				rows := min(job.rows, job.m - first)
				shape := [2]int{rows, job.a.shape[1]}
				a := job.a
				a.shape = shape[:]
				a.offset += first * job.a.strides[0]
				_product_kernel_serial(job.output[first*job.n:][:rows*job.n], a, job.b, job.accumulate)
			}, &job)
			return
		}
	}
	_product_kernel_serial(output, a, b, accumulate)
}
