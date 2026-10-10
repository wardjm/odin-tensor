package tensor

import "core:slice"
import "core:testing"
import "core:thread"

Product_Case :: struct {
	m, k, n:                  int,
	transpose_a, transpose_b: bool,
}

// PRODUCT_CASES are each large enough to split, and between them cover full,
// half and quarter column tiles, the unblocked strip, transposed operands, a
// vector right operand and uneven row splits.
PRODUCT_CASES :: []Product_Case{
	{301, 200, 64, false, false},
	{257, 300, 8, false, false},
	{150, 400, 13, true, false},
	{97, 800, 4, false, true},
	{120, 900, 3, true, true},
	{2001, 300, 1, false, false},
	{40, 9000, 16, true, false},
}

// threaded_product computes one product with the current thread count, the
// operands filled deterministically and read through transposed views when
// asked.
threaded_product :: proc($T: typeid, c: Product_Case) -> []T {
	a_shape := c.transpose_a ? []int{c.k, c.m} : []int{c.m, c.k}
	b_shape := c.transpose_b ? []int{c.n, c.k} : []int{c.k, c.n}
	a := new_tensor(T, a_shape) or_else panic("a")
	defer release(&a)
	b := new_tensor(T, b_shape) or_else panic("b")
	defer release(&b)
	for &value, i in a.data do value = T(i % 13) - 6
	for &value, i in b.data do value = T(i % 7) - 3
	a_view, b_view := as_view(a), as_view(b)
	a_shape_buffer, a_strides, b_shape_buffer, b_strides: [2]int
	if c.transpose_a do a_view = transpose_view(a_view, a_shape_buffer[:], a_strides[:]) or_else panic("transpose")
	if c.transpose_b do b_view = transpose_view(b_view, b_shape_buffer[:], b_strides[:]) or_else panic("transpose")
	if c.n == 1 {
		// A vector right operand.
		b_view.shape, b_view.strides = b_view.shape[:1], b_view.strides[:1]
		b_view.strides[0] = b_view.strides[0] if !c.transpose_b else 1
	}
	result := matmul_views(a_view, b_view) or_else panic("matmul")
	defer release(&result)
	return slice.clone(result.data)
}

// all_products computes every case for float types and an integer type, plus
// a broadcast batch, many small batches and a matmul backward pass.
all_products :: proc() -> (results: [dynamic][]f64, integer_results: [dynamic][]i32) {
	for c in PRODUCT_CASES {
		f32_result := threaded_product(f32, c)
		defer delete(f32_result)
		f32_widened := make([]f64, len(f32_result))
		for value, i in f32_result do f32_widened[i] = f64(value)
		append(&results, f32_widened, threaded_product(f64, c))
		append(&integer_results, threaded_product(i32, c))
	}
	for batch_shapes in ([][2][]int{{{6, 40, 300}, {300, 50}}, {{600, 8, 9}, {9, 70}}, {{2, 1, 80, 90}, {5, 90, 60}}}) {
		a := new_tensor(f64, batch_shapes[0]) or_else panic("a")
		defer release(&a)
		b := new_tensor(f64, batch_shapes[1]) or_else panic("b")
		defer release(&b)
		for &value, i in a.data do value = f64(i % 17) * 0.25 - 2
		for &value, i in b.data do value = f64(i % 5) * 0.5 - 1
		product := matmul(a, b) or_else panic("matmul")
		append(&results, slice.clone(product.data))
		release(&product)
	}
	{
		tape := tape_init(f64) or_else panic("tape")
		defer tape_release(&tape)
		weights := new_tensor(f64, []int{300, 70}) or_else panic("w")
		defer release(&weights)
		inputs := new_tensor(f64, []int{256, 300}) or_else panic("x")
		defer release(&inputs)
		for &value, i in weights.data do value = f64(i % 23) * 0.01 - 0.1
		for &value, i in inputs.data do value = f64(i % 19) * 0.05 - 0.4
		w := tape_add_leaf_borrow(&tape, &weights) or_else panic("leaf")
		x := tape_add_leaf_borrow(&tape, &inputs) or_else panic("leaf")
		product := tape_matmul(&tape, x, w) or_else panic("matmul")
		activated := tape_tanh(&tape, product) or_else panic("tanh")
		total := tape_sum_all(&tape, activated) or_else panic("sum")
		if tape_backward(&tape, total) != .None do panic("backward")
		for value in ([]Tape_Value(f64){w, x}) {
			gradient := tape_gradient(&tape, value) or_else panic("gradient")
			append(&results, slice.clone(gradient.data))
		}
	}
	return
}

delete_results :: proc(results: [dynamic][]f64, integer_results: [dynamic][]i32) {
	for r in results do delete(r)
	for r in integer_results do delete(r)
	delete(results)
	delete(integer_results)
}

expect_same_results :: proc(t: ^testing.T, threads: int, reference, actual: [dynamic][]f64, integer_reference, integer_actual: [dynamic][]i32, loc := #caller_location) {
	for r, i in reference {
		testing.expectf(t, slice.equal(r, actual[i]), "%d threads: float result %d differs from one thread", threads, i, loc = loc)
	}
	for r, i in integer_reference {
		testing.expectf(t, slice.equal(r, integer_actual[i]), "%d threads: integer result %d differs from one thread", threads, i, loc = loc)
	}
}

Concurrent_Caller :: struct {
	reference: []f64,
	matches:   bool,
}

// Every thread-count change lives in this one test: the setting is global and
// tests run concurrently.
@(test)
test_threaded_products_match_one_thread :: proc(t: ^testing.T) {
	defer set_thread_count(1)
	testing.expect_value(t, thread_count(), 1)
	testing.expect_value(t, set_thread_count(0), Tensor_Error.Invalid_Shape)
	testing.expect_value(t, thread_count(), 1)

	reference, integer_reference := all_products()
	defer delete_results(reference, integer_reference)
	// Odd counts leave uneven row ranges.
	for threads in ([]int{2, 3, 4, 8}) {
		require_ok(t, set_thread_count(threads))
		testing.expect_value(t, thread_count(), threads)
		actual, integer_actual := all_products()
		defer delete_results(actual, integer_actual)
		expect_same_results(t, threads, reference, actual, integer_reference, integer_actual)
	}

	// Several threads multiplying at once: one gets the workers, the rest run
	// on their own threads, and every result is unchanged.
	require_ok(t, set_thread_count(4))
	c := PRODUCT_CASES[0]
	expected := threaded_product(f64, c)
	defer delete(expected)
	callers: [4]Concurrent_Caller
	workers: [4]^thread.Thread
	for &caller, i in callers {
		caller.reference = expected
		workers[i] = thread.create_and_start_with_data(&caller, proc(data: rawptr) {
			caller := (^Concurrent_Caller)(data)
			caller.matches = true
			for _ in 0..<20 {
				result := threaded_product(f64, PRODUCT_CASES[0])
				caller.matches &&= slice.equal(result, caller.reference)
				delete(result)
			}
		}, context)
	}
	// Reconfiguring while they run is safe too.
	for threads in ([]int{1, 6, 2, 4}) {
		require_ok(t, set_thread_count(threads))
	}
	for worker, i in workers {
		thread.join(worker)
		thread.destroy(worker)
		testing.expectf(t, callers[i].matches, "concurrent caller %d got a different result", i)
	}

	// Starting and stopping workers repeatedly neither hangs nor leaks.
	for round in 0..<30 {
		require_ok(t, set_thread_count(1 + round%5))
		result := threaded_product(f64, c)
		testing.expect(t, slice.equal(result, expected))
		delete(result)
	}
	require_ok(t, set_thread_count(1))
	testing.expect_value(t, thread_count(), 1)
}
