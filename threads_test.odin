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
	{
		// The weight gradient here is xᵀ @ upstream, [4, 20000] @ [20000, 3]:
		// a depth-split product that adds into an existing gradient.
		tape := tape_init(f64) or_else panic("tape")
		defer tape_release(&tape)
		inputs := filled([]int{20000, 4}, 9)
		defer release(&inputs)
		weights := filled([]int{4, 3}, 10)
		defer release(&weights)
		x := tape_add_leaf_borrow(&tape, &inputs, false) or_else panic("leaf")
		w := tape_add_leaf_borrow(&tape, &weights) or_else panic("leaf")
		product := tape_matmul(&tape, x, w) or_else panic("matmul")
		total := tape_sum_all(&tape, tape_tanh(&tape, product) or_else panic("tanh")) or_else panic("sum")
		for _ in 0..<2 {
			if tape_backward(&tape, total) != .None do panic("backward")
		}
		gradient := tape_gradient(&tape, w) or_else panic("gradient")
		append(&results, slice.clone(gradient.data))
	}
	return
}

// filled is a tensor of shape with deterministic values in about [-2, 2].
filled :: proc(shape: []int, salt: int) -> Tensor(f64) {
	t := new_tensor(f64, shape) or_else panic("new")
	for &value, i in t.data do value = f64((i*7 + salt) % 29) * 0.14 - 2
	return t
}

// all_elementwise computes element-wise, im2col and pooling results, forward
// and backward, each large enough to split between threads.
all_elementwise :: proc() -> (results: [dynamic][]f64) {
	big := filled([]int{300, 400}, 1)
	defer release(&big)
	other := filled([]int{300, 400}, 2)
	defer release(&other)
	row := filled([]int{400}, 3)
	defer release(&row)
	for &value in other.data do value = value + 3 // keep divisors away from zero
	shape_buffer, strides_buffer: [2]int
	transposed := transpose_view(as_view(big), shape_buffer[:], strides_buffer[:]) or_else panic("transpose")

	keep :: proc(results: ^[dynamic][]f64, t: Tensor(f64)) {
		t := t
		append(results, slice.clone(t.data))
		release(&t)
	}
	keep(&results, add(big, other) or_else panic("add"))
	keep(&results, div(big, other) or_else panic("div"))
	keep(&results, mul(big, row) or_else panic("mul")) // broadcast rows
	keep(&results, tanh_views(transposed) or_else panic("tanh")) // strided source
	keep(&results, exp(big) or_else panic("exp"))
	in_place := copy_view(as_view(big)) or_else panic("copy")
	relu_in_place(&in_place)
	add_in_place(&in_place, row)
	keep(&results, in_place)

	image := filled([]int{16, 30, 30, 3}, 4)
	defer release(&image)
	axes_shape, axes_strides: [4]int
	// The same values read channels-first, so pixels and channels are strided.
	channels_first := filled([]int{16, 3, 30, 30}, 4)
	defer release(&channels_first)
	permuted := permute_view(as_view(channels_first), []int{0, 2, 3, 1}, axes_shape[:], axes_strides[:]) or_else panic("permute")
	keep(&results, im2col(image, 5, 5) or_else panic("im2col"))
	keep(&results, im2col_views(permuted, 3, 4, 2) or_else panic("im2col"))
	keep(&results, max_pool2d(image, 2, 2) or_else panic("pool"))
	keep(&results, max_pool2d_views(permuted, 3, 2) or_else panic("pool"))

	// A convolution layer's graph and a same-shape and a broadcast product.
	tape := tape_init(f64) or_else panic("tape")
	defer tape_release(&tape)
	weights := filled([]int{5, 5, 3, 8}, 5)
	defer release(&weights)
	for &value in weights.data do value *= 0.05
	bias := filled([]int{8}, 6)
	defer release(&bias)
	x := tape_add_leaf_borrow(&tape, &image) or_else panic("leaf")
	w := tape_add_leaf_borrow(&tape, &weights) or_else panic("leaf")
	b := tape_add_leaf_borrow(&tape, &bias) or_else panic("leaf")
	windows := tape_im2col(&tape, x, 5, 5) or_else panic("im2col")
	rows := tape_reshape(&tape, windows, []int{-1, 75}) or_else panic("reshape")
	kernels := tape_reshape(&tape, w, []int{75, 8}) or_else panic("reshape")
	product := tape_matmul(&tape, rows, kernels) or_else panic("matmul")
	biased := tape_add(&tape, product, b) or_else panic("add")
	images := tape_reshape(&tape, biased, []int{16, 26, 26, 8}) or_else panic("reshape")
	activated := tape_relu(&tape, images) or_else panic("relu")
	squared := tape_mul(&tape, activated, activated) or_else panic("mul")
	summed := tape_add(&tape, squared, activated) or_else panic("add")
	pooled := tape_max_pool2d(&tape, summed, 2, 2) or_else panic("pool")
	sigmoid := tape_sigmoid(&tape, pooled) or_else panic("sigmoid")
	total := tape_sum_all(&tape, sigmoid) or_else panic("sum")
	if tape_backward(&tape, total) != .None do panic("backward")
	// A second pass adds into the existing gradients.
	if tape_backward(&tape, total) != .None do panic("backward")
	for value in ([]Tape_Value(f64){x, w, b, activated}) {
		gradient := tape_gradient(&tape, value) or_else panic("gradient")
		append(&results, slice.clone(gradient.data))
	}

	// A repeating operand long enough that its gradient's sums are split.
	wide := filled([]int{600, 256}, 7)
	defer release(&wide)
	scale := filled([]int{256}, 8)
	defer release(&scale)
	tape_reset(&tape)
	wide_value := tape_add_leaf_borrow(&tape, &wide) or_else panic("leaf")
	scale_value := tape_add_leaf_borrow(&tape, &scale) or_else panic("leaf")
	scaled := tape_mul(&tape, wide_value, scale_value) or_else panic("mul")
	divided := tape_div(&tape, scaled, scale_value) or_else panic("div")
	wide_total := tape_sum_all(&tape, tape_tanh(&tape, divided) or_else panic("tanh")) or_else panic("sum")
	if tape_backward(&tape, wide_total) != .None do panic("backward")
	for value in ([]Tape_Value(f64){wide_value, scale_value}) {
		gradient := tape_gradient(&tape, value) or_else panic("gradient")
		append(&results, slice.clone(gradient.data))
	}

	// Batch normalization, alone and with ReLU, whose sums split rows.
	normalized_input := filled([]int{2000, 40}, 11)
	defer release(&normalized_input)
	features := filled([]int{40}, 12)
	defer release(&features)
	for relu in ([]bool{false, true}) {
		tape_reset(&tape)
		input_value := tape_add_leaf_borrow(&tape, &normalized_input) or_else panic("leaf")
		// The same values scale and shift.
		features_value := tape_add_leaf_borrow(&tape, &features) or_else panic("leaf")
		normalized := tape_batch_norm(&tape, input_value, features_value, features_value, 1e-5, relu = relu) or_else panic("batch_norm")
		normalized_total := tape_sum_all(&tape, tape_tanh(&tape, normalized) or_else panic("tanh")) or_else panic("sum")
		if tape_backward(&tape, normalized_total) != .None do panic("backward")
		value := tape_value(&tape, normalized) or_else panic("value")
		append(&results, slice.clone(value.data[value.offset:][:2000*40]))
		for leaf in ([]Tape_Value(f64){input_value, features_value}) {
			gradient := tape_gradient(&tape, leaf) or_else panic("gradient")
			append(&results, slice.clone(gradient.data))
		}
	}

	// A dense layer's affine map, whose bias gradient sums 600 rows.
	dense_weights := filled([]int{256, 40}, 9)
	defer release(&dense_weights)
	dense_bias := filled([]int{40}, 10)
	defer release(&dense_bias)
	keep(&results, linear(wide, dense_weights, dense_bias) or_else panic("linear"))
	tape_reset(&tape)
	wide_value = tape_add_leaf_borrow(&tape, &wide) or_else panic("leaf")
	weights_value := tape_add_leaf_borrow(&tape, &dense_weights) or_else panic("leaf")
	bias_value := tape_add_leaf_borrow(&tape, &dense_bias) or_else panic("leaf")
	affine := tape_linear(&tape, wide_value, weights_value, bias_value) or_else panic("linear")
	affine_total := tape_sum_all(&tape, tape_tanh(&tape, affine) or_else panic("tanh")) or_else panic("sum")
	if tape_backward(&tape, affine_total) != .None do panic("backward")
	for value in ([]Tape_Value(f64){wide_value, weights_value, bias_value}) {
		gradient := tape_gradient(&tape, value) or_else panic("gradient")
		append(&results, slice.clone(gradient.data))
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
	elementwise_reference := all_elementwise()
	defer delete_results(elementwise_reference, nil)
	// Odd counts leave uneven row ranges.
	for threads in ([]int{2, 3, 4, 8}) {
		require_ok(t, set_thread_count(threads))
		testing.expect_value(t, thread_count(), threads)
		actual, integer_actual := all_products()
		defer delete_results(actual, integer_actual)
		expect_same_results(t, threads, reference, actual, integer_reference, integer_actual)
		elementwise_actual := all_elementwise()
		defer delete_results(elementwise_actual, nil)
		for r, i in elementwise_reference {
			testing.expectf(t, slice.equal(r, elementwise_actual[i]), "%d threads: element-wise result %d differs from one thread", threads, i)
		}
		// Few runs of columns leave threads idle, so column sums split the
		// rows along the pairwise tree; each must still be one pairwise sum.
		expect_split_column_sums_match_pairwise(t, f32, 65536, 32)
		expect_split_column_sums_match_pairwise(t, f32, 20001, 7)
		expect_split_column_sums_match_pairwise(t, f64, 70000, 1)
		expect_split_column_sums_match_pairwise(t, f32, 4099, 40)
		expect_column_sums_match_pairwise(t, f32, 5, 3000, 40)
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

// expect_sliced_product checks a long product against the definition of the
// depth split: the in-order sum of the products of 2048-long slices of k,
// each computed whole. transpose_a stores a as [k, m], as a weight gradient
// reads its input.
expect_sliced_product :: proc(t: ^testing.T, $T: typeid, m, k, n: int, transpose_a: bool, loc := #caller_location) {
	a_storage := new_tensor(T, []int{k, m} if transpose_a else []int{m, k}) or_else panic("a")
	defer release(&a_storage)
	b_storage := new_tensor(T, []int{k, n}) or_else panic("b")
	defer release(&b_storage)
	fill_product_operand(a_storage.data, 0)
	fill_product_operand(b_storage.data, 3)
	a, b := as_view(a_storage), as_view(b_storage)
	a_shape, a_strides: [2]int
	if transpose_a {
		a = transpose_view(a, a_shape[:], a_strides[:]) or_else panic("transpose")
	}
	product := matmul_views(a, b) or_else panic("matmul")
	defer release(&product)

	expected := new_tensor(T, []int{m, n}) or_else panic("expected")
	defer release(&expected)
	for first := 0; first < k; first += _DEPTH_SPLIT_SLICE {
		depth := min(_DEPTH_SPLIT_SLICE, k-first)
		a_slice_shape, a_slice_strides, b_slice_shape, b_slice_strides: [2]int
		a_slice := slice_view(a, 1, first, depth, 1, a_slice_shape[:], a_slice_strides[:]) or_else panic("slice")
		b_slice := slice_view(b, 0, first, depth, 1, b_slice_shape[:], b_slice_strides[:]) or_else panic("slice")
		partial := matmul_views(a_slice, b_slice) or_else panic("partial")
		defer release(&partial)
		expect_product_matches_naive(t, partial.data, a_slice, b_slice, loc = loc)
		for &value, i in expected.data do value += partial.data[i]
	}
	testing.expectf(t, slice.equal(product.data, expected.data), "[%d, %d] @ [%d, %d] is not the in-order sum of its slices' products", m, k, k, n, loc = loc)
}

@(test)
test_long_products_sum_fixed_slices :: proc(t: ^testing.T) {
	// From 2^13 inner elements, with a small output, the product sums slices.
	expect_sliced_product(t, f64, 3, 8192, 5, false)
	expect_sliced_product(t, f64, 25, 20000, 8, true)
	expect_sliced_product(t, f64, 7, 9000, 16, true)
	// The partial products' room is in bytes, so f32 holds twice as many: a
	// first convolution's weight gradient, 32 slices of 27×32, fits.
	expect_sliced_product(t, f32, 27, 65536, 32, true)
	// Just below the threshold, or with too many partial products to hold,
	// it is one sequential sum, exactly the naive product.
	expect_matmul_matches_naive(t, f64, 3, 8191, 5, false, false)
	expect_matmul_matches_naive(t, f32, 300, 9000, 32, true, false)
	expect_matmul_matches_naive(t, f64, 27, 65536, 32, true, false)
}
