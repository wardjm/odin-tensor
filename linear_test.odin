package tensor

import "core:math"
import "core:mem"
import "core:testing"

// naive_linear_element computes element (i, j) of x @ weights + bias as the
// product kernels do: starting from bias[j] and adding the products in order,
// or, for products the kernel sums over fixed slices of k, starting from
// bias[j] and adding each slice's own sum in turn.
naive_linear_element :: proc(x, weights: Tensor_View($T), bias: []T, i, j: int, fused: bool) -> T {
	m, k, n := x.shape[0], x.shape[1], weights.shape[1]
	slices := (k + _DEPTH_SPLIT_SLICE - 1) / _DEPTH_SPLIT_SLICE
	slice_depth := k
	if k >= _DEPTH_SPLIT_MIN_DEPTH && slices*m*n <= _DEPTH_SPLIT_MAX_PARTIALS {
		slice_depth = _DEPTH_SPLIT_SLICE
	}
	value := bias[j]
	for first := 0; first < k; first += slice_depth {
		sum: T
		if slice_depth == k {
			sum = value
		}
		for p in first..<min(first+slice_depth, k) {
			sum = naive_multiply_add(x.data[x.offset + i*x.strides[0] + p*x.strides[1]], weights.data[weights.offset + p*weights.strides[0] + j*weights.strides[1]], sum, fused)
		}
		value = sum if slice_depth == k else value + sum
	}
	return value
}

// expect_linear_matches_naive computes linear for an m×k x and k×n weights,
// with x transposed when asked and a bias read with a stride of two, and
// compares every element exactly with naive_linear_element.
expect_linear_matches_naive :: proc(t: ^testing.T, $T: typeid, m, k, n: int, transpose_x: bool, loc := #caller_location) {
	x_storage, err := new_tensor(T, []int{k, m} if transpose_x else []int{m, k})
	require_ok(t, err, loc)
	defer release(&x_storage)
	fill_product_operand(x_storage.data, 0)
	weights: Tensor(T)
	weights, err = new_tensor(T, []int{k, n})
	require_ok(t, err, loc)
	defer release(&weights)
	fill_product_operand(weights.data, 3)
	bias_storage: Tensor(T)
	bias_storage, err = new_tensor(T, []int{2*n})
	require_ok(t, err, loc)
	defer release(&bias_storage)
	fill_product_operand(bias_storage.data, 5)
	bias_shape, bias_strides := [1]int{n}, [1]int{2}
	bias := Tensor_View(T){data = bias_storage.data, shape = bias_shape[:], strides = bias_strides[:]}
	x := as_view(x_storage)
	x_shape, x_strides: [2]int
	if transpose_x {
		x, err = transpose_view(x, x_shape[:], x_strides[:])
		require_ok(t, err, loc)
	}

	result: Tensor(T)
	result, err = linear_views(x, as_view(weights), bias)
	require_ok(t, err, loc)
	defer release(&result)
	expect_shape(t, result.shape, []int{m, n})
	starts := make([]T, n)
	defer delete(starts)
	for j in 0..<n {
		starts[j] = bias_storage.data[2*j]
	}
	mismatches := 0
	for i in 0..<m {
		for j in 0..<n {
			if result.data[i*n + j] != naive_linear_element(x, as_view(weights), starts, i, j, _product_fused()) {
				mismatches += 1
			}
		}
	}
	testing.expectf(t, mismatches == 0, "%v [%d, %d] @ [%d, %d] + bias: %d elements differ from the naive sum", typeid_of(T), m, k, k, n, mismatches, loc = loc)
}

@(test)
test_linear_starts_each_sum_from_the_bias :: proc(t: ^testing.T) {
	// Packed, with partial tiles, panels and k blocks; too few rows to pack;
	// too narrow for a column panel; and a long k summed over slices.
	expect_linear_matches_naive(t, f32, 131, 270, 150, false)
	expect_linear_matches_naive(t, f32, 37, 300, 13, true)
	expect_linear_matches_naive(t, f64, 67, 260, 70, true)
	expect_linear_matches_naive(t, f32, 3, 300, 37, false)
	expect_linear_matches_naive(t, f32, 19, 300, 2, false)
	expect_linear_matches_naive(t, f64, 3, 9000, 5, false)
	expect_linear_matches_naive(t, i32, 13, 20, 9, false)
}

@(test)
test_linear_validates_arguments :: proc(t: ^testing.T) {
	x, err := new_tensor(f32, []int{4, 3})
	require_ok(t, err)
	defer release(&x)
	weights: Tensor(f32)
	weights, err = new_tensor(f32, []int{3, 2})
	require_ok(t, err)
	defer release(&weights)
	bias: Tensor(f32)
	bias, err = new_tensor(f32, []int{2})
	require_ok(t, err)
	defer release(&bias)
	wrong: Tensor(f32)
	wrong, err = new_tensor(f32, []int{3})
	require_ok(t, err)
	defer release(&wrong)
	vector: Tensor(f32)
	vector, err = new_tensor(f32, []int{3})
	require_ok(t, err)
	defer release(&vector)

	_, err = linear(x, weights, wrong)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	_, err = linear(x, x, bias)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	_, err = linear(vector, weights, bias)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)

	tape: Tape(f32)
	tape, err = tape_init(f32)
	require_ok(t, err)
	defer tape_release(&tape)
	xv := tape_add_leaf_borrow(&tape, &x) or_else panic("leaf")
	wv := tape_add_leaf_borrow(&tape, &weights) or_else panic("leaf")
	bv := tape_add_leaf_borrow(&tape, &wrong) or_else panic("leaf")
	_, err = tape_linear(&tape, xv, wv, bv)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	_, err = tape_linear(&tape, xv, wv, Tape_Value(f32){})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)

	// Without tracked parents the result has no gradient.
	constant_x := tape_add_leaf_borrow(&tape, &x, false) or_else panic("leaf")
	constant_w := tape_add_leaf_borrow(&tape, &weights, false) or_else panic("leaf")
	constant_b := tape_add_leaf_borrow(&tape, &bias, false) or_else panic("leaf")
	affine := tape_linear(&tape, constant_x, constant_w, constant_b) or_else panic("linear")
	_, grad_err := tape_gradient(&tape, affine)
	testing.expect_value(t, grad_err, Tensor_Error.Gradient_Not_Tracked)
}

// linear_helper is a @ b plus a constant bias.
linear_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
	weights := tape_value(tape, b) or_return
	bias := constant_feature_leaf(tape, weights.shape[1], -0.5) or_return
	return tape_linear(tape, a, b, bias)
}

// linear_bias_helper is a @ constant weights plus bias b, with a transposed
// so the product reads it strided.
linear_bias_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
	transposed := tape_transpose(tape, a) or_return
	x := tape_value(tape, transposed) or_return
	bias := tape_value(tape, b) or_return
	values := new_tensor(f64, []int{x.shape[1], bias.shape[0]}) or_return
	defer release(&values)
	for &value, i in values.data {
		value = math.sin(0.3*f64(i))
	}
	weights := tape_add_leaf_copy(tape, as_view(values), false) or_return
	return tape_linear(tape, transposed, weights, b)
}

@(test)
test_tape_linear_matches_finite_differences :: proc(t: ^testing.T) {
	values := make([]f64, 7*6)
	defer delete(values)
	for &value, i in values {
		value = math.cos(0.4*f64(i)) * 2
	}
	check_tape_helper(t, .Linear, values[:35], []int{7, 5}, values[:15], []int{5, 3}, linear_helper)
	// The transposed [5, 7] a is a [7, 5] x; b is a bias of 4.
	check_tape_helper(t, .Linear, values[:35], []int{5, 7}, []f64{0.5, -1, 2, 0.25}, []int{4}, linear_bias_helper)
}

linear_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 4, allocator) or_return
	defer tape_release(&tape)
	x := new_tensor(f64, []int{8, 3}, allocator) or_return
	defer release(&x)
	for &value, i in x.data {
		value = f64(i % 5)
	}
	weights := new_tensor(f64, []int{3, 6}, allocator) or_return
	defer release(&weights)
	fill(&weights, 0.5)
	bias := new_tensor(f64, []int{6}, allocator) or_return
	defer release(&bias)
	fill(&bias, 2)
	xv := tape_add_leaf_borrow(&tape, &x) or_return
	wv := tape_add_leaf_borrow(&tape, &weights) or_return
	bv := tape_add_leaf_borrow(&tape, &bias) or_return
	affine := tape_linear(&tape, xv, wv, bv) or_return
	total := tape_sum_all(&tape, affine) or_return
	return tape_backward(&tape, total)
}

@(test)
test_tape_linear_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, linear_allocation_scenario)
}
