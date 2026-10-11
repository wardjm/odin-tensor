package tensor

import "core:math"
import "core:mem"
import "core:testing"

batch_norm_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (Tape_Value(f64), Tensor_Error) {
	return tape_batch_norm(tape, a, b, 1e-3)
}

// transposed_batch_norm_helper normalizes a transposed view, so x is not
// contiguous in either pass.
transposed_batch_norm_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
	transposed := tape_transpose(tape, a) or_return
	return tape_batch_norm(tape, transposed, b, 1e-3)
}

@(test)
test_tape_batch_norm_matches_direct_formula :: proc(t: ^testing.T) {
	ROWS, FEATURES :: 6, 4
	x, err := new_tensor(f64, []int{2, 3, FEATURES})
	require_ok(t, err)
	defer release(&x)
	for &value, i in x.data {
		value = math.sin(0.9*f64(i)) * f64(1 + i%FEATURES) + f64(i%FEATURES)
	}
	scale: Tensor(f64)
	scale, err = from_slice(f64, []f64{2, -1, 0.5, 1.25}, []int{FEATURES})
	require_ok(t, err)
	defer release(&scale)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	xv := tape_add_leaf_borrow(&tape, &x) or_else panic("leaf")
	sv := tape_add_leaf_borrow(&tape, &scale) or_else panic("leaf")
	mean, variance: [FEATURES]f64
	normalized := tape_batch_norm(&tape, xv, sv, 1e-5, mean[:], variance[:]) or_else panic("batch_norm")
	value := tape_value(&tape, normalized) or_else panic("value")
	expect_shape(t, value.shape, []int{2, 3, FEATURES})

	for f in 0..<FEATURES {
		want_mean, want_variance: f64
		for row in 0..<ROWS {
			want_mean += x.data[row*FEATURES + f] / ROWS
		}
		for row in 0..<ROWS {
			deviation := x.data[row*FEATURES + f] - want_mean
			want_variance += deviation * deviation / ROWS
		}
		testing.expect(t, abs(mean[f]-want_mean) < 1e-12 && abs(variance[f]-want_variance) < 1e-12, "statistics")
		for row in 0..<ROWS {
			want := (x.data[row*FEATURES + f] - want_mean) / math.sqrt(want_variance+1e-5) * scale.data[f]
			testing.expectf(t, abs(value.data[value.offset + row*FEATURES + f]-want) < 1e-12, "element %d, %d: got %v, want %v", row, f, value.data[value.offset + row*FEATURES + f], want)
		}
	}
}

// The fused operation computes exactly what the composition of tape
// operations it replaces computes in the forward pass.
@(test)
test_tape_batch_norm_forward_matches_composition_exactly :: proc(t: ^testing.T) {
	ROWS, FEATURES :: 300, 20
	x, err := new_tensor(f32, []int{ROWS, FEATURES})
	require_ok(t, err)
	defer release(&x)
	for &value, i in x.data {
		value = f32(math.sin(0.37*f64(i)) * 3 + f64(i%7))
	}
	scale: Tensor(f32)
	scale, err = new_tensor(f32, []int{FEATURES})
	require_ok(t, err)
	defer release(&scale)
	for &value, i in scale.data {
		value = 0.5 + f32(i)/7
	}
	epsilon_value: Tensor(f32)
	epsilon_value, err = from_slice(f32, []f32{1e-5}, []int{1})
	require_ok(t, err)
	defer release(&epsilon_value)
	tape: Tape(f32)
	tape, err = tape_init(f32)
	require_ok(t, err)
	defer tape_release(&tape)
	xv := tape_add_leaf_borrow(&tape, &x) or_else panic("leaf")
	sv := tape_add_leaf_borrow(&tape, &scale) or_else panic("leaf")
	ev := tape_add_leaf_borrow(&tape, &epsilon_value, false) or_else panic("leaf")
	fused := tape_batch_norm(&tape, xv, sv, 1e-5) or_else panic("batch_norm")

	mean := tape_mean_axis(&tape, xv, 0, keep_dims = true) or_else panic("mean")
	centered := tape_sub(&tape, xv, mean) or_else panic("sub")
	squared := tape_mul(&tape, centered, centered) or_else panic("mul")
	variance := tape_mean_axis(&tape, squared, 0, keep_dims = true) or_else panic("mean")
	shifted := tape_add(&tape, variance, ev) or_else panic("add")
	deviation := tape_sqrt(&tape, shifted) or_else panic("sqrt")
	normalized := tape_div(&tape, centered, deviation) or_else panic("div")
	composed := tape_mul(&tape, normalized, sv) or_else panic("mul")

	fused_value := tape_value(&tape, fused) or_else panic("value")
	composed_value := tape_value(&tape, composed) or_else panic("value")
	for i in 0..<ROWS*FEATURES {
		testing.expect_value(t, fused_value.data[fused_value.offset+i], composed_value.data[composed_value.offset+i])
	}
}

@(test)
test_tape_batch_norm_matches_finite_differences :: proc(t: ^testing.T) {
	values := make([]f64, 5*4*3)
	defer delete(values)
	for &value, i in values {
		value = math.sin(0.7*f64(i)) * 2 + f64(i%3)
	}
	check_tape_helper(t, .Batch_Norm, values, []int{5, 4, 3}, []f64{1.5, -0.7, 0.4}, []int{3}, batch_norm_helper)
	check_tape_helper(t, .Batch_Norm, values[:12], []int{12, 1}, []f64{0.8}, []int{1}, batch_norm_helper)
	// [3, 4] transposed is [4, 3]: three features over four rows.
	check_tape_helper(t, .Batch_Norm, values[:12], []int{3, 4}, []f64{1.5, -0.7, 0.4}, []int{3}, transposed_batch_norm_helper)
}

@(test)
test_tape_batch_norm_is_the_same_on_every_thread_count :: proc(t: ^testing.T) {
	ROWS, FEATURES :: 2000, 40
	x, err := new_tensor(f32, []int{ROWS, FEATURES})
	require_ok(t, err)
	defer release(&x)
	for &value, i in x.data {
		value = f32(math.sin(0.11*f64(i)))
	}
	scale: Tensor(f32)
	scale, err = new_tensor(f32, []int{FEATURES})
	require_ok(t, err)
	defer release(&scale)
	fill(&scale, 1.5)
	seed: Tensor(f32)
	seed, err = new_tensor(f32, []int{ROWS, FEATURES})
	require_ok(t, err)
	defer release(&seed)
	for &value, i in seed.data {
		value = f32(math.cos(0.3*f64(i)))
	}

	run :: proc(t: ^testing.T, x, scale, seed: ^Tensor(f32), threads: int) -> (output, x_grad, scale_grad: []f32) {
		require_ok(t, set_thread_count(threads))
		tape, err := tape_init(f32)
		require_ok(t, err)
		defer tape_release(&tape)
		xv := tape_add_leaf_borrow(&tape, x) or_else panic("leaf")
		sv := tape_add_leaf_borrow(&tape, scale) or_else panic("leaf")
		normalized := tape_batch_norm(&tape, xv, sv, 1e-5) or_else panic("batch_norm")
		require_ok(t, tape_backward_with_seed(&tape, normalized, seed^))
		value := tape_value(&tape, normalized) or_else panic("value")
		x_gradient := tape_gradient(&tape, xv) or_else panic("gradient")
		scale_gradient := tape_gradient(&tape, sv) or_else panic("gradient")
		return slice_clone(value.data), slice_clone(x_gradient.data), slice_clone(scale_gradient.data)
	}
	slice_clone :: proc(values: []f32) -> []f32 {
		result := make([]f32, len(values))
		copy(result, values)
		return result
	}
	defer set_thread_count(1)
	serial := [3][]f32{}
	serial[0], serial[1], serial[2] = run(t, &x, &scale, &seed, 1)
	threaded := [3][]f32{}
	threaded[0], threaded[1], threaded[2] = run(t, &x, &scale, &seed, 4)
	for which in 0..<3 {
		defer delete(serial[which])
		defer delete(threaded[which])
		mismatches := 0
		for value, i in serial[which] {
			if threaded[which][i] != value {
				mismatches += 1
			}
		}
		testing.expectf(t, mismatches == 0, "result %d: %d elements differ between 1 and 4 threads", which, mismatches)
	}
}

@(test)
test_tape_batch_norm_validates_arguments :: proc(t: ^testing.T) {
	tape, err := tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	x: Tensor(f64)
	x, err = new_tensor(f64, []int{4, 3})
	require_ok(t, err)
	defer release(&x)
	scale: Tensor(f64)
	scale, err = new_tensor(f64, []int{3})
	require_ok(t, err)
	defer release(&scale)
	wrong_scale: Tensor(f64)
	wrong_scale, err = new_tensor(f64, []int{4})
	require_ok(t, err)
	defer release(&wrong_scale)
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{0, 3})
	require_ok(t, err)
	defer release(&empty)
	xv := tape_add_leaf_borrow(&tape, &x) or_else panic("leaf")
	sv := tape_add_leaf_borrow(&tape, &scale) or_else panic("leaf")
	wv := tape_add_leaf_borrow(&tape, &wrong_scale) or_else panic("leaf")
	ev := tape_add_leaf_borrow(&tape, &empty) or_else panic("leaf")

	_, err = tape_batch_norm(&tape, xv, wv, 1e-5)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	short: [2]f64
	_, err = tape_batch_norm(&tape, xv, sv, 1e-5, mean = short[:])
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	_, err = tape_batch_norm(&tape, ev, sv, 1e-5)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = tape_batch_norm(&tape, Tape_Value(f64){}, sv, 1e-5)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)

	// Without tracked parents the result has no gradient.
	constant := tape_add_leaf_borrow(&tape, &x, false) or_else panic("leaf")
	constant_scale := tape_add_leaf_borrow(&tape, &scale, false) or_else panic("leaf")
	normalized := tape_batch_norm(&tape, constant, constant_scale, 1e-5) or_else panic("batch_norm")
	_, grad_err := tape_gradient(&tape, normalized)
	testing.expect_value(t, grad_err, Tensor_Error.Gradient_Not_Tracked)
}

batch_norm_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 4, allocator) or_return
	defer tape_release(&tape)
	x := new_tensor(f64, []int{4, 2, 3}, allocator) or_return
	defer release(&x)
	for &value, i in x.data {
		value = f64(i % 5)
	}
	scale := new_tensor(f64, []int{3}, allocator) or_return
	defer release(&scale)
	fill(&scale, 2)
	xv := tape_add_leaf_borrow(&tape, &x) or_return
	sv := tape_add_leaf_borrow(&tape, &scale) or_return
	transposed := tape_transpose(&tape, xv) or_return
	// The transposed [3, 2, 4] view has four features and is not contiguous.
	other_scale := new_tensor(f64, []int{4}, allocator) or_return
	defer release(&other_scale)
	ov := tape_add_leaf_borrow(&tape, &other_scale) or_return
	first := tape_batch_norm(&tape, xv, sv, 1e-5) or_return
	second := tape_batch_norm(&tape, transposed, ov, 1e-5) or_return
	first_total := tape_sum_all(&tape, first) or_return
	second_total := tape_sum_all(&tape, second) or_return
	total := tape_add(&tape, first_total, second_total) or_return
	return tape_backward(&tape, total)
}

@(test)
test_tape_batch_norm_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, batch_norm_allocation_scenario)
}
