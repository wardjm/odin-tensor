package tensor

import "core:math"
import "core:mem"
import "core:testing"

@(test)
test_tape_copy_leaf_snapshots_strided_source :: proc(t: ^testing.T) {
	owner, owner_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	require_ok(t, owner_err)
	defer release(&owner)
	shape_buffer := [1]int{}
	stride_buffer := [1]int{}
	reversed, view_err := slice_view(as_view(owner), 0, 2, 3, -1, shape_buffer[:], stride_buffer[:])
	require_ok(t, view_err)

	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	leaf, leaf_err := tape_add_leaf_copy(&tape, reversed)
	require_ok(t, leaf_err)
	owner.data[0], owner.data[1], owner.data[2] = 100, 200, 300

	leaf_view, value_err := tape_value(&tape, leaf)
	require_ok(t, value_err)
	testing.expect_value(t, leaf_view.data[0], 3)
	testing.expect_value(t, leaf_view.data[1], 2)
	testing.expect_value(t, leaf_view.data[2], 1)
	squared, square_err := mul_views(leaf_view, leaf_view)
	require_ok(t, square_err)
	child, record_err := tape_record_binary(&tape, &squared, .Mul, leaf, leaf)
	require_ok(t, record_err)
	seed, seed_err := from_slice(f64, []f64{1, 1, 1}, []int{3})
	require_ok(t, seed_err)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
	gradient, grad_err := tape_gradient(&tape, leaf)
	require_ok(t, grad_err)
	testing.expect_value(t, gradient.data[0], 6)
	testing.expect_value(t, gradient.data[1], 4)
	testing.expect_value(t, gradient.data[2], 2)
}

@(test)
test_tape_rejects_invalid_views_and_operation_shapes :: proc(t: ^testing.T) {
	backing := []f32{1}
	invalid_view := Tensor_View(f32){data = backing, shape = []int{2}, strides = []int{1}}
	tape, tape_err := tape_init(f32, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	_, invalid_leaf_err := tape_add_leaf_view(&tape, invalid_view)
	testing.expect_value(t, invalid_leaf_err, .Out_Of_Bounds)
	testing.expect_value(t, len(tape.nodes), 0)

	left, left_err := from_slice(f32, []f32{1, 2}, []int{2})
	require_ok(t, left_err)
	defer release(&left)
	right, right_err := from_slice(f32, []f32{3, 4}, []int{2})
	require_ok(t, right_err)
	defer release(&right)
	left_value, left_value_err := tape_add_leaf_borrow(&tape, &left)
	require_ok(t, left_value_err)
	right_value, right_value_err := tape_add_leaf_borrow(&tape, &right)
	require_ok(t, right_value_err)

	wrong_output, output_err := new_tensor(f32, []int{1})
	require_ok(t, output_err)
	_, record_err := tape_record_binary(&tape, &wrong_output, .Add, left_value, right_value)
	testing.expect_value(t, record_err, .Shape_Mismatch)
	testing.expect_value(t, len(wrong_output.data), 1)
	testing.expect_value(t, len(tape.nodes), 2)
	release(&wrong_output)

	unary_output, unary_output_err := new_tensor(f32, []int{2})
	require_ok(t, unary_output_err)
	_, unary_record_err := tape_record_unary(&tape, &unary_output, .Add, left_value)
	testing.expect_value(t, unary_record_err, .Unsupported_Operation)
	testing.expect_value(t, len(unary_output.data), 2)
	testing.expect_value(t, len(tape.nodes), 2)
	release(&unary_output)
}

@(test)
test_tape_records_reductions_and_validates_ownership :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, input_err)
	defer release(&input)

	tape, tape_err := tape_init(f64, 5)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, parent_err)

	global_sum, sum_err := sum_all(input)
	require_ok(t, sum_err)
	testing.expect_value(t, global_sum.data[0], 21)
	global_sum_value, sum_record_err := tape_record_sum_all(&tape, &global_sum, parent)
	require_ok(t, sum_record_err)
	testing.expect_value(t, len(global_sum.data), 0)
	expect_parent_gradient(t, &tape, global_sum_value, []f64{2}, []int{}, parent, []f64{2, 2, 2, 2, 2, 2})

	global_mean, mean_err := mean_all(input)
	require_ok(t, mean_err)
	testing.expect_value(t, global_mean.data[0], 3.5)
	global_mean_value, mean_record_err := tape_record_mean_all(&tape, &global_mean, parent)
	require_ok(t, mean_record_err)
	testing.expect_value(t, len(global_mean.data), 0)
	expect_parent_gradient(t, &tape, global_mean_value, []f64{6}, []int{}, parent, []f64{1, 1, 1, 1, 1, 1})

	axis_sum, axis_sum_err := sum_axis(input, 1)
	require_ok(t, axis_sum_err)
	testing.expect_value(t, axis_sum.data[0], 6)
	testing.expect_value(t, axis_sum.data[1], 15)
	axis_sum_value, axis_sum_record_err := tape_record_sum_axis(&tape, &axis_sum, parent, 1)
	require_ok(t, axis_sum_record_err)
	testing.expect_value(t, len(axis_sum.data), 0)
	// A seed of shape [2] spreads each row's value across that row only.
	expect_parent_gradient(t, &tape, axis_sum_value, []f64{1, 2}, []int{2}, parent, []f64{1, 1, 1, 2, 2, 2})

	axis_mean, axis_mean_err := mean_axis(input, 1, true)
	require_ok(t, axis_mean_err)
	testing.expect_value(t, axis_mean.data[0], 2)
	testing.expect_value(t, axis_mean.data[1], 5)
	axis_mean_value, axis_mean_record_err := tape_record_mean_axis(&tape, &axis_mean, parent, 1, true)
	require_ok(t, axis_mean_record_err)
	testing.expect_value(t, len(axis_mean.data), 0)
	// The kept axis makes the seed shape [2,1]; each row's value is divided by 3.
	expect_parent_gradient(t, &tape, axis_mean_value, []f64{3, 6}, []int{2, 1}, parent, []f64{1, 1, 1, 2, 2, 2})
	testing.expect_value(t, len(tape.nodes), 5)

	invalid_axis_output, invalid_axis_err := sum_axis(input, 0)
	require_ok(t, invalid_axis_err)
	_, record_axis_err := tape_record_sum_axis(&tape, &invalid_axis_output, parent, 2)
	testing.expect_value(t, record_axis_err, .Out_Of_Bounds)
	testing.expect_value(t, len(tape.nodes), 5)
	testing.expect_value(t, len(invalid_axis_output.data), 3)
	release(&invalid_axis_output)

	wrong_shape, wrong_shape_err := new_tensor(f64, []int{1, 3})
	require_ok(t, wrong_shape_err)
	_, record_shape_err := tape_record_mean_axis(&tape, &wrong_shape, parent, 1)
	testing.expect_value(t, record_shape_err, .Shape_Mismatch)
	testing.expect_value(t, len(tape.nodes), 5)
	testing.expect_value(t, len(wrong_shape.data), 3)
	release(&wrong_shape)

	foreign_tape, foreign_tape_err := tape_init(f64, 1)
	require_ok(t, foreign_tape_err)
	defer tape_release(&foreign_tape)
	foreign_parent, foreign_parent_err := tape_add_leaf_borrow(&foreign_tape, &input)
	require_ok(t, foreign_parent_err)
	wrong_tape_output, wrong_tape_output_err := sum_all(input)
	require_ok(t, wrong_tape_output_err)
	_, wrong_tape_record_err := tape_record_sum_all(&tape, &wrong_tape_output, foreign_parent)
	testing.expect_value(t, wrong_tape_record_err, .Invalid_Tape)
	testing.expect_value(t, len(tape.nodes), 5)
	testing.expect_value(t, len(wrong_tape_output.data), 1)
	release(&wrong_tape_output)

	grown_tape, grown_tape_err := tape_init(f64, 1)
	require_ok(t, grown_tape_err)
	defer tape_release(&grown_tape)
	grown_parent, grown_parent_err := tape_add_leaf_borrow(&grown_tape, &input)
	require_ok(t, grown_parent_err)
	grown_output, grown_output_err := sum_all(input)
	require_ok(t, grown_output_err)
	// The reservation is only a hint; recording past it grows the tape.
	_, grown_record_err := tape_record_sum_all(&grown_tape, &grown_output, grown_parent)
	require_ok(t, grown_record_err)
	testing.expect_value(t, len(grown_tape.nodes), 2)
	testing.expect_value(t, len(grown_output.data), 0)
}

// expect_parent_gradient clears the tape's gradients, runs backward from root
// with the given seed and checks the gradient that reaches parent.
expect_parent_gradient :: proc(t: ^testing.T, tape: ^Tape(f64), root: Tape_Value(f64), seed_values: []f64, seed_shape: []int, parent: Tape_Value(f64), expected: []f64, loc := #caller_location) {
	seed, seed_err := from_slice(f64, seed_values, seed_shape)
	require_ok(t, seed_err, loc)
	defer release(&seed)
	tape_zero_grad(tape)
	require_ok(t, tape_backward_with_seed(tape, root, seed), loc)
	gradient, grad_err := tape_gradient(tape, parent)
	require_ok(t, grad_err, loc)
	expect_f64_data(t, gradient.data, expected, loc = loc)
}

@(test)
test_tape_sum_all_backward_broadcasts_and_accumulates :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	require_ok(t, input_err)
	defer release(&input)

	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, parent_err)
	output, output_err := sum_all(input)
	require_ok(t, output_err)
	testing.expect_value(t, output.data[0], 10)
	loss, record_err := tape_record_sum_all(&tape, &output, parent)
	require_ok(t, record_err)
	testing.expect_value(t, tape_backward(&tape, loss), .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)
	for value in gradient.data {
		testing.expect_value(t, value, 1)
	}
	testing.expect_value(t, tape_backward(&tape, loss), .None)
	for value in gradient.data {
		testing.expect_value(t, value, 2)
	}

	scalar, scalar_err := from_slice(f64, []f64{7}, []int{})
	require_ok(t, scalar_err)
	defer release(&scalar)
	scalar_tape, scalar_tape_err := tape_init(f64, 2)
	require_ok(t, scalar_tape_err)
	defer tape_release(&scalar_tape)
	scalar_parent, scalar_parent_err := tape_add_leaf_borrow(&scalar_tape, &scalar)
	require_ok(t, scalar_parent_err)
	scalar_output, scalar_output_err := sum_all(scalar)
	require_ok(t, scalar_output_err)
	testing.expect_value(t, scalar_output.data[0], 7)
	scalar_loss, scalar_record_err := tape_record_sum_all(&scalar_tape, &scalar_output, scalar_parent)
	require_ok(t, scalar_record_err)
	testing.expect_value(t, tape_backward(&scalar_tape, scalar_loss), .None)
	scalar_gradient, scalar_gradient_err := tape_gradient(&scalar_tape, scalar_parent)
	require_ok(t, scalar_gradient_err)
	testing.expect_value(t, scalar_gradient.data[0], 1)

	empty, empty_err := new_tensor(f64, []int{0, 2})
	require_ok(t, empty_err)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	require_ok(t, empty_tape_err)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	require_ok(t, empty_parent_err)
	empty_output, empty_output_err := sum_all(empty)
	require_ok(t, empty_output_err)
	testing.expect_value(t, empty_output.data[0], 0)
	empty_loss, empty_record_err := tape_record_sum_all(&empty_tape, &empty_output, empty_parent)
	require_ok(t, empty_record_err)
	testing.expect_value(t, tape_backward(&empty_tape, empty_loss), .None)
	empty_gradient, empty_gradient_err := tape_gradient(&empty_tape, empty_parent)
	require_ok(t, empty_gradient_err)
	testing.expect_value(t, len(empty_gradient.data), 0)

	owner, owner_err := from_slice(f64, []f64{10, 20, 30, 40}, []int{4})
	require_ok(t, owner_err)
	defer release(&owner)
	shape_buffer := [1]int{}
	stride_buffer := [1]int{}
	reversed, view_err := slice_view(as_view(owner), 0, 3, 3, -1, shape_buffer[:], stride_buffer[:])
	require_ok(t, view_err)
	view_tape, view_tape_err := tape_init(f64, 2)
	require_ok(t, view_tape_err)
	defer tape_release(&view_tape)
	view_parent, view_parent_err := tape_add_leaf_view(&view_tape, reversed)
	require_ok(t, view_parent_err)
	view_output, view_output_err := sum_all_views(reversed)
	require_ok(t, view_output_err)
	testing.expect_value(t, view_output.data[0], 90)
	view_loss, view_record_err := tape_record_sum_all(&view_tape, &view_output, view_parent)
	require_ok(t, view_record_err)
	testing.expect_value(t, tape_backward(&view_tape, view_loss), .None)
	view_gradient, view_gradient_err := tape_gradient(&view_tape, view_parent)
	require_ok(t, view_gradient_err)
	testing.expect_value(t, view_gradient.data[0], 1)
	testing.expect_value(t, view_gradient.data[1], 1)
	testing.expect_value(t, view_gradient.data[2], 1)
}

@(test)
test_tape_mean_all_backward_scales_and_accumulates :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	require_ok(t, input_err)
	defer release(&input)

	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, parent_err)
	output, output_err := mean_all(input)
	require_ok(t, output_err)
	testing.expect_value(t, output.data[0], 2.5)
	loss, record_err := tape_record_mean_all(&tape, &output, parent)
	require_ok(t, record_err)
	testing.expect_value(t, tape_backward(&tape, loss), .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)
	for value in gradient.data {
		testing.expect_value(t, value, 0.25)
	}
	testing.expect_value(t, tape_backward(&tape, loss), .None)
	for value in gradient.data {
		testing.expect_value(t, value, 0.5)
	}

	scalar, scalar_err := from_slice(f64, []f64{7}, []int{})
	require_ok(t, scalar_err)
	defer release(&scalar)
	scalar_tape, scalar_tape_err := tape_init(f64, 2)
	require_ok(t, scalar_tape_err)
	defer tape_release(&scalar_tape)
	scalar_parent, scalar_parent_err := tape_add_leaf_borrow(&scalar_tape, &scalar)
	require_ok(t, scalar_parent_err)
	scalar_output, scalar_output_err := mean_all(scalar)
	require_ok(t, scalar_output_err)
	testing.expect_value(t, scalar_output.data[0], 7)
	scalar_loss, scalar_record_err := tape_record_mean_all(&scalar_tape, &scalar_output, scalar_parent)
	require_ok(t, scalar_record_err)
	testing.expect_value(t, tape_backward(&scalar_tape, scalar_loss), .None)
	scalar_gradient, scalar_gradient_err := tape_gradient(&scalar_tape, scalar_parent)
	require_ok(t, scalar_gradient_err)
	testing.expect_value(t, scalar_gradient.data[0], 1)

	empty, empty_err := new_tensor(f64, []int{0, 2})
	require_ok(t, empty_err)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	require_ok(t, empty_tape_err)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	require_ok(t, empty_parent_err)
	_, failed_output_err := mean_all(empty)
	testing.expect_value(t, failed_output_err, .Empty_Reduction)
	testing.expect_value(t, len(empty_tape.nodes), 1)

	manual_output, manual_output_err := new_tensor(f64, []int{})
	require_ok(t, manual_output_err)
	_, manual_record_err := tape_record_mean_all(&empty_tape, &manual_output, empty_parent)
	testing.expect_value(t, manual_record_err, .Empty_Reduction)
	testing.expect_value(t, len(empty_tape.nodes), 1)
	testing.expect_value(t, len(manual_output.data), 1)
	release(&manual_output)
}

@(test)
test_tape_sum_axis_backward_expands_across_reduced_axis :: proc(t: ^testing.T) {
	// A one-dimensional reduction produces a scalar upstream gradient.
	vector, vector_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	require_ok(t, vector_err)
	defer release(&vector)
	vector_tape, vector_tape_err := tape_init(f64, 2)
	require_ok(t, vector_tape_err)
	defer tape_release(&vector_tape)
	vector_parent, vector_parent_err := tape_add_leaf_borrow(&vector_tape, &vector)
	require_ok(t, vector_parent_err)
	vector_output, vector_output_err := sum_axis(vector, 0)
	require_ok(t, vector_output_err)
	testing.expect_value(t, vector_output.data[0], 6)
	vector_loss, vector_record_err := tape_record_sum_axis(&vector_tape, &vector_output, vector_parent, 0)
	require_ok(t, vector_record_err)
	vector_seed, vector_seed_err := from_slice(f64, []f64{2}, []int{})
	require_ok(t, vector_seed_err)
	defer release(&vector_seed)
	testing.expect_value(t, tape_backward_with_seed(&vector_tape, vector_loss, vector_seed), .None)
	vector_gradient, vector_gradient_err := tape_gradient(&vector_tape, vector_parent)
	require_ok(t, vector_gradient_err)
	for value in vector_gradient.data {
		testing.expect_value(t, value, 2)
	}

	// Batched reduction with keep_dims=false and keep_dims=true.
	batch, batch_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, batch_err)
	defer release(&batch)
	batch_tape, batch_tape_err := tape_init(f64, 3)
	require_ok(t, batch_tape_err)
	defer tape_release(&batch_tape)
	batch_parent, batch_parent_err := tape_add_leaf_borrow(&batch_tape, &batch)
	require_ok(t, batch_parent_err)
	row_sums, row_sums_err := sum_axis(batch, 1)
	require_ok(t, row_sums_err)
	row_loss, row_record_err := tape_record_sum_axis(&batch_tape, &row_sums, batch_parent, 1)
	require_ok(t, row_record_err)
	row_seed, row_seed_err := from_slice(f64, []f64{10, 20}, []int{2})
	require_ok(t, row_seed_err)
	defer release(&row_seed)
	testing.expect_value(t, tape_backward_with_seed(&batch_tape, row_loss, row_seed), .None)
	batch_gradient, batch_gradient_err := tape_gradient(&batch_tape, batch_parent)
	require_ok(t, batch_gradient_err)
	testing.expect_value(t, batch_gradient.data[0], 10)
	testing.expect_value(t, batch_gradient.data[2], 10)
	testing.expect_value(t, batch_gradient.data[3], 20)
	testing.expect_value(t, batch_gradient.data[5], 20)

	column_sums, column_sums_err := sum_axis(batch, 0, true)
	require_ok(t, column_sums_err)
	column_loss, column_record_err := tape_record_sum_axis(&batch_tape, &column_sums, batch_parent, 0, true)
	require_ok(t, column_record_err)
	column_seed, column_seed_err := from_slice(f64, []f64{1, 2, 3}, []int{1, 3})
	require_ok(t, column_seed_err)
	defer release(&column_seed)
	testing.expect_value(t, tape_backward_with_seed(&batch_tape, column_loss, column_seed), .None)
	testing.expect_value(t, batch_gradient.data[0], 11)
	testing.expect_value(t, batch_gradient.data[1], 12)
	testing.expect_value(t, batch_gradient.data[2], 13)
	testing.expect_value(t, batch_gradient.data[3], 21)
	testing.expect_value(t, batch_gradient.data[4], 22)
	testing.expect_value(t, batch_gradient.data[5], 23)

	// Higher-rank reduction maps upstream coordinates around a removed axis.
	higher, higher_err := from_slice(f64, []f64{0, 0, 0, 0, 0, 0, 0, 0}, []int{2, 2, 2})
	require_ok(t, higher_err)
	defer release(&higher)
	higher_tape, higher_tape_err := tape_init(f64, 2)
	require_ok(t, higher_tape_err)
	defer tape_release(&higher_tape)
	higher_parent, higher_parent_err := tape_add_leaf_borrow(&higher_tape, &higher)
	require_ok(t, higher_parent_err)
	higher_output, higher_output_err := sum_axis(higher, 1)
	require_ok(t, higher_output_err)
	higher_loss, higher_record_err := tape_record_sum_axis(&higher_tape, &higher_output, higher_parent, 1)
	require_ok(t, higher_record_err)
	higher_seed, higher_seed_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	require_ok(t, higher_seed_err)
	defer release(&higher_seed)
	testing.expect_value(t, tape_backward_with_seed(&higher_tape, higher_loss, higher_seed), .None)
	higher_gradient, higher_gradient_err := tape_gradient(&higher_tape, higher_parent)
	require_ok(t, higher_gradient_err)
	testing.expect_value(t, higher_gradient.data[0], 1)
	testing.expect_value(t, higher_gradient.data[1], 2)
	testing.expect_value(t, higher_gradient.data[2], 1)
	testing.expect_value(t, higher_gradient.data[3], 2)
	testing.expect_value(t, higher_gradient.data[4], 3)
	testing.expect_value(t, higher_gradient.data[5], 4)
	testing.expect_value(t, higher_gradient.data[6], 3)
	testing.expect_value(t, higher_gradient.data[7], 4)
}

@(test)
test_tape_sum_axis_backward_handles_strided_and_empty_views :: proc(t: ^testing.T) {
	owner, owner_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, owner_err)
	defer release(&owner)
	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, transposed_err := transpose_view(as_view(owner), transposed_shape[:], transposed_strides[:])
	require_ok(t, transposed_err)
	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_view(&tape, transposed)
	require_ok(t, parent_err)
	output, output_err := sum_axis_views(transposed, 1)
	require_ok(t, output_err)
	loss, record_err := tape_record_sum_axis(&tape, &output, parent, 1)
	require_ok(t, record_err)
	seed, seed_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	require_ok(t, seed_err)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, loss, seed), .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)
	testing.expect_value(t, gradient.data[0], 1)
	testing.expect_value(t, gradient.data[1], 1)
	testing.expect_value(t, gradient.data[2], 2)
	testing.expect_value(t, gradient.data[3], 2)
	testing.expect_value(t, gradient.data[4], 3)
	testing.expect_value(t, gradient.data[5], 3)

	empty, empty_err := new_tensor(f64, []int{2, 0, 3})
	require_ok(t, empty_err)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	require_ok(t, empty_tape_err)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	require_ok(t, empty_parent_err)
	empty_output, empty_output_err := sum_axis(empty, 1)
	require_ok(t, empty_output_err)
	testing.expect_value(t, len(empty_output.data), 6)
	for value in empty_output.data {
		testing.expect_value(t, value, 0)
	}
	empty_loss, empty_record_err := tape_record_sum_axis(&empty_tape, &empty_output, empty_parent, 1)
	require_ok(t, empty_record_err)
	empty_seed, empty_seed_err := new_tensor(f64, []int{2, 3})
	require_ok(t, empty_seed_err)
	defer release(&empty_seed)
	for i in 0..<len(empty_seed.data) {
		empty_seed.data[i] = f64(i+1)
	}
	testing.expect_value(t, tape_backward_with_seed(&empty_tape, empty_loss, empty_seed), .None)
	empty_gradient, empty_gradient_err := tape_gradient(&empty_tape, empty_parent)
	require_ok(t, empty_gradient_err)
	testing.expect_value(t, len(empty_gradient.data), 0)
}

@(test)
test_tape_mean_axis_backward_scales_and_handles_views :: proc(t: ^testing.T) {
	batch, batch_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, batch_err)
	defer release(&batch)
	tape, tape_err := tape_init(f64, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &batch)
	require_ok(t, parent_err)

	row_means, row_means_err := mean_axis(batch, 1)
	require_ok(t, row_means_err)
	testing.expect_value(t, row_means.data[0], 2)
	testing.expect_value(t, row_means.data[1], 5)
	row_loss, row_record_err := tape_record_mean_axis(&tape, &row_means, parent, 1)
	require_ok(t, row_record_err)
	row_seed, row_seed_err := from_slice(f64, []f64{6, 9}, []int{2})
	require_ok(t, row_seed_err)
	defer release(&row_seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, row_loss, row_seed), .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)
	testing.expect_value(t, gradient.data[0], 2)
	testing.expect_value(t, gradient.data[1], 2)
	testing.expect_value(t, gradient.data[2], 2)
	testing.expect_value(t, gradient.data[3], 3)
	testing.expect_value(t, gradient.data[4], 3)
	testing.expect_value(t, gradient.data[5], 3)

	// Keep the reduced axis and reduce a different dimension. Dividing by 2
	// maps each seeded column value equally to the two rows.
	column_means, column_means_err := mean_axis(batch, 0, true)
	require_ok(t, column_means_err)
	column_loss, column_record_err := tape_record_mean_axis(&tape, &column_means, parent, 0, true)
	require_ok(t, column_record_err)
	column_seed, column_seed_err := from_slice(f64, []f64{4, 8, 12}, []int{1, 3})
	require_ok(t, column_seed_err)
	defer release(&column_seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, column_loss, column_seed), .None)
	testing.expect_value(t, gradient.data[0], 4)
	testing.expect_value(t, gradient.data[1], 6)
	testing.expect_value(t, gradient.data[2], 8)
	testing.expect_value(t, gradient.data[3], 5)
	testing.expect_value(t, gradient.data[4], 7)
	testing.expect_value(t, gradient.data[5], 9)

	// A transposed view retains logical gradient order and the correct divisor.
	owner, owner_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, owner_err)
	defer release(&owner)
	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, transposed_err := transpose_view(as_view(owner), transposed_shape[:], transposed_strides[:])
	require_ok(t, transposed_err)
	view_tape, view_tape_err := tape_init(f64, 2)
	require_ok(t, view_tape_err)
	defer tape_release(&view_tape)
	view_parent, view_parent_err := tape_add_leaf_view(&view_tape, transposed)
	require_ok(t, view_parent_err)
	view_output, view_output_err := mean_axis_views(transposed, 1)
	require_ok(t, view_output_err)
	view_loss, view_record_err := tape_record_mean_axis(&view_tape, &view_output, view_parent, 1)
	require_ok(t, view_record_err)
	view_seed, view_seed_err := from_slice(f64, []f64{2, 4, 6}, []int{3})
	require_ok(t, view_seed_err)
	defer release(&view_seed)
	testing.expect_value(t, tape_backward_with_seed(&view_tape, view_loss, view_seed), .None)
	view_gradient, view_gradient_err := tape_gradient(&view_tape, view_parent)
	require_ok(t, view_gradient_err)
	testing.expect_value(t, view_gradient.data[0], 1)
	testing.expect_value(t, view_gradient.data[1], 1)
	testing.expect_value(t, view_gradient.data[2], 2)
	testing.expect_value(t, view_gradient.data[3], 2)
	testing.expect_value(t, view_gradient.data[4], 3)
	testing.expect_value(t, view_gradient.data[5], 3)
}

@(test)
test_tape_mean_axis_empty_reduction_does_not_record :: proc(t: ^testing.T) {
	empty, empty_err := new_tensor(f64, []int{2, 0, 3})
	require_ok(t, empty_err)
	defer release(&empty)
	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &empty)
	require_ok(t, parent_err)

	_, eager_err := mean_axis(empty, 1)
	testing.expect_value(t, eager_err, .Empty_Reduction)
	manual_output, output_err := new_tensor(f64, []int{2, 3})
	require_ok(t, output_err)
	_, record_err := tape_record_mean_axis(&tape, &manual_output, parent, 1)
	testing.expect_value(t, record_err, .Empty_Reduction)
	testing.expect_value(t, len(tape.nodes), 1)
	testing.expect_value(t, len(manual_output.data), 6)
	release(&manual_output)
}

@(test)
test_tape_scalar_backward_accumulates_and_zeroes :: proc(t: ^testing.T) {
	loss, err := from_slice(f32, []f32{3.5}, []int{})
	require_ok(t, err)
	defer release(&loss)

	tape, tape_err := tape_init(f32, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	loss_value, leaf_err := tape_add_leaf_borrow(&tape, &loss)
	require_ok(t, leaf_err)

	testing.expect_value(t, tape_backward(&tape, loss_value), .None)
	grad, grad_err := tape_gradient(&tape, loss_value)
	require_ok(t, grad_err)
	testing.expect_value(t, len(grad.shape), 0)
	testing.expect_value(t, grad.data[0], 1)
	testing.expect_value(t, tape_backward(&tape, loss_value), .None)
	testing.expect_value(t, grad.data[0], 2)
	tape_zero_grad(&tape)
	testing.expect_value(t, grad.data[0], 0)
}

@(test)
test_tape_non_scalar_seed_and_validation :: proc(t: ^testing.T) {
	input, err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	require_ok(t, err)
	defer release(&input)
	seed, seed_err := from_slice(f64, []f64{0.25, -1, 2}, []int{3})
	require_ok(t, seed_err)
	defer release(&seed)
	wrong_seed, wrong_seed_err := new_tensor(f64, []int{1, 3})
	require_ok(t, wrong_seed_err)
	defer release(&wrong_seed)

	tape, tape_err := tape_init(f64, 1)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	value, leaf_err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, leaf_err)
	testing.expect_value(t, tape_backward(&tape, value), .Shape_Mismatch)
	testing.expect_value(t, tape_backward_with_seed(&tape, value, wrong_seed), .Shape_Mismatch)
	testing.expect_value(t, tape_backward_with_seed(&tape, value, seed), .None)
	grad, grad_err := tape_gradient(&tape, value)
	require_ok(t, grad_err)
	testing.expect_value(t, grad.data[0], 0.25)
	testing.expect_value(t, grad.data[1], -1)
	testing.expect_value(t, grad.data[2], 2)
}

@(test)
test_tape_records_owned_operation_and_checks_graph_identity :: proc(t: ^testing.T) {
	input, err := from_slice(f32, []f32{1, 2}, []int{2})
	require_ok(t, err)
	defer release(&input)

	tape, tape_err := tape_init(f32, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, parent_err)

	output, output_err := from_slice(f32, []f32{2, 4}, []int{2})
	require_ok(t, output_err)
	child, child_err := tape_record_binary(&tape, &output, .Add, parent, parent)
	require_ok(t, child_err)
	testing.expect_value(t, len(output.data), 0)
	forward, forward_err := tape_value(&tape, child)
	require_ok(t, forward_err)
	testing.expect_value(t, forward.data[1], 4)
	seed, seed_err := from_slice(f32, []f32{2, 4}, []int{2})
	require_ok(t, seed_err)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
	child_grad, child_grad_err := tape_gradient(&tape, child)
	require_ok(t, child_grad_err)
	testing.expect_value(t, child_grad.data[0], 2)
	testing.expect_value(t, child_grad.data[1], 4)
	parent_grad, parent_grad_err := tape_gradient(&tape, parent)
	require_ok(t, parent_grad_err)
	testing.expect_value(t, parent_grad.data[0], 4)
	testing.expect_value(t, parent_grad.data[1], 8)

	foreign_tape, foreign_tape_err := tape_init(f32, 1)
	require_ok(t, foreign_tape_err)
	defer tape_release(&foreign_tape)
	foreign_value, foreign_err := tape_add_leaf_borrow(&foreign_tape, &input)
	require_ok(t, foreign_err)
	invalid_output, invalid_output_err := new_tensor(f32, []int{2})
	require_ok(t, invalid_output_err)
	defer release(&invalid_output)
	_, invalid_tape_err := tape_record_binary(&tape, &invalid_output, .Add, parent, foreign_value)
	testing.expect_value(t, invalid_tape_err, .Invalid_Tape)
}

@(test)
test_tape_elementwise_reverse_rules :: proc(t: ^testing.T) {
	a, a_err := from_slice(f64, []f64{6, 8}, []int{2})
	require_ok(t, a_err)
	defer release(&a)
	b, b_err := from_slice(f64, []f64{2, 4}, []int{2})
	require_ok(t, b_err)
	defer release(&b)
	seed, seed_err := from_slice(f64, []f64{1, 2}, []int{2})
	require_ok(t, seed_err)
	defer release(&seed)

	operations := [4]Tape_Operation{.Add, .Sub, .Mul, .Div}
	for operation in operations {
		tape, tape_err := tape_init(f64, 3)
		require_ok(t, tape_err)
		defer tape_release(&tape)
		av, av_err := tape_add_leaf_borrow(&tape, &a)
		require_ok(t, av_err)
		bv, bv_err := tape_add_leaf_borrow(&tape, &b)
		require_ok(t, bv_err)
		output, output_err := new_tensor(f64, []int{2})
		require_ok(t, output_err)
		child, child_err := tape_record_binary(&tape, &output, operation, av, bv)
		require_ok(t, child_err)
		testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
		grad_a, grad_a_err := tape_gradient(&tape, av)
		grad_b, grad_b_err := tape_gradient(&tape, bv)
		require_ok(t, grad_a_err)
		require_ok(t, grad_b_err)
		for i in 0..<2 {
			epsilon := 1e-5
			numeric_a := (binary_scalar_value(a.data[i]+epsilon, b.data[i], operation)-binary_scalar_value(a.data[i]-epsilon, b.data[i], operation))/(2*epsilon) * seed.data[i]
			numeric_b := (binary_scalar_value(a.data[i], b.data[i]+epsilon, operation)-binary_scalar_value(a.data[i], b.data[i]-epsilon, operation))/(2*epsilon) * seed.data[i]
			testing.expectf(t, abs(grad_a.data[i]-numeric_a) < 1e-8, "index %d: got %v, want %v", i, grad_a.data[i], numeric_a)
			testing.expectf(t, abs(grad_b.data[i]-numeric_b) < 1e-8, "index %d: got %v, want %v", i, grad_b.data[i], numeric_b)
		}
		#partial switch operation {
		case .Add:
			testing.expect_value(t, grad_a.data[0], 1)
			testing.expect_value(t, grad_a.data[1], 2)
			testing.expect_value(t, grad_b.data[0], 1)
			testing.expect_value(t, grad_b.data[1], 2)
		case .Sub:
			testing.expect_value(t, grad_a.data[0], 1)
			testing.expect_value(t, grad_a.data[1], 2)
			testing.expect_value(t, grad_b.data[0], -1)
			testing.expect_value(t, grad_b.data[1], -2)
		case .Mul:
			testing.expect_value(t, grad_a.data[0], 2)
			testing.expect_value(t, grad_a.data[1], 8)
			testing.expect_value(t, grad_b.data[0], 6)
			testing.expect_value(t, grad_b.data[1], 16)
		case .Div:
			testing.expect_value(t, grad_a.data[0], 0.5)
			testing.expect_value(t, grad_a.data[1], 0.5)
			testing.expect_value(t, grad_b.data[0], -1.5)
			testing.expect_value(t, grad_b.data[1], -1)
		}
		if operation == .Mul {
			// A repeated pass adds only the new seed contribution, not the
			// previously accumulated intermediate gradient a second time.
			testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
			testing.expect_value(t, grad_a.data[0], 4)
			testing.expect_value(t, grad_a.data[1], 16)
			testing.expect_value(t, grad_b.data[0], 12)
			testing.expect_value(t, grad_b.data[1], 32)
		}
	}
}

@(test)
test_tape_elementwise_broadcast_gradient_reduction :: proc(t: ^testing.T) {
	a, a_err := from_slice(f32, []f32{2, 3}, []int{2, 1})
	require_ok(t, a_err)
	defer release(&a)
	b, b_err := from_slice(f32, []f32{4, 5, 6}, []int{1, 3})
	require_ok(t, b_err)
	defer release(&b)
	seed, seed_err := from_slice(f32, []f32{1, 1, 1, 1, 1, 1}, []int{2, 3})
	require_ok(t, seed_err)
	defer release(&seed)

	tape, tape_err := tape_init(f32, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	av, av_err := tape_add_leaf_borrow(&tape, &a)
	require_ok(t, av_err)
	bv, bv_err := tape_add_leaf_borrow(&tape, &b)
	require_ok(t, bv_err)
	output, output_err := new_tensor(f32, []int{2, 3})
	require_ok(t, output_err)
	child, child_err := tape_record_binary(&tape, &output, .Mul, av, bv)
	require_ok(t, child_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
	grad_a, grad_a_err := tape_gradient(&tape, av)
	grad_b, grad_b_err := tape_gradient(&tape, bv)
	require_ok(t, grad_a_err)
	require_ok(t, grad_b_err)
	testing.expect_value(t, grad_a.data[0], 15)
	testing.expect_value(t, grad_a.data[1], 15)
	testing.expect_value(t, grad_b.data[0], 5)
	testing.expect_value(t, grad_b.data[1], 5)
	testing.expect_value(t, grad_b.data[2], 5)
}


@(test)
test_tape_activation_reverse_rules :: proc(t: ^testing.T) {
	signed_values := []f64{-1.1, 0.4, 1.7}
	// log and sqrt are only differentiable for positive inputs.
	positive_values := []f64{0.3, 0.4, 1.7}
	seed_values := []f64{0.5, -1.25, 2}
	operations := [6]Tape_Operation{.Relu, .Sigmoid, .Tanh, .Exp, .Log, .Sqrt}
	for operation in operations {
		input_values := signed_values
		if operation == .Log || operation == .Sqrt {
			input_values = positive_values
		}
		input, input_err := from_slice(f64, input_values, []int{3})
		require_ok(t, input_err)
		defer release(&input)
		seed, seed_err := from_slice(f64, seed_values, []int{3})
		require_ok(t, seed_err)
		defer release(&seed)

		tape, tape_err := tape_init(f64, 2)
		require_ok(t, tape_err)
		defer tape_release(&tape)
		parent, parent_err := tape_add_leaf_borrow(&tape, &input)
		require_ok(t, parent_err)
		output, output_err := activation_forward(input, operation)
		require_ok(t, output_err)
		child, child_err := tape_record_unary(&tape, &output, operation, parent)
		require_ok(t, child_err)
		testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
		gradient, gradient_err := tape_gradient(&tape, parent)
		require_ok(t, gradient_err)

		for i in 0..<len(input_values) {
			epsilon := 1e-6
			numeric := (activation_scalar(input_values[i]+epsilon, operation)-activation_scalar(input_values[i]-epsilon, operation))/(2*epsilon) * seed_values[i]
			testing.expectf(t, abs(gradient.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, gradient.data[i], numeric)
		}
		if operation == .Relu {
			testing.expect_value(t, gradient.data[0], 0)
			testing.expect_value(t, gradient.data[1], -1.25)
			testing.expect_value(t, gradient.data[2], 2)
		}
	}
}

@(test)
test_tape_activation_backward_through_strided_parent :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{1, 2, 4, 5, 8, 10}, []int{3, 2})
	require_ok(t, err)
	defer release(&storage)
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	// transposed is [[1, 4, 8], [2, 5, 10]] with strides {1, 2}.
	transposed, view_err := transpose_view(as_view(storage), shape_buffer[:], strides_buffer[:])
	require_ok(t, view_err)

	tape, tape_err := tape_init(f64, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_view(&tape, transposed)
	require_ok(t, parent_err)
	logarithm, log_err := tape_log(&tape, parent)
	require_ok(t, log_err)
	loss, loss_err := tape_sum_all(&tape, logarithm)
	require_ok(t, loss_err)
	testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
	// The log rule reads the strided input, so each gradient is 1/x in the
	// view's logical order.
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)
	expect_f64_data(t, gradient.data, []f64{1, 0.25, 0.125, 0.5, 0.2, 0.1})
}

activation_forward :: proc(input: Tensor(f64), operation: Tape_Operation) -> (output: Tensor(f64), err: Tensor_Error) {
	#partial switch operation {
	case .Relu: return relu(input)
	case .Sigmoid: return sigmoid(input)
	case .Tanh: return tanh(input)
	case .Exp: return exp(input)
	case .Log: return log(input)
	case .Sqrt: return sqrt(input)
	case: return output, .Unsupported_Operation
	}
}

activation_scalar :: proc(value: f64, operation: Tape_Operation) -> f64 {
	#partial switch operation {
	case .Relu: return max(value, 0)
	case .Sigmoid:
		if value >= 0 { return 1/(1+math.exp(-value)) }
		e := math.exp(value)
		return e/(1+e)
	case .Tanh: return math.tanh(value)
	case .Exp: return math.exp(value)
	case .Log: return math.ln(value)
	case .Sqrt: return math.sqrt(value)
	case: return 0
	}
}

@(test)
test_tape_dense_layer_end_to_end_gradients :: proc(t: ^testing.T) {
	input_values := []f64{0.3, -0.7}
	weight_values := []f64{0.2, -0.4, 0.6, 0.1}
	bias_values := []f64{0.05, -0.2}
	target_values := []f64{0.1, 0.8}
	input, input_err := from_slice(f64, input_values, []int{2})
	require_ok(t, input_err)
	defer release(&input)
	weights, weight_err := from_slice(f64, weight_values, []int{2, 2})
	require_ok(t, weight_err)
	defer release(&weights)
	bias, bias_err := from_slice(f64, bias_values, []int{2})
	require_ok(t, bias_err)
	defer release(&bias)
	target, target_err := from_slice(f64, target_values, []int{2})
	require_ok(t, target_err)
	defer release(&target)

	// One dense layer: tanh(input · weights + bias), trained with squared error.
	tape, tape_err := tape_init(f64, 9)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	input_value, err := tape_add_leaf_borrow(&tape, &input)
	require_ok(t, err)
	weight_value, weight_tape_err := tape_add_leaf_borrow(&tape, &weights)
	require_ok(t, weight_tape_err)
	bias_value, bias_tape_err := tape_add_leaf_borrow(&tape, &bias)
	require_ok(t, bias_tape_err)
	target_value, target_tape_err := tape_add_leaf_borrow(&tape, &target, false)
	require_ok(t, target_tape_err)

	// The tape_<op> helpers evaluate and record each step in one call.
	linear_value, linear_err := tape_dot(&tape, input_value, weight_value)
	require_ok(t, linear_err)
	biased_value, biased_err := tape_add(&tape, linear_value, bias_value)
	require_ok(t, biased_err)
	activated_value, activated_err := tape_tanh(&tape, biased_value)
	require_ok(t, activated_err)
	residual_value, residual_err := tape_sub(&tape, activated_value, target_value)
	require_ok(t, residual_err)
	loss_value, loss_err := tape_dot(&tape, residual_value, residual_value)
	require_ok(t, loss_err)
	testing.expect_value(t, tape_backward(&tape, loss_value), .None)

	input_grad, input_grad_err := tape_gradient(&tape, input_value)
	weight_grad, weight_grad_err := tape_gradient(&tape, weight_value)
	bias_grad, bias_grad_err := tape_gradient(&tape, bias_value)
	require_ok(t, input_grad_err)
	require_ok(t, weight_grad_err)
	require_ok(t, bias_grad_err)
	epsilon := 1e-6
	for i in 0..<len(input_values) {
		plus := make([]f64, len(input_values))
		minus := make([]f64, len(input_values))
		copy(plus, input_values)
		copy(minus, input_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dense_squared_loss(plus, weight_values, bias_values, target_values)-dense_squared_loss(minus, weight_values, bias_values, target_values))/(2*epsilon)
		testing.expectf(t, abs(input_grad.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, input_grad.data[i], numeric)
		delete(plus)
		delete(minus)
	}
	for i in 0..<len(weight_values) {
		plus := make([]f64, len(weight_values))
		minus := make([]f64, len(weight_values))
		copy(plus, weight_values)
		copy(minus, weight_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dense_squared_loss(input_values, plus, bias_values, target_values)-dense_squared_loss(input_values, minus, bias_values, target_values))/(2*epsilon)
		testing.expectf(t, abs(weight_grad.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, weight_grad.data[i], numeric)
		delete(plus)
		delete(minus)
	}
	for i in 0..<len(bias_values) {
		plus := make([]f64, len(bias_values))
		minus := make([]f64, len(bias_values))
		copy(plus, bias_values)
		copy(minus, bias_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dense_squared_loss(input_values, weight_values, plus, target_values)-dense_squared_loss(input_values, weight_values, minus, target_values))/(2*epsilon)
		testing.expectf(t, abs(bias_grad.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, bias_grad.data[i], numeric)
		delete(plus)
		delete(minus)
	}
}

dense_squared_loss :: proc(input, weights, bias, target: []f64) -> f64 {
	loss: f64 = 0
	for output_index in 0..<len(bias) {
		activation := bias[output_index]
		for input_index in 0..<len(input) {
			activation += input[input_index] * weights[input_index*len(bias)+output_index]
		}
		residual := math.tanh(activation) - target[output_index]
		loss += residual * residual
	}
	return loss
}

@(test)
test_tape_dot_and_matmul_reverse_rules :: proc(t: ^testing.T) {
	check_dot_reverse_case(t, []f64{1, -2, 3}, []int{3}, []f64{4, 0.5, -1}, []int{3}, []f64{1}, []int{}, .Dot)
	check_dot_reverse_case(t, []f64{1, 2, -1, 3, 4, 2}, []int{2, 3}, []f64{2, -1, 0.5}, []int{3}, []f64{1.5, -2}, []int{2}, .Dot)
	check_dot_reverse_case(t, []f64{1, -2, 3}, []int{3}, []f64{2, 1, -1, 3, 0.5, 4}, []int{3, 2}, []f64{-1, 2}, []int{2}, .Dot)
	check_dot_reverse_case(t, []f64{1, 2, -1, 3, 4, 2}, []int{2, 3}, []f64{2, 1, -1, 3, 0.5, 4}, []int{3, 2}, []f64{1, -2, 0.5, 3}, []int{2, 2}, .Dot)
	check_dot_reverse_case(t, []f64{1, 2, -1, 3, 4, 2}, []int{2, 3}, []f64{2, 1, -1, 3, 0.5, 4}, []int{3, 2}, []f64{1, -2, 0.5, 3}, []int{2, 2}, .Matmul)
}

@(test)
test_tape_batched_matmul_matches_finite_differences :: proc(t: ^testing.T) {
	values := []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9, 0.7, -0.5, 1.3, -0.2, 1.1, 0.6}
	check_tape_helper(t, .Matmul, values, []int{2, 2, 3}, values[:6], []int{3, 2})
	// a's leading batch axis of size 1 broadcasts, so its gradient sums both batches.
	check_tape_helper(t, .Matmul, values[:6], []int{1, 2, 3}, values, []int{2, 3, 2})
	// Batch axes broadcast on both sides.
	check_tape_helper(t, .Matmul, values, []int{2, 1, 2, 3}, values[:6], []int{2, 3, 1})
	// A rank-two a broadcasts against b's batch axes.
	check_tape_helper(t, .Matmul, values[:6], []int{2, 3}, values, []int{2, 1, 3, 2})
	// Vectors promote as in rank two.
	check_tape_helper(t, .Matmul, values[:3], []int{3}, values, []int{2, 3, 2})
	check_tape_helper(t, .Matmul, values, []int{2, 2, 3}, values[:3], []int{3})
}

@(test)
test_tape_batched_matmul_gradients_through_permuted_view :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9, 0.7, -0.5, 1.3, -0.2, 1.1, 0.6}, []int{2, 2, 3})
	require_ok(t, err)
	defer release(&storage)
	shape_buffer, strides_buffer := [3]int{}, [3]int{}
	// permuted transposes each batch's matrix: shape {2, 3, 2}, strides {6, 1, 3}.
	permuted, view_err := permute_view(as_view(storage), []int{0, 2, 1}, shape_buffer[:], strides_buffer[:])
	require_ok(t, view_err)
	contiguous: Tensor(f64)
	contiguous, err = copy_view(permuted)
	require_ok(t, err)
	defer release(&contiguous)
	b: Tensor(f64)
	b, err = from_slice(f64, []f64{2, -1, 0.5, 3}, []int{2, 2})
	require_ok(t, err)
	defer release(&b)
	seed: Tensor(f64)
	seed, err = new_tensor(f64, []int{2, 3, 2})
	require_ok(t, err)
	defer release(&seed)
	for &value, i in seed.data {
		value = 1 + math.sin(f64(i))
	}

	view_tape, view_tape_err := tape_init(f64)
	require_ok(t, view_tape_err)
	defer tape_release(&view_tape)
	view_a, view_a_err := tape_add_leaf_view(&view_tape, permuted)
	require_ok(t, view_a_err)
	view_b, view_b_err := tape_add_leaf_borrow(&view_tape, &b)
	require_ok(t, view_b_err)
	view_product, view_product_err := tape_matmul(&view_tape, view_a, view_b)
	require_ok(t, view_product_err)
	testing.expect_value(t, tape_backward_with_seed(&view_tape, view_product, seed), Tensor_Error.None)

	copy_tape, copy_tape_err := tape_init(f64)
	require_ok(t, copy_tape_err)
	defer tape_release(&copy_tape)
	copy_a, copy_a_err := tape_add_leaf_borrow(&copy_tape, &contiguous)
	require_ok(t, copy_a_err)
	copy_b, copy_b_err := tape_add_leaf_borrow(&copy_tape, &b)
	require_ok(t, copy_b_err)
	copy_product, copy_product_err := tape_matmul(&copy_tape, copy_a, copy_b)
	require_ok(t, copy_product_err)
	testing.expect_value(t, tape_backward_with_seed(&copy_tape, copy_product, seed), Tensor_Error.None)

	view_a_grad, view_a_grad_err := tape_gradient(&view_tape, view_a)
	require_ok(t, view_a_grad_err)
	copy_a_grad, copy_a_grad_err := tape_gradient(&copy_tape, copy_a)
	require_ok(t, copy_a_grad_err)
	expect_f64_data(t, view_a_grad.data, copy_a_grad.data)
	view_b_grad, view_b_grad_err := tape_gradient(&view_tape, view_b)
	require_ok(t, view_b_grad_err)
	copy_b_grad, copy_b_grad_err := tape_gradient(&copy_tape, copy_b)
	require_ok(t, copy_b_grad_err)
	expect_f64_data(t, view_b_grad.data, copy_b_grad.data)
}

// The other matmul backward tests are too small for the product kernel to
// block. Here a transposed [9,70] a broadcasts against a [2,70,10] b, so both
// gradient products hit the register tiles and both tails, k spans two strided
// blocks, and the second batch accumulates onto the first's gradient. With 3
// columns in a, a's gradient product is too narrow for a column tile, and with
// 1 column in b, so is b's; both then take the narrow column path.
@(test)
test_tape_blocked_matmul_backward_matches_naive :: proc(t: ^testing.T) {
	expect_matmul_backward_matches_naive(t, 9, 70, 10)
	expect_matmul_backward_matches_naive(t, 9, 3, 12)
	expect_matmul_backward_matches_naive(t, 9, 70, 1)
}

// expect_matmul_backward_matches_naive records a transposed [M,K] a times a
// [2,K,N] b and compares both gradients exactly with naive sums.
expect_matmul_backward_matches_naive :: proc(t: ^testing.T, M, K, N: int, loc := #caller_location) {
	BATCHES :: 2
	a, err := new_tensor(f64, []int{K, M})
	require_ok(t, err, loc)
	defer release(&a)
	for &value, i in a.data {
		value = math.sin(0.37*f64(i))
	}
	b: Tensor(f64)
	b, err = new_tensor(f64, []int{BATCHES, K, N})
	require_ok(t, err, loc)
	defer release(&b)
	for &value, i in b.data {
		value = math.cos(0.23*f64(i))
	}
	seed: Tensor(f64)
	seed, err = new_tensor(f64, []int{BATCHES, M, N})
	require_ok(t, err, loc)
	defer release(&seed)
	for &value, i in seed.data {
		value = 1 + math.sin(f64(i))
	}

	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err, loc)
	defer tape_release(&tape)
	av, bv, a_transposed, product: Tape_Value(f64)
	av, err = tape_add_leaf_borrow(&tape, &a)
	require_ok(t, err, loc)
	bv, err = tape_add_leaf_borrow(&tape, &b)
	require_ok(t, err, loc)
	a_transposed, err = tape_transpose(&tape, av)
	require_ok(t, err, loc)
	product, err = tape_matmul(&tape, a_transposed, bv)
	require_ok(t, err, loc)
	testing.expect_value(t, tape_backward_with_seed(&tape, product, seed), Tensor_Error.None, loc)

	// a's gradient is laid out as a, [K,M]; each element sums over both batches.
	expected_a := make([]f64, K*M)
	defer delete(expected_a)
	for p in 0..<K {
		for i in 0..<M {
			sum: f64 = 0
			for batch in 0..<BATCHES {
				for j in 0..<N {
					sum += seed.data[(batch*M + i)*N + j] * b.data[(batch*K + p)*N + j]
				}
			}
			expected_a[p*M + i] = sum
		}
	}
	expected_b := make([]f64, BATCHES*K*N)
	defer delete(expected_b)
	for batch in 0..<BATCHES {
		for p in 0..<K {
			for j in 0..<N {
				sum: f64 = 0
				for i in 0..<M {
					sum += a.data[p*M + i] * seed.data[(batch*M + i)*N + j]
				}
				expected_b[(batch*K + p)*N + j] = sum
			}
		}
	}
	a_grad, a_grad_err := tape_gradient(&tape, av)
	require_ok(t, a_grad_err, loc)
	expect_f64_data(t, a_grad.data, expected_a, loc = loc)
	b_grad, b_grad_err := tape_gradient(&tape, bv)
	require_ok(t, b_grad_err, loc)
	expect_f64_data(t, b_grad.data, expected_b, loc = loc)
}

@(test)
test_tape_dot_rejects_batched_operands :: proc(t: ^testing.T) {
	a, err := new_tensor(f64, []int{2, 2, 3})
	require_ok(t, err)
	defer release(&a)
	b: Tensor(f64)
	b, err = new_tensor(f64, []int{3, 2})
	require_ok(t, err)
	defer release(&b)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	av, bv: Tape_Value(f64)
	av, err = tape_add_leaf_borrow(&tape, &a)
	require_ok(t, err)
	bv, err = tape_add_leaf_borrow(&tape, &b)
	require_ok(t, err)
	_, err = tape_dot(&tape, av, bv)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
}

check_dot_reverse_case :: proc(t: ^testing.T, a_values: []f64, a_shape: []int, b_values: []f64, b_shape: []int, seed_values: []f64, seed_shape: []int, operation: Tape_Operation) {
	a, a_err := from_slice(f64, a_values, a_shape)
	require_ok(t, a_err)
	defer release(&a)
	b, b_err := from_slice(f64, b_values, b_shape)
	require_ok(t, b_err)
	defer release(&b)
	seed, seed_err := from_slice(f64, seed_values, seed_shape)
	require_ok(t, seed_err)
	defer release(&seed)

	tape, tape_err := tape_init(f64, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	av, av_err := tape_add_leaf_borrow(&tape, &a)
	require_ok(t, av_err)
	bv, bv_err := tape_add_leaf_borrow(&tape, &b)
	require_ok(t, bv_err)
	output, output_err := dot(a, b)
	require_ok(t, output_err)
	child, child_err := tape_record_binary(&tape, &output, operation, av, bv)
	require_ok(t, child_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, child, seed), .None)
	grad_a, grad_a_err := tape_gradient(&tape, av)
	grad_b, grad_b_err := tape_gradient(&tape, bv)
	require_ok(t, grad_a_err)
	require_ok(t, grad_b_err)

	epsilon := 1e-5
	for i in 0..<len(a_values) {
		plus := make([]f64, len(a_values))
		minus := make([]f64, len(a_values))
		copy(plus, a_values)
		copy(minus, a_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dot_weighted_value(plus, a_shape, b_values, b_shape, seed_values)-dot_weighted_value(minus, a_shape, b_values, b_shape, seed_values))/(2*epsilon)
		testing.expectf(t, abs(grad_a.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, grad_a.data[i], numeric)
		delete(plus)
		delete(minus)
	}
	for i in 0..<len(b_values) {
		plus := make([]f64, len(b_values))
		minus := make([]f64, len(b_values))
		copy(plus, b_values)
		copy(minus, b_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dot_weighted_value(a_values, a_shape, plus, b_shape, seed_values)-dot_weighted_value(a_values, a_shape, minus, b_shape, seed_values))/(2*epsilon)
		testing.expectf(t, abs(grad_b.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, grad_b.data[i], numeric)
		delete(plus)
		delete(minus)
	}
}

dot_weighted_value :: proc(a: []f64, a_shape: []int, b: []f64, b_shape: []int, seed: []f64) -> f64 {
	a_rank, b_rank := len(a_shape), len(b_shape)
	m, n := 1, 1
	if a_rank == 2 { m = a_shape[0] }
	if b_rank == 2 { n = b_shape[1] }
	inner_count := a_shape[a_rank-1]
	value: f64 = 0
	for i in 0..<m {
		for j in 0..<n {
			output_index := 0
			if a_rank == 2 && b_rank == 2 { output_index = i*n+j
			} else if a_rank == 2 { output_index = i
			} else if b_rank == 2 { output_index = j }
			for inner in 0..<inner_count {
				a_index := inner
				b_index := inner
				if a_rank == 2 { a_index = i*inner_count+inner }
				if b_rank == 2 { b_index = inner*n+j }
				value += seed[output_index]*a[a_index]*b[b_index]
			}
		}
	}
	return value
}

binary_scalar_value :: proc(a, b: f64, operation: Tape_Operation) -> f64 {
	#partial switch operation {
	case .Add: return a+b
	case .Sub: return a-b
	case .Mul: return a*b
	case .Div: return a/b
	case: return 0
	}
}

@(test)
test_reduction_gradients_match_finite_differences :: proc(t: ^testing.T) {
	// The cases cover rank-zero/global reductions, a batched global mean, a
	// higher-rank axis sum, and an axis mean over a transposed strided view.
	check_reduction_finite_difference(t, []f64{1.25}, []int{}, .Sum_All, -1, false, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Mean_All, -1, false, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9, 0.8, 0.1}, []int{2, 2, 2}, .Sum_Axis, 1, true, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Mean_Axis, 1, false, true)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Prod_All, -1, false, true)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9, 0.8, 0.1}, []int{2, 2, 2}, .Prod_Axis, 1, true, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Prod_Axis, 1, false, true)
	// Max and min have no ties here, so the gradient is unique. The transposed
	// cases check that the chosen element is found in logical order.
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Max_All, -1, false, true)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Min_All, -1, false, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9, 0.8, 0.1}, []int{2, 2, 2}, .Max_Axis, 1, true, false)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Min_Axis, 1, false, true)
	check_reduction_finite_difference(t, []f64{0.2, -0.7, 1.1, 0.4, -0.3, 0.9}, []int{2, 3}, .Max_Axis, 0, false, true)
}

check_reduction_finite_difference :: proc(t: ^testing.T, values: []f64, shape: []int, operation: Tape_Operation, axis: int, keep_dims, transpose: bool) {
	input, input_err := from_slice(f64, values, shape)
	require_ok(t, input_err)
	defer release(&input)
	input_view := as_view(input)
	view_shape := [2]int{}
	view_strides := [2]int{}
	if transpose {
		input_view, input_err = transpose_view(as_view(input), view_shape[:], view_strides[:])
		require_ok(t, input_err)
	}

	tape, tape_err := tape_init(f64, 2)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_view(&tape, input_view)
	require_ok(t, parent_err)
	output, output_err := reduction_output(input_view, operation, axis, keep_dims)
	require_ok(t, output_err)
	seed, seed_err := new_tensor(f64, output.shape)
	require_ok(t, seed_err)
	defer release(&seed)
	for i in 0..<len(seed.data) {
		seed.data[i] = 0.35 + f64(i)*0.17
	}
	loss, record_err := record_reduction(&tape, &output, operation, parent, axis, keep_dims)
	require_ok(t, record_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, loss, seed), .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	require_ok(t, gradient_err)

	// Central differences use epsilon=1e-6; f64 analytical and numerical
	// gradients must agree within 1e-8.
	epsilon := 1e-6
	for i in 0..<len(gradient.data) {
		plus := make([]f64, len(values))
		minus := make([]f64, len(values))
		copy(plus, values)
		copy(minus, values)
		storage_index := i
		if transpose {
			// Logical transposed coordinate (row, column) maps to backing
			// coordinate (column, row) in the 2x3 owner.
			storage_index = (i % 2)*3 + i/2
		}
		plus[storage_index] += epsilon
		minus[storage_index] -= epsilon
		numeric := (reduction_objective(plus, shape, transpose, operation, axis, keep_dims, seed.data)-reduction_objective(minus, shape, transpose, operation, axis, keep_dims, seed.data))/(2*epsilon)
		testing.expectf(t, abs(gradient.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, gradient.data[i], numeric)
		delete(plus)
		delete(minus)
	}
}

reduction_output :: proc(source: Tensor_View(f64), operation: Tape_Operation, axis: int, keep_dims: bool) -> (Tensor(f64), Tensor_Error) {
	#partial switch operation {
	case .Sum_All: return sum_all_views(source)
	case .Mean_All: return mean_all_views(source)
	case .Sum_Axis: return sum_axis_views(source, axis, keep_dims)
	case .Mean_Axis: return mean_axis_views(source, axis, keep_dims)
	case .Prod_All: return prod_all_views(source)
	case .Prod_Axis: return prod_axis_views(source, axis, keep_dims)
	case .Max_All: return max_all_views(source)
	case .Min_All: return min_all_views(source)
	case .Max_Axis: return max_axis_views(source, axis, keep_dims)
	case .Min_Axis: return min_axis_views(source, axis, keep_dims)
	case: return {}, .Unsupported_Operation
	}
}

record_reduction :: proc(tape: ^Tape(f64), output: ^Tensor(f64), operation: Tape_Operation, parent: Tape_Value(f64), axis: int, keep_dims: bool) -> (Tape_Value(f64), Tensor_Error) {
	#partial switch operation {
	case .Sum_All: return tape_record_sum_all(tape, output, parent)
	case .Mean_All: return tape_record_mean_all(tape, output, parent)
	case .Sum_Axis: return tape_record_sum_axis(tape, output, parent, axis, keep_dims)
	case .Mean_Axis: return tape_record_mean_axis(tape, output, parent, axis, keep_dims)
	case .Prod_All: return tape_record_prod_all(tape, output, parent)
	case .Prod_Axis: return tape_record_prod_axis(tape, output, parent, axis, keep_dims)
	case .Max_All: return tape_record_max_all(tape, output, parent)
	case .Min_All: return tape_record_min_all(tape, output, parent)
	case .Max_Axis: return tape_record_max_axis(tape, output, parent, axis, keep_dims)
	case .Min_Axis: return tape_record_min_axis(tape, output, parent, axis, keep_dims)
	case: return {}, .Unsupported_Operation
	}
}

reduction_objective :: proc(values: []f64, shape: []int, transpose: bool, operation: Tape_Operation, axis: int, keep_dims: bool, seed: []f64) -> f64 {
	input, input_err := from_slice(f64, values, shape)
	if input_err != .None { return 0 }
	defer release(&input)
	source := as_view(input)
	view_shape := [2]int{}
	view_strides := [2]int{}
	if transpose {
		source, input_err = transpose_view(as_view(input), view_shape[:], view_strides[:])
		if input_err != .None { return 0 }
	}
	output, output_err := reduction_output(source, operation, axis, keep_dims)
	if output_err != .None { return 0 }
	defer release(&output)
	objective: f64 = 0
	for i in 0..<len(output.data) {
		objective += output.data[i]*seed[i]
	}
	return objective
}

@(test)
test_tape_batched_mse_with_mean_all_matches_finite_differences :: proc(t: ^testing.T) {
	predictions := []f64{0.2, -0.4, 1.1, 0.7, -0.3, 0.9}
	targets := []f64{0.1, 0.5, 0.8, -0.2, 0.4, 0.3}
	prediction, prediction_err := from_slice(f64, predictions, []int{3, 2})
	require_ok(t, prediction_err)
	defer release(&prediction)
	target, target_err := from_slice(f64, targets, []int{3, 2})
	require_ok(t, target_err)
	defer release(&target)

	// MSE = mean_all((prediction - target) * (prediction - target)).
	tape, tape_err := tape_init(f64, 5)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	prediction_value, prediction_value_err := tape_add_leaf_borrow(&tape, &prediction)
	require_ok(t, prediction_value_err)
	target_value, target_value_err := tape_add_leaf_borrow(&tape, &target, false)
	require_ok(t, target_value_err)
	residual_value, residual_err := tape_sub(&tape, prediction_value, target_value)
	require_ok(t, residual_err)
	squared_value, squared_err := tape_mul(&tape, residual_value, residual_value)
	require_ok(t, squared_err)
	loss_value, loss_err := tape_mean_all(&tape, squared_value)
	require_ok(t, loss_err)
	testing.expect_value(t, tape_backward(&tape, loss_value), .None)
	gradient, gradient_err := tape_gradient(&tape, prediction_value)
	require_ok(t, gradient_err)

	epsilon := 1e-6
	for i in 0..<len(predictions) {
		plus := make([]f64, len(predictions))
		minus := make([]f64, len(predictions))
		copy(plus, predictions)
		copy(minus, predictions)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (batched_mse_value(plus, targets)-batched_mse_value(minus, targets))/(2*epsilon)
		testing.expectf(t, abs(gradient.data[i]-numeric) < 1e-8, "index %d: got %v, want %v", i, gradient.data[i], numeric)
		delete(plus)
		delete(minus)
	}
}

batched_mse_value :: proc(predictions, targets: []f64) -> f64 {
	value: f64 = 0
	for i in 0..<len(predictions) {
		residual := predictions[i]-targets[i]
		value += residual*residual
	}
	return value/f64(len(predictions))
}

@(test)
test_tape_value_and_gradient_reject_invalid_handles :: proc(t: ^testing.T) {
	tape, tape_err := tape_init(f64, 4)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	input, err := from_slice(f64, []f64{-1, 2}, []int{2})
	require_ok(t, err)
	defer release(&input)

	_, err = tape_value(&tape, Tape_Value(f64){})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_gradient(&tape, Tape_Value(f64){id = 0, identity = tape.identity})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	testing.expect_value(t, tape_backward(&tape, Tape_Value(f64){id = -1, identity = tape.identity}), Tensor_Error.Invalid_Tape)

	constant, constant_err := tape_add_leaf_borrow(&tape, &input, false)
	require_ok(t, constant_err)
	forward: Tensor_View(f64)
	forward, err = tape_value(&tape, constant)
	require_ok(t, err)
	testing.expect_value(t, forward.data[1], 2)
	_, err = tape_gradient(&tape, constant)
	testing.expect_value(t, err, Tensor_Error.Gradient_Not_Tracked)

	// Operations on constants inherit requires_grad = false.
	rectified: Tensor(f64)
	rectified, err = relu(input)
	require_ok(t, err)
	rectified_value, record_err := tape_record_unary(&tape, &rectified, .Relu, constant)
	require_ok(t, record_err)
	_, err = tape_gradient(&tape, rectified_value)
	testing.expect_value(t, err, Tensor_Error.Gradient_Not_Tracked)
	seed: Tensor(f64)
	seed, err = from_slice(f64, []f64{1, 1}, []int{2})
	require_ok(t, err)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, rectified_value, seed), Tensor_Error.Gradient_Not_Tracked)

	total: Tensor(f64)
	total, err = sum_all(input)
	require_ok(t, err)
	total_value: Tape_Value(f64)
	total_value, err = tape_record_sum_all(&tape, &total, rectified_value)
	require_ok(t, err)
	testing.expect_value(t, tape_backward(&tape, total_value), Tensor_Error.Gradient_Not_Tracked)
	testing.expect_value(t, tape_backward(&tape, rectified_value), Tensor_Error.Shape_Mismatch)
}

@(test)
test_tape_handles_survive_moves_and_reject_stale_tapes :: proc(t: ^testing.T) {
	left, err := from_slice(f64, []f64{1, 2}, []int{2})
	require_ok(t, err)
	defer release(&left)
	right: Tensor(f64)
	right, err = from_slice(f64, []f64{3, 4}, []int{2})
	require_ok(t, err)
	defer release(&right)

	original, tape_err := tape_init(f64)
	require_ok(t, tape_err)
	left_value, left_err := tape_add_leaf_borrow(&original, &left)
	require_ok(t, left_err)
	right_value, right_err := tape_add_leaf_borrow(&original, &right)
	require_ok(t, right_err)

	// Moving the Tape value keeps its handles valid.
	tape := original
	original = {}
	defer tape_release(&tape)
	product, product_err := tape_mul(&tape, left_value, right_value)
	require_ok(t, product_err)
	_, err = tape_value(&original, product)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	total, total_err := tape_sum_all(&tape, product)
	require_ok(t, total_err)
	testing.expect_value(t, tape_backward(&tape, total), Tensor_Error.None)
	left_grad, left_grad_err := tape_gradient(&tape, left_value)
	require_ok(t, left_grad_err)
	expect_f64_data(t, left_grad.data, []f64{3, 4})

	// A handle from another tape is rejected even when its id is in range.
	other, other_err := tape_init(f64)
	require_ok(t, other_err)
	defer tape_release(&other)
	other_value, other_value_err := tape_add_leaf_borrow(&other, &left)
	require_ok(t, other_value_err)
	_, err = tape_value(&tape, other_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add(&tape, left_value, other_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)

	// tape_reset invalidates old handles but keeps the tape usable.
	tape_reset(&tape)
	testing.expect_value(t, len(tape.nodes), 0)
	_, err = tape_value(&tape, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	testing.expect_value(t, tape_backward(&tape, total), Tensor_Error.Invalid_Tape)
	fresh, fresh_err := tape_add_leaf_borrow(&tape, &right)
	require_ok(t, fresh_err)
	testing.expect_value(t, fresh.id, 0)
	_, err = tape_value(&tape, fresh)
	require_ok(t, err)
	// The old handle has the same id as fresh but belongs to the earlier lifetime.
	_, err = tape_value(&tape, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
}

@(test)
test_tape_nil_tape_and_unsupported_records_leave_output_with_caller :: proc(t: ^testing.T) {
	tape, tape_err := tape_init(f64)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	left, err := from_slice(f64, []f64{1, 2}, []int{2})
	require_ok(t, err)
	defer release(&left)
	right: Tensor(f64)
	right, err = from_slice(f64, []f64{3, 4}, []int{2})
	require_ok(t, err)
	defer release(&right)
	left_value, left_err := tape_add_leaf_borrow(&tape, &left)
	require_ok(t, left_err)
	right_value, right_err := tape_add_leaf_borrow(&tape, &right)
	require_ok(t, right_err)
	output: Tensor(f64)
	output, err = add(left, right)
	require_ok(t, err)
	defer release(&output)

	_, err = tape_record_unary(&tape, &output, .Add, left_value)
	testing.expect_value(t, err, Tensor_Error.Unsupported_Operation)
	_, err = tape_record_binary(&tape, &output, .Tanh, left_value, right_value)
	testing.expect_value(t, err, Tensor_Error.Unsupported_Operation)
	testing.expect_value(t, len(tape.nodes), 2)

	_, err = tape_record_binary((^Tape(f64))(nil), &output, .Add, left_value, right_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_record_unary((^Tape(f64))(nil), &output, .Tanh, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_record_unary((^Tape(f64))(nil), &output, .Tanh, Tape_Value(f64){})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_record_sum_all((^Tape(f64))(nil), &output, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add_leaf_view((^Tape(f64))(nil), as_view(left))
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add_leaf_copy((^Tape(f64))(nil), as_view(left))
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add_leaf_borrow((^Tape(f64))(nil), &left)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add((^Tape(f64))(nil), left_value, right_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	testing.expect_value(t, len(output.data), 2)
	testing.expect_value(t, output.data[1], 6)
}

@(test)
test_tape_broadcast_gradients_through_strided_parent :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	require_ok(t, err)
	defer release(&storage)
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	// transposed is [[1, 3, 5], [2, 4, 6]] with strides {1, 2}.
	transposed, view_err := transpose_view(as_view(storage), shape_buffer[:], strides_buffer[:])
	require_ok(t, view_err)
	row: Tensor(f64)
	row, err = from_slice(f64, []f64{10, 20, 30}, []int{3})
	require_ok(t, err)
	defer release(&row)
	seed: Tensor(f64)
	seed, err = from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, err)
	defer release(&seed)

	tape, tape_err := tape_init(f64, 4)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	matrix_value, matrix_err := tape_add_leaf_view(&tape, transposed)
	require_ok(t, matrix_err)
	row_value, row_err := tape_add_leaf_borrow(&tape, &row)
	require_ok(t, row_err)

	product: Tensor(f64)
	product, err = mul_views(transposed, as_view(row))
	require_ok(t, err)
	product_value, product_err := tape_record_binary(&tape, &product, .Mul, matrix_value, row_value)
	require_ok(t, product_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, product_value, seed), Tensor_Error.None)
	matrix_grad, matrix_grad_err := tape_gradient(&tape, matrix_value)
	require_ok(t, matrix_grad_err)
	expect_f64_data(t, matrix_grad.data, []f64{10, 40, 90, 40, 100, 180})
	row_grad, row_grad_err := tape_gradient(&tape, row_value)
	require_ok(t, row_grad_err)
	expect_f64_data(t, row_grad.data, []f64{9, 26, 51})

	tape_zero_grad(&tape)
	expect_f64_data(t, matrix_grad.data, []f64{0, 0, 0, 0, 0, 0})
	quotient: Tensor(f64)
	quotient, err = div_views(transposed, as_view(row))
	require_ok(t, err)
	quotient_value, quotient_err := tape_record_binary(&tape, &quotient, .Div, matrix_value, row_value)
	require_ok(t, quotient_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, quotient_value, seed), Tensor_Error.None)
	expect_f64_data(t, matrix_grad.data, []f64{0.1, 0.1, 0.1, 0.4, 0.25, 0.2})
	expect_f64_data(t, row_grad.data, []f64{-9.0/100, -26.0/400, -51.0/900})
}

@(test)
test_tape_binary_backward_broadcasts_scalars_and_missing_axes :: proc(t: ^testing.T) {
	scalar, err := from_slice(f64, []f64{3}, []int{})
	require_ok(t, err)
	defer release(&scalar)
	grid: Tensor(f64)
	grid, err = from_slice(f64, []f64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{2, 2, 3})
	require_ok(t, err)
	defer release(&grid)
	// column has shape [2, 1], so it is missing the leading axis of grid and is
	// broadcast along the last one.
	column: Tensor(f64)
	column, err = from_slice(f64, []f64{2, 3}, []int{2, 1})
	require_ok(t, err)
	defer release(&column)

	tape, tape_err := tape_init(f64, 8)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	scalar_value, scalar_err := tape_add_leaf_borrow(&tape, &scalar)
	require_ok(t, scalar_err)
	grid_value, grid_err := tape_add_leaf_borrow(&tape, &grid)
	require_ok(t, grid_err)
	column_value, column_err := tape_add_leaf_borrow(&tape, &column)
	require_ok(t, column_err)

	scaled, scaled_err := tape_mul(&tape, scalar_value, grid_value)
	require_ok(t, scaled_err)
	scaled_total, scaled_total_err := tape_sum_all(&tape, scaled)
	require_ok(t, scaled_total_err)
	testing.expect_value(t, tape_backward(&tape, scaled_total), Tensor_Error.None)
	scalar_grad, scalar_grad_err := tape_gradient(&tape, scalar_value)
	require_ok(t, scalar_grad_err)
	expect_f64_data(t, scalar_grad.data, []f64{78})
	grid_grad, grid_grad_err := tape_gradient(&tape, grid_value)
	require_ok(t, grid_grad_err)
	expect_f64_data(t, grid_grad.data, []f64{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})

	// The column is the second operand here, so the b-side gradient index
	// must stay put along each row.
	tape_zero_grad(&tape)
	product, product_err := tape_mul(&tape, grid_value, column_value)
	require_ok(t, product_err)
	product_total, product_total_err := tape_sum_all(&tape, product)
	require_ok(t, product_total_err)
	testing.expect_value(t, tape_backward(&tape, product_total), Tensor_Error.None)
	column_grad, column_grad_err := tape_gradient(&tape, column_value)
	require_ok(t, column_grad_err)
	expect_f64_data(t, column_grad.data, []f64{30, 48})
	expect_f64_data(t, grid_grad.data, []f64{2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3})
}

@(test)
test_tape_binary_backward_through_reversed_view :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{1, 2, 3, 4, 5}, []int{1, 5})
	require_ok(t, err)
	defer release(&storage)
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	// reversed is [[5, 4, 3, 2, 1]] with strides {5, -1}.
	reversed, view_err := slice_view(as_view(storage), 1, -1, 5, -1, shape_buffer[:], strides_buffer[:])
	require_ok(t, view_err)
	column: Tensor(f64)
	column, err = from_slice(f64, []f64{10, 20, 30}, []int{3, 1})
	require_ok(t, err)
	defer release(&column)
	// A non-uniform seed makes both gradients depend on element order.
	seed: Tensor(f64)
	seed, err = from_slice(f64, []f64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, []int{3, 5})
	require_ok(t, err)
	defer release(&seed)

	tape, tape_err := tape_init(f64, 3)
	require_ok(t, tape_err)
	defer tape_release(&tape)
	reversed_value, reversed_err := tape_add_leaf_view(&tape, reversed)
	require_ok(t, reversed_err)
	column_value, column_err := tape_add_leaf_borrow(&tape, &column)
	require_ok(t, column_err)
	product, product_err := tape_mul(&tape, reversed_value, column_value)
	require_ok(t, product_err)
	testing.expect_value(t, tape_backward_with_seed(&tape, product, seed), Tensor_Error.None)

	reversed_grad, reversed_grad_err := tape_gradient(&tape, reversed_value)
	require_ok(t, reversed_grad_err)
	expect_f64_data(t, reversed_grad.data, []f64{460, 520, 580, 640, 700})
	column_grad, column_grad_err := tape_gradient(&tape, column_value)
	require_ok(t, column_grad_err)
	expect_f64_data(t, column_grad.data, []f64{35, 110, 185})
}

tape_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 8, allocator) or_return
	defer tape_release(&tape)
	weights := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2}, allocator) or_return
	defer release(&weights)
	weights_value := tape_add_leaf_borrow(&tape, &weights) or_return
	inputs_value := tape_add_leaf_copy(&tape, as_view(weights), false) or_return
	weights_view := tape_value(&tape, weights_value) or_return
	inputs_view := tape_value(&tape, inputs_value) or_return

	product := matmul_views(weights_view, inputs_view, allocator) or_return
	defer release(&product)
	product_value := tape_record_binary(&tape, &product, .Matmul, weights_value, inputs_value) or_return
	product_view := tape_value(&tape, product_value) or_return
	activated := tanh_views(product_view, allocator) or_return
	defer release(&activated)
	activated_value := tape_record_unary(&tape, &activated, .Tanh, product_value) or_return
	exponential_value := tape_exp(&tape, activated_value) or_return
	root_value := tape_sqrt(&tape, exponential_value) or_return
	logged_value := tape_log(&tape, root_value) or_return
	transposed_value := tape_transpose(&tape, logged_value) or_return
	// A transposed value is not contiguous, so this reshape copies.
	reshaped_value := tape_reshape(&tape, transposed_value, []int{1, 4}) or_return
	sliced_value := tape_slice(&tape, reshaped_value, 1, 0, 2, 2) or_return
	peaks_value := tape_max_axis(&tape, sliced_value, 0, true) or_return
	peak_products_value := tape_prod_axis(&tape, peaks_value, 1, true) or_return
	peak_products_view := tape_value(&tape, peak_products_value) or_return
	means := mean_axis_views(peak_products_view, 1, false, allocator) or_return
	defer release(&means)
	means_value := tape_record_mean_axis(&tape, &means, peak_products_value, 1) or_return
	mean_product_value := tape_prod_all(&tape, means_value) or_return
	mean_product_view := tape_value(&tape, mean_product_value) or_return
	lowest := min_all_views(mean_product_view, allocator) or_return
	defer release(&lowest)
	lowest_value := tape_record_min_all(&tape, &lowest, mean_product_value) or_return
	lowest_view := tape_value(&tape, lowest_value) or_return
	total := sum_all_views(lowest_view, allocator) or_return
	defer release(&total)
	total_value := tape_record_sum_all(&tape, &total, lowest_value) or_return
	return tape_backward(&tape, total_value)
}

@(test)
test_tape_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, tape_allocation_scenario)
}

// long_prod_root multiplies three products of x, each too long for the
// backward pass's stack scratch: along a contiguous axis, over a strided view,
// and over the contiguous whole.
long_prod_root :: proc(tape: ^Tape(f64), x: Tape_Value(f64)) -> (root: Tape_Value(f64), err: Tensor_Error) {
	rows := tape_prod_axis(tape, x, 1) or_return
	row_product := tape_prod_all(tape, rows) or_return
	transposed := tape_transpose(tape, x) or_return
	strided_product := tape_prod_all(tape, transposed) or_return
	product := tape_prod_all(tape, x) or_return
	partial := tape_mul(tape, row_product, strided_product) or_return
	return tape_mul(tape, partial, product)
}

long_prod_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 8, allocator) or_return
	defer tape_release(&tape)
	x := new_tensor(f64, []int{2, _PROD_STACK_COUNT+1}, allocator) or_return
	defer release(&x)
	x_value := tape_add_leaf_borrow(&tape, &x) or_return
	root := long_prod_root(&tape, x_value) or_return
	return tape_backward(&tape, root)
}

@(test)
test_tape_long_prod_gradients_allocate_scratch :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, long_prod_allocation_scenario)

	count := 2*(_PROD_STACK_COUNT+1)
	values := make([]f64, count)
	defer delete(values)
	for &value in values {
		value = 1
	}
	values[0] = 2
	x, err := from_slice(f64, values, []int{2, _PROD_STACK_COUNT+1})
	require_ok(t, err)
	defer release(&x)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	x_value, x_err := tape_add_leaf_borrow(&tape, &x)
	require_ok(t, x_err)
	root, root_err := long_prod_root(&tape, x_value)
	require_ok(t, root_err)
	// Each of the three products is 2, so the root is 8. Through each one,
	// x[0] gets 1 * 4 and every other element 2 * 4.
	expected := make([]f64, count)
	defer delete(expected)
	for &value in expected {
		value = 24
	}
	expected[0] = 12
	expect_parent_gradient(t, &tape, root, []f64{1}, []int{}, x_value, expected)
}

// apply_tape_helper runs one tape_<op> helper. b is ignored by unary and
// reduction operations.
apply_tape_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (Tape_Value(f64), Tensor_Error) {
	#partial switch operation {
	case .Add:
		return tape_add(tape, a, b)
	case .Sub:
		return tape_sub(tape, a, b)
	case .Mul:
		return tape_mul(tape, a, b)
	case .Div:
		return tape_div(tape, a, b)
	case .Dot:
		return tape_dot(tape, a, b)
	case .Matmul:
		return tape_matmul(tape, a, b)
	case .Relu:
		return tape_relu(tape, a)
	case .Sigmoid:
		return tape_sigmoid(tape, a)
	case .Tanh:
		return tape_tanh(tape, a)
	case .Exp:
		return tape_exp(tape, a)
	case .Log:
		return tape_log(tape, a)
	case .Sqrt:
		return tape_sqrt(tape, a)
	case .Sum_All:
		return tape_sum_all(tape, a)
	case .Mean_All:
		return tape_mean_all(tape, a)
	case .Sum_Axis:
		return tape_sum_axis(tape, a, -1)
	case .Mean_Axis:
		return tape_mean_axis(tape, a, 0, true)
	case .Prod_All:
		return tape_prod_all(tape, a)
	case .Prod_Axis:
		return tape_prod_axis(tape, a, -1, true)
	case .Max_All:
		return tape_max_all(tape, a)
	case .Min_All:
		return tape_min_all(tape, a)
	case .Max_Axis:
		return tape_max_axis(tape, a, -1)
	case .Min_Axis:
		return tape_min_axis(tape, a, 0, true)
	case .Transpose:
		return tape_transpose(tape, a)
	case .Slice:
		return tape_slice(tape, a, -1, -1, 2, -2)
	case .Reshape:
		return tape_reshape(tape, a, []int{-1, 2})
	case .Im2col:
		return tape_im2col(tape, a, 2, 2)
	case .Max_Pool2D:
		return tape_max_pool2d(tape, a, 2, 2)
	}
	return {}, .Unsupported_Operation
}

Tape_Helper :: #type proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (Tape_Value(f64), Tensor_Error)

// helper_objective resets the tape, applies operation, and records
// sum_all(sin-weighted output) so every output element has a distinct weight.
helper_objective :: proc(tape: ^Tape(f64), a, b: ^Tensor(f64), operation: Tape_Operation, apply: Tape_Helper) -> (loss: Tape_Value(f64), a_value, b_value: Tape_Value(f64), err: Tensor_Error) {
	tape_reset(tape)
	a_value = tape_add_leaf_borrow(tape, a) or_return
	b_value = tape_add_leaf_borrow(tape, b) or_return
	output := apply(tape, a_value, b_value, operation) or_return
	output_view := tape_value(tape, output) or_return
	weights := new_tensor(f64, output_view.shape) or_return
	defer release(&weights)
	for &weight, i in weights.data {
		weight = 1 + math.sin(f64(i))
	}
	weight_value := tape_add_leaf_copy(tape, as_view(weights), false) or_return
	weighted := tape_mul(tape, output, weight_value) or_return
	loss = tape_sum_all(tape, weighted) or_return
	return loss, a_value, b_value, .None
}

// check_tape_helper compares the gradients of helper_objective with central
// finite differences. apply builds the graph; operation names it in failures.
check_tape_helper :: proc(t: ^testing.T, operation: Tape_Operation, a_values: []f64, a_shape: []int, b_values: []f64, b_shape: []int, apply: Tape_Helper = apply_tape_helper) {
	a, err := from_slice(f64, a_values, a_shape)
	require_ok(t, err)
	defer release(&a)
	b: Tensor(f64)
	b, err = from_slice(f64, b_values, b_shape)
	require_ok(t, err)
	defer release(&b)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)

	loss, a_value, b_value, objective_err := helper_objective(&tape, &a, &b, operation, apply)
	testing.expectf(t, objective_err == .None, "%v: %v", operation, objective_err)
	if objective_err != .None {
		return
	}
	testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
	a_grad, a_grad_err := tape_gradient(&tape, a_value)
	require_ok(t, a_grad_err)
	b_grad, b_grad_err := tape_gradient(&tape, b_value)
	require_ok(t, b_grad_err)
	analytic_a := make([]f64, len(a.data))
	defer delete(analytic_a)
	analytic_b := make([]f64, len(b.data))
	defer delete(analytic_b)
	copy(analytic_a, a_grad.data)
	copy(analytic_b, b_grad.data)

	epsilon := 1e-6
	inputs := [2]^Tensor(f64){&a, &b}
	analytic := [2][]f64{analytic_a, analytic_b}
	for input, which in inputs {
		for i in 0..<len(input.data) {
			original := input.data[i]
			input.data[i] = original + epsilon
			plus_loss, _, _, plus_err := helper_objective(&tape, &a, &b, operation, apply)
			input.data[i] = original
			require_ok(t, plus_err)
			plus_view, plus_view_err := tape_value(&tape, plus_loss)
			require_ok(t, plus_view_err)
			plus := plus_view.data[0]
			input.data[i] = original - epsilon
			minus_loss, _, _, minus_err := helper_objective(&tape, &a, &b, operation, apply)
			input.data[i] = original
			require_ok(t, minus_err)
			minus_view, minus_view_err := tape_value(&tape, minus_loss)
			require_ok(t, minus_view_err)
			minus := minus_view.data[0]
			numeric := (plus - minus) / (2*epsilon)
			testing.expectf(t, abs(numeric - analytic[which][i]) < 1e-6, "%v input %d[%d]: analytic %v, numeric %v", operation, which, i, analytic[which][i], numeric)
		}
	}
}

@(test)
test_every_tape_helper_matches_finite_differences :: proc(t: ^testing.T) {
	matrix_values := []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9}
	row_values := []f64{0.7, -0.5, 1.3}
	for operation in ([]Tape_Operation{.Add, .Sub, .Mul, .Div}) {
		check_tape_helper(t, operation, matrix_values, []int{2, 3}, row_values, []int{3})
	}
	check_tape_helper(t, .Dot, matrix_values, []int{2, 3}, row_values, []int{3})
	// matmul promotes a vector left operand to a row, as in NumPy.
	check_tape_helper(t, .Matmul, row_values, []int{3}, matrix_values, []int{3, 2})
	// matrix_values has no ties, so max and min are differentiable there.
	for operation in ([]Tape_Operation{.Relu, .Sigmoid, .Tanh, .Sum_All, .Mean_All, .Sum_Axis, .Mean_Axis, .Prod_All, .Prod_Axis, .Max_All, .Min_All, .Max_Axis, .Min_Axis}) {
		check_tape_helper(t, operation, matrix_values, []int{2, 3}, row_values, []int{3})
	}
	// log and sqrt are only differentiable for positive inputs.
	positive_values := []f64{0.3, 1.2, 0.8, 1.5, 0.4, 2.9}
	for operation in ([]Tape_Operation{.Exp, .Log, .Sqrt}) {
		check_tape_helper(t, operation, positive_values, []int{2, 3}, row_values, []int{3})
	}
	for operation in ([]Tape_Operation{.Transpose, .Slice, .Reshape}) {
		check_tape_helper(t, operation, matrix_values, []int{2, 3}, row_values, []int{3})
	}
}

@(test)
test_tape_views_chain_into_other_operations :: proc(t: ^testing.T) {
	// a is [3,2], so transpose(a) @ b is [2,3] @ [3,2].
	transpose_then_matmul :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), _: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
		transposed := tape_transpose(tape, a) or_return
		return tape_matmul(tape, transposed, b)
	}
	check_tape_helper(t, .Transpose, []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9}, []int{3, 2}, []f64{0.7, -0.5, 1.3, 0.2, -0.9, 1.1}, []int{3, 2}, transpose_then_matmul)

	// Rows 2 and 0 of a [3,4] matrix, then a column sum: row 1 gets no gradient.
	slice_then_sum :: proc(tape: ^Tape(f64), a, _: Tape_Value(f64), _: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
		rows := tape_slice(tape, a, 0, 2, 2, -2) or_return
		return tape_sum_axis(tape, rows, 0)
	}
	check_tape_helper(t, .Slice, []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9, 0.7, -0.5, 1.3, -0.2, 1.1, 0.6}, []int{3, 4}, []f64{0}, []int{1}, slice_then_sum)

	// A transposed value is not contiguous, so tape_reshape copies it.
	transpose_then_reshape :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), _: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
		transposed := tape_transpose(tape, a) or_return
		flat := tape_reshape(tape, transposed, []int{6}) or_return
		return tape_mul(tape, flat, b)
	}
	check_tape_helper(t, .Reshape, []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9}, []int{2, 3}, []f64{0.7, -0.5, 1.3, 0.2, -0.9, 1.1}, []int{6}, transpose_then_reshape)
}

@(test)
test_tape_views_borrow_and_validate :: proc(t: ^testing.T) {
	x, err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	require_ok(t, err)
	defer release(&x)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	x_value, x_err := tape_add_leaf_borrow(&tape, &x)
	require_ok(t, x_err)

	transposed, transposed_err := tape_transpose(&tape, x_value)
	require_ok(t, transposed_err)
	transposed_view, transposed_view_err := tape_value(&tape, transposed)
	require_ok(t, transposed_view_err)
	testing.expectf(t, raw_data(transposed_view.data) == raw_data(x.data), "tape_transpose should borrow x's storage")
	contiguous, copy_err := copy_view(transposed_view)
	require_ok(t, copy_err)
	defer release(&contiguous)
	expect_f64_data(t, contiguous.data, []f64{1, 4, 2, 5, 3, 6})

	// Negative start and step count from the end: columns 2 and 0.
	sliced, sliced_err := tape_slice(&tape, x_value, 1, -1, 2, -2)
	require_ok(t, sliced_err)
	sliced_view, sliced_view_err := tape_value(&tape, sliced)
	require_ok(t, sliced_view_err)
	testing.expectf(t, raw_data(sliced_view.data) == raw_data(x.data), "tape_slice should borrow x's storage")
	sliced_copy, sliced_copy_err := copy_view(sliced_view)
	require_ok(t, sliced_copy_err)
	defer release(&sliced_copy)
	expect_f64_data(t, sliced_copy.data, []f64{3, 1, 6, 4})

	reshaped, reshaped_err := tape_reshape(&tape, x_value, []int{3, -1})
	require_ok(t, reshaped_err)
	reshaped_view, reshaped_view_err := tape_value(&tape, reshaped)
	require_ok(t, reshaped_view_err)
	testing.expectf(t, raw_data(reshaped_view.data) == raw_data(x.data), "tape_reshape of a contiguous value should borrow its storage")
	testing.expectf(t, _shapes_match(reshaped_view.shape, []int{3, 2}), "tape_reshape should infer -1 as 2, got shape %v", reshaped_view.shape)

	// The gradient of every view lands on the element it read.
	seed, seed_err := from_slice(f64, []f64{10, 20, 30, 40}, []int{2, 2})
	require_ok(t, seed_err)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, sliced, seed), Tensor_Error.None)
	x_grad, x_grad_err := tape_gradient(&tape, x_value)
	require_ok(t, x_grad_err)
	expect_f64_data(t, x_grad.data, []f64{20, 0, 10, 40, 0, 30})

	_, bad_slice := tape_slice(&tape, x_value, 1, 2, 2)
	testing.expect_value(t, bad_slice, Tensor_Error.Out_Of_Bounds)
	_, bad_negative_start := tape_slice(&tape, x_value, 1, -4, 1)
	testing.expect_value(t, bad_negative_start, Tensor_Error.Out_Of_Bounds)
	_, bad_axis := tape_slice(&tape, x_value, 2, 0, 1)
	testing.expect_value(t, bad_axis, Tensor_Error.Out_Of_Bounds)
	_, bad_reshape := tape_reshape(&tape, x_value, []int{4, -1})
	testing.expect_value(t, bad_reshape, Tensor_Error.Shape_Mismatch)
	_, stale := tape_transpose(&tape, Tape_Value(f64){})
	testing.expect_value(t, stale, Tensor_Error.Invalid_Tape)

	// An empty slice records an empty value and passes back no gradient.
	empty, empty_err := tape_slice(&tape, x_value, 0, 2, 0)
	require_ok(t, empty_err)
	empty_seed, empty_seed_err := new_tensor(f64, []int{0, 3})
	require_ok(t, empty_seed_err)
	defer release(&empty_seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, empty, empty_seed), Tensor_Error.None)
	expect_f64_data(t, x_grad.data, []f64{20, 0, 10, 40, 0, 30})
}

@(test)
test_tape_max_and_min_route_gradient_like_argmax :: proc(t: ^testing.T) {
	tied, err := from_slice(f64, []f64{3, 3}, []int{2})
	require_ok(t, err)
	defer release(&tied)
	// A NaN wins over every number, as in argmax and argmin.
	with_nan, nan_err := from_slice(f64, []f64{1, math.nan_f64(), 2, math.nan_f64()}, []int{2, 2})
	require_ok(t, nan_err)
	defer release(&with_nan)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)

	for operation in ([]Tape_Operation{.Max_All, .Min_All, .Max_Axis, .Min_Axis}) {
		tape_reset(&tape)
		x_value, x_err := tape_add_leaf_borrow(&tape, &tied)
		require_ok(t, x_err)
		y, y_err := apply_tape_helper(&tape, x_value, x_value, operation)
		testing.expectf(t, y_err == .None, "%v: %v", operation, y_err)
		loss, loss_err := tape_sum_all(&tape, y)
		require_ok(t, loss_err)
		testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
		x_grad, x_grad_err := tape_gradient(&tape, x_value)
		require_ok(t, x_grad_err)
		testing.expectf(t, x_grad.data[0] == 1 && x_grad.data[1] == 0, "%v: gradient of [3,3] is %v, want [1,0]", operation, x_grad.data)

		tape_reset(&tape)
		nan_value, nan_value_err := tape_add_leaf_borrow(&tape, &with_nan)
		require_ok(t, nan_value_err)
		y, y_err = apply_tape_helper(&tape, nan_value, nan_value, operation)
		testing.expectf(t, y_err == .None, "%v: %v", operation, y_err)
		loss, loss_err = tape_sum_all(&tape, y)
		require_ok(t, loss_err)
		testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
		nan_grad, nan_grad_err := tape_gradient(&tape, nan_value)
		require_ok(t, nan_grad_err)
		// Global reductions pick the first NaN; axis -1 picks it in each row,
		// and axis 0 picks it in the second column only.
		want := []f64{0, 1, 0, 0}
		#partial switch operation {
		case .Max_Axis: want = []f64{0, 1, 0, 1}
		case .Min_Axis: want = []f64{1, 1, 0, 0}
		}
		expect_f64_data(t, nan_grad.data, want)
	}
}

@(test)
test_tape_max_and_min_reject_empty_reductions :: proc(t: ^testing.T) {
	empty, err := new_tensor(f64, []int{2, 0})
	require_ok(t, err)
	defer release(&empty)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	x_value, x_err := tape_add_leaf_borrow(&tape, &empty)
	require_ok(t, x_err)

	_, max_err := tape_max_all(&tape, x_value)
	testing.expect_value(t, max_err, Tensor_Error.Empty_Reduction)
	_, min_err := tape_min_axis(&tape, x_value, 1)
	testing.expect_value(t, min_err, Tensor_Error.Empty_Reduction)
	// Reducing the non-empty axis leaves an empty result, which is fine.
	columns, columns_err := tape_max_axis(&tape, x_value, 0)
	require_ok(t, columns_err)
	empty_seed, empty_seed_err := new_tensor(f64, []int{0})
	require_ok(t, empty_seed_err)
	defer release(&empty_seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, columns, empty_seed), Tensor_Error.None)
	empty_grad, empty_grad_err := tape_gradient(&tape, x_value)
	require_ok(t, empty_grad_err)
	testing.expect_value(t, len(empty_grad.data), 0)

	// The record procedures check too, and leave the output with the caller.
	manual, manual_err := new_tensor(f64, []int{})
	require_ok(t, manual_err)
	defer release(&manual)
	_, record_err := tape_record_min_all(&tape, &manual, x_value)
	testing.expect_value(t, record_err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, len(manual.data), 1)
	manual_rows, manual_rows_err := new_tensor(f64, []int{2})
	require_ok(t, manual_rows_err)
	defer release(&manual_rows)
	_, axis_record_err := tape_record_max_axis(&tape, &manual_rows, x_value, 1)
	testing.expect_value(t, axis_record_err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, len(manual_rows.data), 2)
}

@(test)
test_tape_prod_gradients_handle_zeros_and_empty_products :: proc(t: ^testing.T) {
	// Each row has one zero, two zeros, or none. Only a single zero leaves a
	// nonzero gradient, at the zero itself.
	rows, err := from_slice(f64, []f64{2, 0, 3, 0, 5, 0, 1, 4, 0.5}, []int{3, 3})
	require_ok(t, err)
	defer release(&rows)
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	require_ok(t, err)
	defer release(&empty)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)

	x_value, x_err := tape_add_leaf_borrow(&tape, &rows)
	require_ok(t, x_err)
	row_products, row_err := tape_prod_axis(&tape, x_value, 1)
	require_ok(t, row_err)
	row_view, row_view_err := tape_value(&tape, row_products)
	require_ok(t, row_view_err)
	expect_f64_data(t, row_view.data, []f64{0, 0, 2})
	expect_parent_gradient(t, &tape, row_products, []f64{1, 1, 1}, []int{3}, x_value, []f64{0, 6, 0, 0, 0, 0, 2, 0.5, 4})

	// Down the columns, the reduced axis and its gradient both step by 3.
	column_products, column_err := tape_prod_axis(&tape, x_value, 0)
	require_ok(t, column_err)
	column_view, column_view_err := tape_value(&tape, column_products)
	require_ok(t, column_view_err)
	expect_f64_data(t, column_view.data, []f64{0, 0, 0})
	expect_parent_gradient(t, &tape, column_products, []f64{1, 1, 1}, []int{3}, x_value, []f64{0, 20, 0, 2, 0, 1.5, 0, 0, 0})

	// The whole matrix holds three zeros, so every gradient is zero.
	total, total_err := tape_prod_all(&tape, x_value)
	require_ok(t, total_err)
	expect_parent_gradient(t, &tape, total, []f64{1}, []int{}, x_value, []f64{0, 0, 0, 0, 0, 0, 0, 0, 0})
	first_row, slice_err := tape_slice(&tape, x_value, 0, 0, 1)
	require_ok(t, slice_err)
	first_total, first_total_err := tape_prod_all(&tape, first_row)
	require_ok(t, first_total_err)
	expect_parent_gradient(t, &tape, first_total, []f64{2}, []int{}, x_value, []f64{0, 12, 0, 0, 0, 0, 0, 0, 0})

	// A product of one element passes the upstream gradient straight through.
	single, single_err := tape_slice(&tape, x_value, 1, 1, 1)
	require_ok(t, single_err)
	single_products, single_products_err := tape_prod_axis(&tape, single, 1)
	require_ok(t, single_products_err)
	single_view, single_view_err := tape_value(&tape, single_products)
	require_ok(t, single_view_err)
	expect_f64_data(t, single_view.data, []f64{0, 5, 4})
	expect_parent_gradient(t, &tape, single_products, []f64{3, 4, 7}, []int{3}, x_value, []f64{0, 3, 0, 0, 4, 0, 0, 7, 0})

	// Empty products are 1 and send nothing back.
	empty_value, empty_err := tape_add_leaf_borrow(&tape, &empty)
	require_ok(t, empty_err)
	empty_total, empty_total_err := tape_prod_all(&tape, empty_value)
	require_ok(t, empty_total_err)
	empty_total_view, empty_total_view_err := tape_value(&tape, empty_total)
	require_ok(t, empty_total_view_err)
	expect_f64_data(t, empty_total_view.data, []f64{1})
	testing.expect_value(t, tape_backward(&tape, empty_total), Tensor_Error.None)
	empty_rows, empty_rows_err := tape_prod_axis(&tape, empty_value, 1)
	require_ok(t, empty_rows_err)
	empty_rows_view, empty_rows_view_err := tape_value(&tape, empty_rows)
	require_ok(t, empty_rows_view_err)
	expect_f64_data(t, empty_rows_view.data, []f64{1, 1})
	expect_parent_gradient(t, &tape, empty_rows, []f64{1, 1}, []int{2}, empty_value, []f64{})
}

@(test)
test_tape_max_and_min_axis_gradients_through_permuted_view :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{0.3, -1.2, 0.8, 1.5, -0.4, 0.9, 0.7, -0.5, 1.3, -0.2, 1.1, 0.6}, []int{2, 2, 3})
	require_ok(t, err)
	defer release(&storage)
	shape_buffer, strides_buffer := [3]int{}, [3]int{}
	// permuted has shape {3, 2, 2} and strides {1, 6, 3}, so reducing the
	// middle axis walks a non-contiguous outer and inner index at once.
	permuted, view_err := permute_view(as_view(storage), []int{2, 0, 1}, shape_buffer[:], strides_buffer[:])
	require_ok(t, view_err)
	contiguous: Tensor(f64)
	contiguous, err = copy_view(permuted)
	require_ok(t, err)
	defer release(&contiguous)
	seed: Tensor(f64)
	seed, err = new_tensor(f64, []int{3, 2})
	require_ok(t, err)
	defer release(&seed)
	for &value, i in seed.data {
		value = f64(i + 1)
	}

	extreme_middle_axis :: proc(tape: ^Tape(f64), x: Tape_Value(f64), operation: Tape_Operation) -> (Tape_Value(f64), Tensor_Error) {
		if operation == .Max_Axis {
			return tape_max_axis(tape, x, 1)
		}
		return tape_min_axis(tape, x, 1)
	}
	for operation in ([]Tape_Operation{.Max_Axis, .Min_Axis}) {
		view_tape, view_tape_err := tape_init(f64)
		require_ok(t, view_tape_err)
		defer tape_release(&view_tape)
		view_x, view_x_err := tape_add_leaf_view(&view_tape, permuted)
		require_ok(t, view_x_err)
		view_y, view_y_err := extreme_middle_axis(&view_tape, view_x, operation)
		testing.expectf(t, view_y_err == .None, "%v: %v", operation, view_y_err)
		testing.expect_value(t, tape_backward_with_seed(&view_tape, view_y, seed), Tensor_Error.None)

		copy_tape, copy_tape_err := tape_init(f64)
		require_ok(t, copy_tape_err)
		defer tape_release(&copy_tape)
		copy_x, copy_x_err := tape_add_leaf_borrow(&copy_tape, &contiguous)
		require_ok(t, copy_x_err)
		copy_y, copy_y_err := extreme_middle_axis(&copy_tape, copy_x, operation)
		testing.expectf(t, copy_y_err == .None, "%v: %v", operation, copy_y_err)
		testing.expect_value(t, tape_backward_with_seed(&copy_tape, copy_y, seed), Tensor_Error.None)

		view_grad, view_grad_err := tape_gradient(&view_tape, view_x)
		require_ok(t, view_grad_err)
		copy_grad, copy_grad_err := tape_gradient(&copy_tape, copy_x)
		require_ok(t, copy_grad_err)
		expect_f64_data(t, view_grad.data, copy_grad.data)
	}
}

@(test)
test_tape_log_and_sqrt_of_zero_give_infinite_gradients :: proc(t: ^testing.T) {
	x, err := from_slice(f64, []f64{0, 4}, []int{2})
	require_ok(t, err)
	defer release(&x)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)

	// d/dx log x = 1/x and d/dx sqrt x = 1/(2 sqrt x) are both 1/4 at x = 4.
	for operation in ([]Tape_Operation{.Log, .Sqrt}) {
		tape_reset(&tape)
		x_value, x_err := tape_add_leaf_borrow(&tape, &x)
		require_ok(t, x_err)
		y, y_err := apply_tape_helper(&tape, x_value, x_value, operation)
		testing.expectf(t, y_err == .None, "%v: %v", operation, y_err)
		loss, loss_err := tape_sum_all(&tape, y)
		require_ok(t, loss_err)
		testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
		x_grad, x_grad_err := tape_gradient(&tape, x_value)
		require_ok(t, x_grad_err)
		testing.expectf(t, math.is_inf(x_grad.data[0], 1), "%v: gradient at 0 is %v, want +Inf", operation, x_grad.data[0])
		testing.expectf(t, x_grad.data[1] == 0.25, "%v: gradient at 4 is %v, want 0.25", operation, x_grad.data[1])
	}
	// log(0) = -Inf is recorded as the forward value, not rejected.
	tape_reset(&tape)
	x_value, x_value_err := tape_add_leaf_borrow(&tape, &x)
	require_ok(t, x_value_err)
	logged, logged_err := tape_log(&tape, x_value)
	require_ok(t, logged_err)
	logged_view, logged_view_err := tape_value(&tape, logged)
	require_ok(t, logged_view_err)
	testing.expect_value(t, logged_view.data[0], math.inf_f64(-1))
}

@(test)
test_tape_backward_visits_only_reachable_nodes :: proc(t: ^testing.T) {
	x, err := from_slice(f64, []f64{2}, []int{1})
	require_ok(t, err)
	defer release(&x)
	infinity: Tensor(f64)
	infinity, err = from_slice(f64, []f64{math.inf_f64(1)}, []int{1})
	require_ok(t, err)
	defer release(&infinity)
	zero: Tensor(f64)
	zero, err = from_slice(f64, []f64{0}, []int{1})
	require_ok(t, err)
	defer release(&zero)

	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	x_value, x_value_err := tape_add_leaf_borrow(&tape, &x)
	require_ok(t, x_value_err)
	infinity_value, infinity_value_err := tape_add_leaf_borrow(&tape, &infinity, false)
	require_ok(t, infinity_value_err)
	zero_value, zero_value_err := tape_add_leaf_borrow(&tape, &zero, false)
	require_ok(t, zero_value_err)

	// An unrelated branch whose local derivative is infinite does not touch x.
	scaled, scaled_err := tape_mul(&tape, x_value, infinity_value)
	require_ok(t, scaled_err)
	direct, direct_err := tape_sum_all(&tape, x_value)
	require_ok(t, direct_err)
	testing.expect_value(t, tape_backward(&tape, direct), Tensor_Error.None)
	x_grad, x_grad_err := tape_gradient(&tape, x_value)
	require_ok(t, x_grad_err)
	expect_f64_data(t, x_grad.data, []f64{1})

	// On a reachable path, 0 * inf gives NaN, as in other autodiff libraries.
	tape_zero_grad(&tape)
	silenced, silenced_err := tape_mul(&tape, scaled, zero_value)
	require_ok(t, silenced_err)
	loss, loss_err := tape_sum_all(&tape, silenced)
	require_ok(t, loss_err)
	testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
	testing.expectf(t, math.is_nan(x_grad.data[0]), "got %v, want NaN", x_grad.data[0])
}

@(test)
test_tape_axis_reduction_backward_expands_every_axis :: proc(t: ^testing.T) {
	// Every axis of a [2, 3, 4] input, with a distinct upstream value per
	// output element, so a gradient routed to the wrong place shows.
	input_values: [24]f64
	for &value, i in input_values {
		value = f64(i)
	}
	shape := []int{2, 3, 4}
	for axis in 0..<3 {
		for keep_dims in ([]bool{false, true}) {
			for mean in ([]bool{false, true}) {
				input, input_err := from_slice(f64, input_values[:], shape)
				require_ok(t, input_err)
				defer release(&input)
				tape, tape_err := tape_init(f64, 4)
				require_ok(t, tape_err)
				defer tape_release(&tape)
				parent, parent_err := tape_add_leaf_borrow(&tape, &input)
				require_ok(t, parent_err)
				reduced: Tape_Value(f64)
				reduce_err: Tensor_Error
				if mean {
					reduced, reduce_err = tape_mean_axis(&tape, parent, axis, keep_dims)
				} else {
					reduced, reduce_err = tape_sum_axis(&tape, parent, axis, keep_dims)
				}
				require_ok(t, reduce_err)
				reduced_value, value_err := tape_value(&tape, reduced)
				require_ok(t, value_err)
				seed, seed_err := new_tensor(f64, reduced_value.shape)
				require_ok(t, seed_err)
				defer release(&seed)
				for &value, i in seed.data {
					value = f64(i + 1)
				}
				testing.expect_value(t, tape_backward_with_seed(&tape, reduced, seed), .None)
				gradient, gradient_err := tape_gradient(&tape, parent)
				require_ok(t, gradient_err)

				for i in 0..<2 do for j in 0..<3 do for k in 0..<4 {
					index := [3]int{i, j, k}
					// The upstream index drops the reduced axis; keep_dims
					// leaves it at length one, which is the same linear index.
					upstream := 0
					for a in 0..<3 {
						if a != axis {
							upstream = upstream*shape[a] + index[a]
						}
					}
					want := seed.data[upstream]
					if mean {
						want /= f64(shape[axis])
					}
					got := gradient.data[gradient.offset + (i*3 + j)*4 + k]
					testing.expectf(t, got == want, "axis %d keep_dims %v mean %v at %v: got %v, want %v", axis, keep_dims, mean, index, got, want)
				}
			}
		}
	}
}

// Both operands broadcast along interior axes, as when nearest-neighbour
// upsampling multiplies [n, h, 1, w, 1, c] by [f, 1, f, 1]: each gradient
// element sums the upstream values of every output it reached, in row-major
// order.
@(test)
test_tape_mul_backward_sums_interior_broadcasts :: proc(t: ^testing.T) {
	N, H, W, C, F :: 2, 3, 4, 5, 2
	x, err := new_tensor(f64, []int{N, H, 1, W, 1, C})
	require_ok(t, err)
	defer release(&x)
	for &value, i in x.data {
		value = math.sin(0.3*f64(i))
	}
	y: Tensor(f64)
	y, err = new_tensor(f64, []int{F, 1, F, 1})
	require_ok(t, err)
	defer release(&y)
	copy(y.data, []f64{0.5, -1.25, 2, 0.75})
	seed: Tensor(f64)
	seed, err = new_tensor(f64, []int{N, H, F, W, F, C})
	require_ok(t, err)
	defer release(&seed)
	for &value, i in seed.data {
		value = math.cos(0.7*f64(i))
	}

	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	xv := tape_add_leaf_borrow(&tape, &x) or_else panic("leaf")
	yv := tape_add_leaf_borrow(&tape, &y) or_else panic("leaf")
	product := tape_mul(&tape, xv, yv) or_else panic("mul")
	require_ok(t, tape_backward_with_seed(&tape, product, seed))

	expected_x := make([]f64, len(x.data))
	defer delete(expected_x)
	expected_y := make([]f64, len(y.data))
	defer delete(expected_y)
	next := 0
	for n in 0..<N do for h in 0..<H do for i in 0..<F do for w in 0..<W do for j in 0..<F do for c in 0..<C {
		x_index := ((n*H + h)*W + w)*C + c
		y_index := i*F + j
		expected_x[x_index] += seed.data[next] * y.data[y_index]
		expected_y[y_index] += seed.data[next] * x.data[x_index]
		next += 1
	}
	x_grad := tape_gradient(&tape, xv) or_else panic("gradient")
	y_grad := tape_gradient(&tape, yv) or_else panic("gradient")
	for value, i in x_grad.data {
		testing.expect_value(t, value, expected_x[i])
	}
	for value, i in y_grad.data {
		testing.expect_value(t, value, expected_y[i])
	}
}
