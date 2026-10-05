package tensor

import "core:math"
import "core:mem"
import "core:testing"

@(test)
test_tape_copy_leaf_snapshots_strided_source :: proc(t: ^testing.T) {
	owner, owner_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	testing.expect(t, owner_err == .None)
	defer release(&owner)
	shape_buffer := [1]int{}
	stride_buffer := [1]int{}
	reversed, view_err := slice_view(as_view(owner), 0, 2, 3, -1, shape_buffer[:], stride_buffer[:])
	testing.expect(t, view_err == .None)

	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	leaf, leaf_err := tape_add_leaf_copy(&tape, reversed)
	testing.expect(t, leaf_err == .None)
	testing.expect(t, tape.nodes[leaf.id].owns_value)
	owner.data[0], owner.data[1], owner.data[2] = 100, 200, 300

	leaf_view, value_err := tape_value(&tape, leaf)
	testing.expect(t, value_err == .None)
	testing.expect(t, leaf_view.data[0] == 3 && leaf_view.data[1] == 2 && leaf_view.data[2] == 1)
	squared, square_err := mul_views(leaf_view, leaf_view)
	testing.expect(t, square_err == .None)
	child, record_err := tape_record_binary(&tape, &squared, .Mul, leaf, leaf)
	testing.expect(t, record_err == .None)
	seed, seed_err := from_slice(f64, []f64{1, 1, 1}, []int{3})
	testing.expect(t, seed_err == .None)
	defer release(&seed)
	testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
	gradient, grad_err := tape_gradient(&tape, leaf)
	testing.expect(t, grad_err == .None)
	testing.expect(t, gradient.data[0] == 6 && gradient.data[1] == 4 && gradient.data[2] == 2)
}

@(test)
test_tape_rejects_invalid_views_and_operation_shapes :: proc(t: ^testing.T) {
	backing := []f32{1}
	invalid_view := Tensor_View(f32){data = backing, shape = []int{2}, strides = []int{1}}
	tape, tape_err := tape_init(f32, 3)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	_, invalid_leaf_err := tape_add_leaf_view(&tape, invalid_view)
	testing.expect(t, invalid_leaf_err == .Out_Of_Bounds)
	testing.expect(t, len(tape.nodes) == 0)

	left, left_err := from_slice(f32, []f32{1, 2}, []int{2})
	testing.expect(t, left_err == .None)
	defer release(&left)
	right, right_err := from_slice(f32, []f32{3, 4}, []int{2})
	testing.expect(t, right_err == .None)
	defer release(&right)
	left_value, left_value_err := tape_add_leaf_borrow(&tape, &left)
	testing.expect(t, left_value_err == .None)
	right_value, right_value_err := tape_add_leaf_borrow(&tape, &right)
	testing.expect(t, right_value_err == .None)

	wrong_output, output_err := new_tensor(f32, []int{1})
	testing.expect(t, output_err == .None)
	_, record_err := tape_record_binary(&tape, &wrong_output, .Add, left_value, right_value)
	testing.expect(t, record_err == .Shape_Mismatch)
	testing.expect(t, len(wrong_output.data) == 1 && len(tape.nodes) == 2)
	release(&wrong_output)

	unary_output, unary_output_err := new_tensor(f32, []int{2})
	testing.expect(t, unary_output_err == .None)
	_, unary_record_err := tape_record_unary(&tape, &unary_output, .Add, left_value)
	testing.expect(t, unary_record_err == .Unsupported_Operation)
	testing.expect(t, len(unary_output.data) == 2 && len(tape.nodes) == 2)
	release(&unary_output)
}

@(test)
test_tape_records_reductions_and_validates_ownership :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, input_err == .None)
	defer release(&input)

	tape, tape_err := tape_init(f64, 5)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, parent_err == .None)

	global_sum, sum_err := sum_all(input)
	testing.expect(t, sum_err == .None && global_sum.data[0] == 21)
	global_sum_value, sum_record_err := tape_record_sum_all(&tape, &global_sum, parent)
	testing.expect(t, sum_record_err == .None && len(global_sum.data) == 0)
	testing.expect(t, tape.nodes[global_sum_value.id].operation == .Sum_All)

	global_mean, mean_err := mean_all(input)
	testing.expect(t, mean_err == .None && global_mean.data[0] == 3.5)
	global_mean_value, mean_record_err := tape_record_mean_all(&tape, &global_mean, parent)
	testing.expect(t, mean_record_err == .None && len(global_mean.data) == 0)
	testing.expect(t, tape.nodes[global_mean_value.id].operation == .Mean_All)

	axis_sum, axis_sum_err := sum_axis(input, 1)
	testing.expect(t, axis_sum_err == .None && axis_sum.data[0] == 6 && axis_sum.data[1] == 15)
	axis_sum_value, axis_sum_record_err := tape_record_sum_axis(&tape, &axis_sum, parent, 1)
	testing.expect(t, axis_sum_record_err == .None && len(axis_sum.data) == 0)
	testing.expect(t, tape.nodes[axis_sum_value.id].operation == .Sum_Axis)
	testing.expect(t, tape.nodes[axis_sum_value.id].axis == 1 && !tape.nodes[axis_sum_value.id].keep_dims)

	axis_mean, axis_mean_err := mean_axis(input, 1, true)
	testing.expect(t, axis_mean_err == .None && axis_mean.data[0] == 2 && axis_mean.data[1] == 5)
	axis_mean_value, axis_mean_record_err := tape_record_mean_axis(&tape, &axis_mean, parent, 1, true)
	testing.expect(t, axis_mean_record_err == .None && len(axis_mean.data) == 0)
	testing.expect(t, tape.nodes[axis_mean_value.id].operation == .Mean_Axis)
	testing.expect(t, tape.nodes[axis_mean_value.id].axis == 1 && tape.nodes[axis_mean_value.id].keep_dims)
	testing.expect(t, len(tape.nodes) == 5)

	invalid_axis_output, invalid_axis_err := sum_axis(input, 0)
	testing.expect(t, invalid_axis_err == .None)
	_, record_axis_err := tape_record_sum_axis(&tape, &invalid_axis_output, parent, 2)
	testing.expect(t, record_axis_err == .Out_Of_Bounds)
	testing.expect(t, len(tape.nodes) == 5 && len(invalid_axis_output.data) == 3)
	release(&invalid_axis_output)

	wrong_shape, wrong_shape_err := new_tensor(f64, []int{1, 3})
	testing.expect(t, wrong_shape_err == .None)
	_, record_shape_err := tape_record_mean_axis(&tape, &wrong_shape, parent, 1)
	testing.expect(t, record_shape_err == .Shape_Mismatch)
	testing.expect(t, len(tape.nodes) == 5 && len(wrong_shape.data) == 3)
	release(&wrong_shape)

	foreign_tape, foreign_tape_err := tape_init(f64, 1)
	testing.expect(t, foreign_tape_err == .None)
	defer tape_release(&foreign_tape)
	foreign_parent, foreign_parent_err := tape_add_leaf_borrow(&foreign_tape, &input)
	testing.expect(t, foreign_parent_err == .None)
	wrong_tape_output, wrong_tape_output_err := sum_all(input)
	testing.expect(t, wrong_tape_output_err == .None)
	_, wrong_tape_record_err := tape_record_sum_all(&tape, &wrong_tape_output, foreign_parent)
	testing.expect(t, wrong_tape_record_err == .Invalid_Tape)
	testing.expect(t, len(tape.nodes) == 5 && len(wrong_tape_output.data) == 1)
	release(&wrong_tape_output)

	grown_tape, grown_tape_err := tape_init(f64, 1)
	testing.expect(t, grown_tape_err == .None)
	defer tape_release(&grown_tape)
	grown_parent, grown_parent_err := tape_add_leaf_borrow(&grown_tape, &input)
	testing.expect(t, grown_parent_err == .None)
	grown_output, grown_output_err := sum_all(input)
	testing.expect(t, grown_output_err == .None)
	// The reservation is only a hint; recording past it grows the tape.
	_, grown_record_err := tape_record_sum_all(&grown_tape, &grown_output, grown_parent)
	testing.expect(t, grown_record_err == .None)
	testing.expect(t, len(grown_tape.nodes) == 2 && len(grown_output.data) == 0)
}

@(test)
test_tape_sum_all_backward_broadcasts_and_accumulates :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, input_err == .None)
	defer release(&input)

	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, parent_err == .None)
	output, output_err := sum_all(input)
	testing.expect(t, output_err == .None && output.data[0] == 10)
	loss, record_err := tape_record_sum_all(&tape, &output, parent)
	testing.expect(t, record_err == .None)
	testing.expect(t, tape_backward(&tape, loss) == .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	testing.expect(t, gradient_err == .None)
	for value in gradient.data {
		testing.expect(t, value == 1)
	}
	testing.expect(t, tape_backward(&tape, loss) == .None)
	for value in gradient.data {
		testing.expect(t, value == 2)
	}

	scalar, scalar_err := from_slice(f64, []f64{7}, []int{})
	testing.expect(t, scalar_err == .None)
	defer release(&scalar)
	scalar_tape, scalar_tape_err := tape_init(f64, 2)
	testing.expect(t, scalar_tape_err == .None)
	defer tape_release(&scalar_tape)
	scalar_parent, scalar_parent_err := tape_add_leaf_borrow(&scalar_tape, &scalar)
	testing.expect(t, scalar_parent_err == .None)
	scalar_output, scalar_output_err := sum_all(scalar)
	testing.expect(t, scalar_output_err == .None && scalar_output.data[0] == 7)
	scalar_loss, scalar_record_err := tape_record_sum_all(&scalar_tape, &scalar_output, scalar_parent)
	testing.expect(t, scalar_record_err == .None)
	testing.expect(t, tape_backward(&scalar_tape, scalar_loss) == .None)
	scalar_gradient, scalar_gradient_err := tape_gradient(&scalar_tape, scalar_parent)
	testing.expect(t, scalar_gradient_err == .None && scalar_gradient.data[0] == 1)

	empty, empty_err := new_tensor(f64, []int{0, 2})
	testing.expect(t, empty_err == .None)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	testing.expect(t, empty_tape_err == .None)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	testing.expect(t, empty_parent_err == .None)
	empty_output, empty_output_err := sum_all(empty)
	testing.expect(t, empty_output_err == .None && empty_output.data[0] == 0)
	empty_loss, empty_record_err := tape_record_sum_all(&empty_tape, &empty_output, empty_parent)
	testing.expect(t, empty_record_err == .None)
	testing.expect(t, tape_backward(&empty_tape, empty_loss) == .None)
	empty_gradient, empty_gradient_err := tape_gradient(&empty_tape, empty_parent)
	testing.expect(t, empty_gradient_err == .None && len(empty_gradient.data) == 0)

	owner, owner_err := from_slice(f64, []f64{10, 20, 30, 40}, []int{4})
	testing.expect(t, owner_err == .None)
	defer release(&owner)
	shape_buffer := [1]int{}
	stride_buffer := [1]int{}
	reversed, view_err := slice_view(as_view(owner), 0, 3, 3, -1, shape_buffer[:], stride_buffer[:])
	testing.expect(t, view_err == .None)
	view_tape, view_tape_err := tape_init(f64, 2)
	testing.expect(t, view_tape_err == .None)
	defer tape_release(&view_tape)
	view_parent, view_parent_err := tape_add_leaf_view(&view_tape, reversed)
	testing.expect(t, view_parent_err == .None)
	view_output, view_output_err := sum_all_views(reversed)
	testing.expect(t, view_output_err == .None && view_output.data[0] == 90)
	view_loss, view_record_err := tape_record_sum_all(&view_tape, &view_output, view_parent)
	testing.expect(t, view_record_err == .None)
	testing.expect(t, tape_backward(&view_tape, view_loss) == .None)
	view_gradient, view_gradient_err := tape_gradient(&view_tape, view_parent)
	testing.expect(t, view_gradient_err == .None)
	testing.expect(t, view_gradient.data[0] == 1 && view_gradient.data[1] == 1 && view_gradient.data[2] == 1)
}

@(test)
test_tape_mean_all_backward_scales_and_accumulates :: proc(t: ^testing.T) {
	input, input_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, input_err == .None)
	defer release(&input)

	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, parent_err == .None)
	output, output_err := mean_all(input)
	testing.expect(t, output_err == .None && output.data[0] == 2.5)
	loss, record_err := tape_record_mean_all(&tape, &output, parent)
	testing.expect(t, record_err == .None)
	testing.expect(t, tape_backward(&tape, loss) == .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	testing.expect(t, gradient_err == .None)
	for value in gradient.data {
		testing.expect(t, value == 0.25)
	}
	testing.expect(t, tape_backward(&tape, loss) == .None)
	for value in gradient.data {
		testing.expect(t, value == 0.5)
	}

	scalar, scalar_err := from_slice(f64, []f64{7}, []int{})
	testing.expect(t, scalar_err == .None)
	defer release(&scalar)
	scalar_tape, scalar_tape_err := tape_init(f64, 2)
	testing.expect(t, scalar_tape_err == .None)
	defer tape_release(&scalar_tape)
	scalar_parent, scalar_parent_err := tape_add_leaf_borrow(&scalar_tape, &scalar)
	testing.expect(t, scalar_parent_err == .None)
	scalar_output, scalar_output_err := mean_all(scalar)
	testing.expect(t, scalar_output_err == .None && scalar_output.data[0] == 7)
	scalar_loss, scalar_record_err := tape_record_mean_all(&scalar_tape, &scalar_output, scalar_parent)
	testing.expect(t, scalar_record_err == .None)
	testing.expect(t, tape_backward(&scalar_tape, scalar_loss) == .None)
	scalar_gradient, scalar_gradient_err := tape_gradient(&scalar_tape, scalar_parent)
	testing.expect(t, scalar_gradient_err == .None && scalar_gradient.data[0] == 1)

	empty, empty_err := new_tensor(f64, []int{0, 2})
	testing.expect(t, empty_err == .None)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	testing.expect(t, empty_tape_err == .None)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	testing.expect(t, empty_parent_err == .None)
	_, failed_output_err := mean_all(empty)
	testing.expect(t, failed_output_err == .Empty_Reduction)
	testing.expect(t, len(empty_tape.nodes) == 1)

	manual_output, manual_output_err := new_tensor(f64, []int{})
	testing.expect(t, manual_output_err == .None)
	_, manual_record_err := tape_record_mean_all(&empty_tape, &manual_output, empty_parent)
	testing.expect(t, manual_record_err == .Empty_Reduction)
	testing.expect(t, len(empty_tape.nodes) == 1 && len(manual_output.data) == 1)
	release(&manual_output)
}

@(test)
test_tape_sum_axis_backward_expands_across_reduced_axis :: proc(t: ^testing.T) {
	// A one-dimensional reduction produces a scalar upstream gradient.
	vector, vector_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	testing.expect(t, vector_err == .None)
	defer release(&vector)
	vector_tape, vector_tape_err := tape_init(f64, 2)
	testing.expect(t, vector_tape_err == .None)
	defer tape_release(&vector_tape)
	vector_parent, vector_parent_err := tape_add_leaf_borrow(&vector_tape, &vector)
	testing.expect(t, vector_parent_err == .None)
	vector_output, vector_output_err := sum_axis(vector, 0)
	testing.expect(t, vector_output_err == .None && vector_output.data[0] == 6)
	vector_loss, vector_record_err := tape_record_sum_axis(&vector_tape, &vector_output, vector_parent, 0)
	testing.expect(t, vector_record_err == .None)
	vector_seed, vector_seed_err := from_slice(f64, []f64{2}, []int{})
	testing.expect(t, vector_seed_err == .None)
	defer release(&vector_seed)
	testing.expect(t, tape_backward_with_seed(&vector_tape, vector_loss, vector_seed) == .None)
	vector_gradient, vector_gradient_err := tape_gradient(&vector_tape, vector_parent)
	testing.expect(t, vector_gradient_err == .None)
	for value in vector_gradient.data {
		testing.expect(t, value == 2)
	}

	// Batched reduction with keep_dims=false and keep_dims=true.
	batch, batch_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, batch_err == .None)
	defer release(&batch)
	batch_tape, batch_tape_err := tape_init(f64, 3)
	testing.expect(t, batch_tape_err == .None)
	defer tape_release(&batch_tape)
	batch_parent, batch_parent_err := tape_add_leaf_borrow(&batch_tape, &batch)
	testing.expect(t, batch_parent_err == .None)
	row_sums, row_sums_err := sum_axis(batch, 1)
	testing.expect(t, row_sums_err == .None)
	row_loss, row_record_err := tape_record_sum_axis(&batch_tape, &row_sums, batch_parent, 1)
	testing.expect(t, row_record_err == .None)
	row_seed, row_seed_err := from_slice(f64, []f64{10, 20}, []int{2})
	testing.expect(t, row_seed_err == .None)
	defer release(&row_seed)
	testing.expect(t, tape_backward_with_seed(&batch_tape, row_loss, row_seed) == .None)
	batch_gradient, batch_gradient_err := tape_gradient(&batch_tape, batch_parent)
	testing.expect(t, batch_gradient_err == .None)
	testing.expect(t, batch_gradient.data[0] == 10 && batch_gradient.data[2] == 10)
	testing.expect(t, batch_gradient.data[3] == 20 && batch_gradient.data[5] == 20)

	column_sums, column_sums_err := sum_axis(batch, 0, true)
	testing.expect(t, column_sums_err == .None)
	column_loss, column_record_err := tape_record_sum_axis(&batch_tape, &column_sums, batch_parent, 0, true)
	testing.expect(t, column_record_err == .None)
	column_seed, column_seed_err := from_slice(f64, []f64{1, 2, 3}, []int{1, 3})
	testing.expect(t, column_seed_err == .None)
	defer release(&column_seed)
	testing.expect(t, tape_backward_with_seed(&batch_tape, column_loss, column_seed) == .None)
	testing.expect(t, batch_gradient.data[0] == 11 && batch_gradient.data[1] == 12 && batch_gradient.data[2] == 13)
	testing.expect(t, batch_gradient.data[3] == 21 && batch_gradient.data[4] == 22 && batch_gradient.data[5] == 23)

	// Higher-rank reduction maps upstream coordinates around a removed axis.
	higher, higher_err := from_slice(f64, []f64{0, 0, 0, 0, 0, 0, 0, 0}, []int{2, 2, 2})
	testing.expect(t, higher_err == .None)
	defer release(&higher)
	higher_tape, higher_tape_err := tape_init(f64, 2)
	testing.expect(t, higher_tape_err == .None)
	defer tape_release(&higher_tape)
	higher_parent, higher_parent_err := tape_add_leaf_borrow(&higher_tape, &higher)
	testing.expect(t, higher_parent_err == .None)
	higher_output, higher_output_err := sum_axis(higher, 1)
	testing.expect(t, higher_output_err == .None)
	higher_loss, higher_record_err := tape_record_sum_axis(&higher_tape, &higher_output, higher_parent, 1)
	testing.expect(t, higher_record_err == .None)
	higher_seed, higher_seed_err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, higher_seed_err == .None)
	defer release(&higher_seed)
	testing.expect(t, tape_backward_with_seed(&higher_tape, higher_loss, higher_seed) == .None)
	higher_gradient, higher_gradient_err := tape_gradient(&higher_tape, higher_parent)
	testing.expect(t, higher_gradient_err == .None)
	testing.expect(t, higher_gradient.data[0] == 1 && higher_gradient.data[1] == 2)
	testing.expect(t, higher_gradient.data[2] == 1 && higher_gradient.data[3] == 2)
	testing.expect(t, higher_gradient.data[4] == 3 && higher_gradient.data[5] == 4)
	testing.expect(t, higher_gradient.data[6] == 3 && higher_gradient.data[7] == 4)
}

@(test)
test_tape_sum_axis_backward_handles_strided_and_empty_views :: proc(t: ^testing.T) {
	owner, owner_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, owner_err == .None)
	defer release(&owner)
	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, transposed_err := transpose_view(as_view(owner), transposed_shape[:], transposed_strides[:])
	testing.expect(t, transposed_err == .None)
	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_view(&tape, transposed)
	testing.expect(t, parent_err == .None)
	output, output_err := sum_axis_views(transposed, 1)
	testing.expect(t, output_err == .None)
	loss, record_err := tape_record_sum_axis(&tape, &output, parent, 1)
	testing.expect(t, record_err == .None)
	seed, seed_err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	testing.expect(t, seed_err == .None)
	defer release(&seed)
	testing.expect(t, tape_backward_with_seed(&tape, loss, seed) == .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	testing.expect(t, gradient_err == .None)
	testing.expect(t, gradient.data[0] == 1 && gradient.data[1] == 1)
	testing.expect(t, gradient.data[2] == 2 && gradient.data[3] == 2)
	testing.expect(t, gradient.data[4] == 3 && gradient.data[5] == 3)

	empty, empty_err := new_tensor(f64, []int{2, 0, 3})
	testing.expect(t, empty_err == .None)
	defer release(&empty)
	empty_tape, empty_tape_err := tape_init(f64, 2)
	testing.expect(t, empty_tape_err == .None)
	defer tape_release(&empty_tape)
	empty_parent, empty_parent_err := tape_add_leaf_borrow(&empty_tape, &empty)
	testing.expect(t, empty_parent_err == .None)
	empty_output, empty_output_err := sum_axis(empty, 1)
	testing.expect(t, empty_output_err == .None && len(empty_output.data) == 6)
	for value in empty_output.data {
		testing.expect(t, value == 0)
	}
	empty_loss, empty_record_err := tape_record_sum_axis(&empty_tape, &empty_output, empty_parent, 1)
	testing.expect(t, empty_record_err == .None)
	empty_seed, empty_seed_err := new_tensor(f64, []int{2, 3})
	testing.expect(t, empty_seed_err == .None)
	defer release(&empty_seed)
	for i in 0..<len(empty_seed.data) {
		empty_seed.data[i] = f64(i+1)
	}
	testing.expect(t, tape_backward_with_seed(&empty_tape, empty_loss, empty_seed) == .None)
	empty_gradient, empty_gradient_err := tape_gradient(&empty_tape, empty_parent)
	testing.expect(t, empty_gradient_err == .None && len(empty_gradient.data) == 0)
}

@(test)
test_tape_mean_axis_backward_scales_and_handles_views :: proc(t: ^testing.T) {
	batch, batch_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, batch_err == .None)
	defer release(&batch)
	tape, tape_err := tape_init(f64, 3)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &batch)
	testing.expect(t, parent_err == .None)

	row_means, row_means_err := mean_axis(batch, 1)
	testing.expect(t, row_means_err == .None && row_means.data[0] == 2 && row_means.data[1] == 5)
	row_loss, row_record_err := tape_record_mean_axis(&tape, &row_means, parent, 1)
	testing.expect(t, row_record_err == .None)
	row_seed, row_seed_err := from_slice(f64, []f64{6, 9}, []int{2})
	testing.expect(t, row_seed_err == .None)
	defer release(&row_seed)
	testing.expect(t, tape_backward_with_seed(&tape, row_loss, row_seed) == .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	testing.expect(t, gradient_err == .None)
	testing.expect(t, gradient.data[0] == 2 && gradient.data[1] == 2 && gradient.data[2] == 2)
	testing.expect(t, gradient.data[3] == 3 && gradient.data[4] == 3 && gradient.data[5] == 3)

	// Keep the reduced axis and reduce a different dimension. Dividing by 2
	// maps each seeded column value equally to the two rows.
	column_means, column_means_err := mean_axis(batch, 0, true)
	testing.expect(t, column_means_err == .None)
	column_loss, column_record_err := tape_record_mean_axis(&tape, &column_means, parent, 0, true)
	testing.expect(t, column_record_err == .None)
	column_seed, column_seed_err := from_slice(f64, []f64{4, 8, 12}, []int{1, 3})
	testing.expect(t, column_seed_err == .None)
	defer release(&column_seed)
	testing.expect(t, tape_backward_with_seed(&tape, column_loss, column_seed) == .None)
	testing.expect(t, gradient.data[0] == 4 && gradient.data[1] == 6 && gradient.data[2] == 8)
	testing.expect(t, gradient.data[3] == 5 && gradient.data[4] == 7 && gradient.data[5] == 9)

	// A transposed view retains logical gradient order and the correct divisor.
	owner, owner_err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, owner_err == .None)
	defer release(&owner)
	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, transposed_err := transpose_view(as_view(owner), transposed_shape[:], transposed_strides[:])
	testing.expect(t, transposed_err == .None)
	view_tape, view_tape_err := tape_init(f64, 2)
	testing.expect(t, view_tape_err == .None)
	defer tape_release(&view_tape)
	view_parent, view_parent_err := tape_add_leaf_view(&view_tape, transposed)
	testing.expect(t, view_parent_err == .None)
	view_output, view_output_err := mean_axis_views(transposed, 1)
	testing.expect(t, view_output_err == .None)
	view_loss, view_record_err := tape_record_mean_axis(&view_tape, &view_output, view_parent, 1)
	testing.expect(t, view_record_err == .None)
	view_seed, view_seed_err := from_slice(f64, []f64{2, 4, 6}, []int{3})
	testing.expect(t, view_seed_err == .None)
	defer release(&view_seed)
	testing.expect(t, tape_backward_with_seed(&view_tape, view_loss, view_seed) == .None)
	view_gradient, view_gradient_err := tape_gradient(&view_tape, view_parent)
	testing.expect(t, view_gradient_err == .None)
	testing.expect(t, view_gradient.data[0] == 1 && view_gradient.data[1] == 1)
	testing.expect(t, view_gradient.data[2] == 2 && view_gradient.data[3] == 2)
	testing.expect(t, view_gradient.data[4] == 3 && view_gradient.data[5] == 3)
}

@(test)
test_tape_mean_axis_empty_reduction_does_not_record :: proc(t: ^testing.T) {
	empty, empty_err := new_tensor(f64, []int{2, 0, 3})
	testing.expect(t, empty_err == .None)
	defer release(&empty)
	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &empty)
	testing.expect(t, parent_err == .None)

	_, eager_err := mean_axis(empty, 1)
	testing.expect(t, eager_err == .Empty_Reduction)
	manual_output, output_err := new_tensor(f64, []int{2, 3})
	testing.expect(t, output_err == .None)
	_, record_err := tape_record_mean_axis(&tape, &manual_output, parent, 1)
	testing.expect(t, record_err == .Empty_Reduction)
	testing.expect(t, len(tape.nodes) == 1 && len(manual_output.data) == 6)
	release(&manual_output)
}

@(test)
test_tape_scalar_backward_accumulates_and_zeroes :: proc(t: ^testing.T) {
	loss, err := from_slice(f32, []f32{3.5}, []int{})
	testing.expect(t, err == .None)
	defer release(&loss)

	tape, tape_err := tape_init(f32, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	loss_value, leaf_err := tape_add_leaf_borrow(&tape, &loss)
	testing.expect(t, leaf_err == .None)

	testing.expect(t, tape_backward(&tape, loss_value) == .None)
	grad, grad_err := tape_gradient(&tape, loss_value)
	testing.expect(t, grad_err == .None)
	testing.expect(t, len(grad.shape) == 0 && grad.data[0] == 1)
	testing.expect(t, tape_backward(&tape, loss_value) == .None)
	testing.expect(t, grad.data[0] == 2)
	tape_zero_grad(&tape)
	testing.expect(t, grad.data[0] == 0)
}

@(test)
test_tape_non_scalar_seed_and_validation :: proc(t: ^testing.T) {
	input, err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	testing.expect(t, err == .None)
	defer release(&input)
	seed, seed_err := from_slice(f64, []f64{0.25, -1, 2}, []int{3})
	testing.expect(t, seed_err == .None)
	defer release(&seed)
	wrong_seed, wrong_seed_err := new_tensor(f64, []int{1, 3})
	testing.expect(t, wrong_seed_err == .None)
	defer release(&wrong_seed)

	tape, tape_err := tape_init(f64, 1)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	value, leaf_err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, leaf_err == .None)
	testing.expect(t, tape_backward(&tape, value) == .Shape_Mismatch)
	testing.expect(t, tape_backward_with_seed(&tape, value, wrong_seed) == .Shape_Mismatch)
	testing.expect(t, tape_backward_with_seed(&tape, value, seed) == .None)
	grad, grad_err := tape_gradient(&tape, value)
	testing.expect(t, grad_err == .None)
	testing.expect(t, grad.data[0] == 0.25 && grad.data[1] == -1 && grad.data[2] == 2)
}

@(test)
test_tape_records_owned_operation_and_checks_graph_identity :: proc(t: ^testing.T) {
	input, err := from_slice(f32, []f32{1, 2}, []int{2})
	testing.expect(t, err == .None)
	defer release(&input)

	tape, tape_err := tape_init(f32, 3)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, parent_err == .None)

	output, output_err := from_slice(f32, []f32{2, 4}, []int{2})
	testing.expect(t, output_err == .None)
	child, child_err := tape_record_binary(&tape, &output, .Add, parent, parent)
	testing.expect(t, child_err == .None)
	testing.expect(t, len(output.data) == 0)
	testing.expect(t, tape.nodes[child.id].operation == .Add)
	testing.expect(t, tape.nodes[child.id].parent_a == parent.id && tape.nodes[child.id].parent_count == 2)
	forward, forward_err := tape_value(&tape, child)
	testing.expect(t, forward_err == .None && forward.data[1] == 4)
	seed, seed_err := from_slice(f32, []f32{2, 4}, []int{2})
	testing.expect(t, seed_err == .None)
	defer release(&seed)
	testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
	child_grad, child_grad_err := tape_gradient(&tape, child)
	testing.expect(t, child_grad_err == .None && child_grad.data[0] == 2 && child_grad.data[1] == 4)
	parent_grad, parent_grad_err := tape_gradient(&tape, parent)
	testing.expect(t, parent_grad_err == .None && parent_grad.data[0] == 4 && parent_grad.data[1] == 8)

	foreign_tape, foreign_tape_err := tape_init(f32, 1)
	testing.expect(t, foreign_tape_err == .None)
	defer tape_release(&foreign_tape)
	foreign_value, foreign_err := tape_add_leaf_borrow(&foreign_tape, &input)
	testing.expect(t, foreign_err == .None)
	invalid_output, invalid_output_err := new_tensor(f32, []int{2})
	testing.expect(t, invalid_output_err == .None)
	defer release(&invalid_output)
	_, invalid_tape_err := tape_record_binary(&tape, &invalid_output, .Add, parent, foreign_value)
	testing.expect(t, invalid_tape_err == .Invalid_Tape)
}

@(test)
test_tape_elementwise_reverse_rules :: proc(t: ^testing.T) {
	a, a_err := from_slice(f64, []f64{6, 8}, []int{2})
	testing.expect(t, a_err == .None)
	defer release(&a)
	b, b_err := from_slice(f64, []f64{2, 4}, []int{2})
	testing.expect(t, b_err == .None)
	defer release(&b)
	seed, seed_err := from_slice(f64, []f64{1, 2}, []int{2})
	testing.expect(t, seed_err == .None)
	defer release(&seed)

	operations := [4]Tape_Operation{.Add, .Sub, .Mul, .Div}
	for operation in operations {
		tape, tape_err := tape_init(f64, 3)
		testing.expect(t, tape_err == .None)
		defer tape_release(&tape)
		av, av_err := tape_add_leaf_borrow(&tape, &a)
		testing.expect(t, av_err == .None)
		bv, bv_err := tape_add_leaf_borrow(&tape, &b)
		testing.expect(t, bv_err == .None)
		output, output_err := new_tensor(f64, []int{2})
		testing.expect(t, output_err == .None)
		child, child_err := tape_record_binary(&tape, &output, operation, av, bv)
		testing.expect(t, child_err == .None)
		testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
		grad_a, grad_a_err := tape_gradient(&tape, av)
		grad_b, grad_b_err := tape_gradient(&tape, bv)
		testing.expect(t, grad_a_err == .None && grad_b_err == .None)
		for i in 0..<2 {
			epsilon := 1e-5
			numeric_a := (binary_scalar_value(a.data[i]+epsilon, b.data[i], operation)-binary_scalar_value(a.data[i]-epsilon, b.data[i], operation))/(2*epsilon) * seed.data[i]
			numeric_b := (binary_scalar_value(a.data[i], b.data[i]+epsilon, operation)-binary_scalar_value(a.data[i], b.data[i]-epsilon, operation))/(2*epsilon) * seed.data[i]
			testing.expect(t, abs(grad_a.data[i]-numeric_a) < 1e-8)
			testing.expect(t, abs(grad_b.data[i]-numeric_b) < 1e-8)
		}
		#partial switch operation {
		case .Add:
			testing.expect(t, grad_a.data[0] == 1 && grad_a.data[1] == 2)
			testing.expect(t, grad_b.data[0] == 1 && grad_b.data[1] == 2)
		case .Sub:
			testing.expect(t, grad_a.data[0] == 1 && grad_a.data[1] == 2)
			testing.expect(t, grad_b.data[0] == -1 && grad_b.data[1] == -2)
		case .Mul:
			testing.expect(t, grad_a.data[0] == 2 && grad_a.data[1] == 8)
			testing.expect(t, grad_b.data[0] == 6 && grad_b.data[1] == 16)
		case .Div:
			testing.expect(t, grad_a.data[0] == 0.5 && grad_a.data[1] == 0.5)
			testing.expect(t, grad_b.data[0] == -1.5 && grad_b.data[1] == -1)
		}
		if operation == .Mul {
			// A repeated pass adds only the new seed contribution, not the
			// previously accumulated intermediate gradient a second time.
			testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
			testing.expect(t, grad_a.data[0] == 4 && grad_a.data[1] == 16)
			testing.expect(t, grad_b.data[0] == 12 && grad_b.data[1] == 32)
		}
	}
}

@(test)
test_tape_elementwise_broadcast_gradient_reduction :: proc(t: ^testing.T) {
	a, a_err := from_slice(f32, []f32{2, 3}, []int{2, 1})
	testing.expect(t, a_err == .None)
	defer release(&a)
	b, b_err := from_slice(f32, []f32{4, 5, 6}, []int{1, 3})
	testing.expect(t, b_err == .None)
	defer release(&b)
	seed, seed_err := from_slice(f32, []f32{1, 1, 1, 1, 1, 1}, []int{2, 3})
	testing.expect(t, seed_err == .None)
	defer release(&seed)

	tape, tape_err := tape_init(f32, 3)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	av, av_err := tape_add_leaf_borrow(&tape, &a)
	testing.expect(t, av_err == .None)
	bv, bv_err := tape_add_leaf_borrow(&tape, &b)
	testing.expect(t, bv_err == .None)
	output, output_err := new_tensor(f32, []int{2, 3})
	testing.expect(t, output_err == .None)
	child, child_err := tape_record_binary(&tape, &output, .Mul, av, bv)
	testing.expect(t, child_err == .None)
	testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
	grad_a, grad_a_err := tape_gradient(&tape, av)
	grad_b, grad_b_err := tape_gradient(&tape, bv)
	testing.expect(t, grad_a_err == .None && grad_b_err == .None)
	testing.expect(t, grad_a.data[0] == 15 && grad_a.data[1] == 15)
	testing.expect(t, grad_b.data[0] == 5 && grad_b.data[1] == 5 && grad_b.data[2] == 5)
}


@(test)
test_tape_activation_reverse_rules :: proc(t: ^testing.T) {
	input_values := []f64{-1.1, 0.4, 1.7}
	seed_values := []f64{0.5, -1.25, 2}
	operations := [3]Tape_Operation{.Relu, .Sigmoid, .Tanh}
	for operation in operations {
		input, input_err := from_slice(f64, input_values, []int{3})
		testing.expect(t, input_err == .None)
		defer release(&input)
		seed, seed_err := from_slice(f64, seed_values, []int{3})
		testing.expect(t, seed_err == .None)
		defer release(&seed)

		tape, tape_err := tape_init(f64, 2)
		testing.expect(t, tape_err == .None)
		defer tape_release(&tape)
		parent, parent_err := tape_add_leaf_borrow(&tape, &input)
		testing.expect(t, parent_err == .None)
		output, output_err := activation_forward(input, operation)
		testing.expect(t, output_err == .None)
		child, child_err := tape_record_unary(&tape, &output, operation, parent)
		testing.expect(t, child_err == .None)
		testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
		gradient, gradient_err := tape_gradient(&tape, parent)
		testing.expect(t, gradient_err == .None)

		for i in 0..<len(input_values) {
			epsilon := 1e-6
			numeric := (activation_scalar(input_values[i]+epsilon, operation)-activation_scalar(input_values[i]-epsilon, operation))/(2*epsilon) * seed_values[i]
			testing.expect(t, abs(gradient.data[i]-numeric) < 1e-8)
		}
		if operation == .Relu {
			testing.expect(t, gradient.data[0] == 0 && gradient.data[1] == -1.25 && gradient.data[2] == 2)
		}
	}
}

activation_forward :: proc(input: Tensor(f64), operation: Tape_Operation) -> (output: Tensor(f64), err: Tensor_Error) {
	#partial switch operation {
	case .Relu: return relu(input)
	case .Sigmoid: return sigmoid(input)
	case .Tanh: return tanh(input)
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
	testing.expect(t, input_err == .None)
	defer release(&input)
	weights, weight_err := from_slice(f64, weight_values, []int{2, 2})
	testing.expect(t, weight_err == .None)
	defer release(&weights)
	bias, bias_err := from_slice(f64, bias_values, []int{2})
	testing.expect(t, bias_err == .None)
	defer release(&bias)
	target, target_err := from_slice(f64, target_values, []int{2})
	testing.expect(t, target_err == .None)
	defer release(&target)

	// One dense layer: tanh(input · weights + bias), trained with squared error.
	tape, tape_err := tape_init(f64, 9)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	input_value, err := tape_add_leaf_borrow(&tape, &input)
	testing.expect(t, err == .None)
	weight_value, weight_tape_err := tape_add_leaf_borrow(&tape, &weights)
	testing.expect(t, weight_tape_err == .None)
	bias_value, bias_tape_err := tape_add_leaf_borrow(&tape, &bias)
	testing.expect(t, bias_tape_err == .None)
	target_value, target_tape_err := tape_add_leaf_borrow(&tape, &target, false)
	testing.expect(t, target_tape_err == .None)

	// The tape_<op> helpers evaluate and record each step in one call.
	linear_value, linear_err := tape_dot(&tape, input_value, weight_value)
	testing.expect(t, linear_err == .None)
	biased_value, biased_err := tape_add(&tape, linear_value, bias_value)
	testing.expect(t, biased_err == .None)
	activated_value, activated_err := tape_tanh(&tape, biased_value)
	testing.expect(t, activated_err == .None)
	residual_value, residual_err := tape_sub(&tape, activated_value, target_value)
	testing.expect(t, residual_err == .None)
	loss_value, loss_err := tape_dot(&tape, residual_value, residual_value)
	testing.expect(t, loss_err == .None)
	testing.expect(t, tape_backward(&tape, loss_value) == .None)

	input_grad, input_grad_err := tape_gradient(&tape, input_value)
	weight_grad, weight_grad_err := tape_gradient(&tape, weight_value)
	bias_grad, bias_grad_err := tape_gradient(&tape, bias_value)
	testing.expect(t, input_grad_err == .None && weight_grad_err == .None && bias_grad_err == .None)
	epsilon := 1e-6
	for i in 0..<len(input_values) {
		plus := make([]f64, len(input_values))
		minus := make([]f64, len(input_values))
		copy(plus, input_values)
		copy(minus, input_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dense_squared_loss(plus, weight_values, bias_values, target_values)-dense_squared_loss(minus, weight_values, bias_values, target_values))/(2*epsilon)
		testing.expect(t, abs(input_grad.data[i]-numeric) < 1e-8)
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
		testing.expect(t, abs(weight_grad.data[i]-numeric) < 1e-8)
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
		testing.expect(t, abs(bias_grad.data[i]-numeric) < 1e-8)
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

check_dot_reverse_case :: proc(t: ^testing.T, a_values: []f64, a_shape: []int, b_values: []f64, b_shape: []int, seed_values: []f64, seed_shape: []int, operation: Tape_Operation) {
	a, a_err := from_slice(f64, a_values, a_shape)
	testing.expect(t, a_err == .None)
	defer release(&a)
	b, b_err := from_slice(f64, b_values, b_shape)
	testing.expect(t, b_err == .None)
	defer release(&b)
	seed, seed_err := from_slice(f64, seed_values, seed_shape)
	testing.expect(t, seed_err == .None)
	defer release(&seed)

	tape, tape_err := tape_init(f64, 3)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	av, av_err := tape_add_leaf_borrow(&tape, &a)
	testing.expect(t, av_err == .None)
	bv, bv_err := tape_add_leaf_borrow(&tape, &b)
	testing.expect(t, bv_err == .None)
	output, output_err := dot(a, b)
	testing.expect(t, output_err == .None)
	child, child_err := tape_record_binary(&tape, &output, operation, av, bv)
	testing.expect(t, child_err == .None)
	testing.expect(t, tape_backward_with_seed(&tape, child, seed) == .None)
	grad_a, grad_a_err := tape_gradient(&tape, av)
	grad_b, grad_b_err := tape_gradient(&tape, bv)
	testing.expect(t, grad_a_err == .None && grad_b_err == .None)

	epsilon := 1e-5
	for i in 0..<len(a_values) {
		plus := make([]f64, len(a_values))
		minus := make([]f64, len(a_values))
		copy(plus, a_values)
		copy(minus, a_values)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (dot_weighted_value(plus, a_shape, b_values, b_shape, seed_values)-dot_weighted_value(minus, a_shape, b_values, b_shape, seed_values))/(2*epsilon)
		testing.expect(t, abs(grad_a.data[i]-numeric) < 1e-8)
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
		testing.expect(t, abs(grad_b.data[i]-numeric) < 1e-8)
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
}

check_reduction_finite_difference :: proc(t: ^testing.T, values: []f64, shape: []int, operation: Tape_Operation, axis: int, keep_dims, transpose: bool) {
	input, input_err := from_slice(f64, values, shape)
	testing.expect(t, input_err == .None)
	defer release(&input)
	input_view := as_view(input)
	view_shape := [2]int{}
	view_strides := [2]int{}
	if transpose {
		input_view, input_err = transpose_view(as_view(input), view_shape[:], view_strides[:])
		testing.expect(t, input_err == .None)
	}

	tape, tape_err := tape_init(f64, 2)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	parent, parent_err := tape_add_leaf_view(&tape, input_view)
	testing.expect(t, parent_err == .None)
	output, output_err := reduction_output(input_view, operation, axis, keep_dims)
	testing.expect(t, output_err == .None)
	seed, seed_err := new_tensor(f64, output.shape)
	testing.expect(t, seed_err == .None)
	defer release(&seed)
	for i in 0..<len(seed.data) {
		seed.data[i] = 0.35 + f64(i)*0.17
	}
	loss, record_err := record_reduction(&tape, &output, operation, parent, axis, keep_dims)
	testing.expect(t, record_err == .None)
	testing.expect(t, tape_backward_with_seed(&tape, loss, seed) == .None)
	gradient, gradient_err := tape_gradient(&tape, parent)
	testing.expect(t, gradient_err == .None)

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
		testing.expect(t, abs(gradient.data[i]-numeric) < 1e-8)
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
	case: return {}, .Unsupported_Operation
	}
}

record_reduction :: proc(tape: ^Tape(f64), output: ^Tensor(f64), operation: Tape_Operation, parent: Tape_Value(f64), axis: int, keep_dims: bool) -> (Tape_Value(f64), Tensor_Error) {
	#partial switch operation {
	case .Sum_All: return tape_record_sum_all(tape, output, parent)
	case .Mean_All: return tape_record_mean_all(tape, output, parent)
	case .Sum_Axis: return tape_record_sum_axis(tape, output, parent, axis, keep_dims)
	case .Mean_Axis: return tape_record_mean_axis(tape, output, parent, axis, keep_dims)
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
	testing.expect(t, prediction_err == .None)
	defer release(&prediction)
	target, target_err := from_slice(f64, targets, []int{3, 2})
	testing.expect(t, target_err == .None)
	defer release(&target)

	// MSE = mean_all((prediction - target) * (prediction - target)).
	tape, tape_err := tape_init(f64, 5)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	prediction_value, prediction_value_err := tape_add_leaf_borrow(&tape, &prediction)
	testing.expect(t, prediction_value_err == .None)
	target_value, target_value_err := tape_add_leaf_borrow(&tape, &target, false)
	testing.expect(t, target_value_err == .None)
	residual_value, residual_err := tape_sub(&tape, prediction_value, target_value)
	testing.expect(t, residual_err == .None)
	squared_value, squared_err := tape_mul(&tape, residual_value, residual_value)
	testing.expect(t, squared_err == .None)
	loss_value, loss_err := tape_mean_all(&tape, squared_value)
	testing.expect(t, loss_err == .None)
	testing.expect(t, tape_backward(&tape, loss_value) == .None)
	gradient, gradient_err := tape_gradient(&tape, prediction_value)
	testing.expect(t, gradient_err == .None)

	epsilon := 1e-6
	for i in 0..<len(predictions) {
		plus := make([]f64, len(predictions))
		minus := make([]f64, len(predictions))
		copy(plus, predictions)
		copy(minus, predictions)
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (batched_mse_value(plus, targets)-batched_mse_value(minus, targets))/(2*epsilon)
		testing.expect(t, abs(gradient.data[i]-numeric) < 1e-8)
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
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	input, err := from_slice(f64, []f64{-1, 2}, []int{2})
	testing.expect(t, err == .None)
	defer release(&input)

	_, err = tape_value(&tape, Tape_Value(f64){})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_gradient(&tape, Tape_Value(f64){id = 0, identity = tape.identity})
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	testing.expect_value(t, tape_backward(&tape, Tape_Value(f64){id = -1, identity = tape.identity}), Tensor_Error.Invalid_Tape)

	constant, constant_err := tape_add_leaf_borrow(&tape, &input, false)
	testing.expect(t, constant_err == .None)
	forward: Tensor_View(f64)
	forward, err = tape_value(&tape, constant)
	testing.expect(t, err == .None && forward.data[1] == 2)
	_, err = tape_gradient(&tape, constant)
	testing.expect_value(t, err, Tensor_Error.Gradient_Not_Tracked)

	// Operations on constants inherit requires_grad = false.
	rectified: Tensor(f64)
	rectified, err = relu(input)
	testing.expect(t, err == .None)
	rectified_value, record_err := tape_record_unary(&tape, &rectified, .Relu, constant)
	testing.expect(t, record_err == .None)
	_, err = tape_gradient(&tape, rectified_value)
	testing.expect_value(t, err, Tensor_Error.Gradient_Not_Tracked)
	seed: Tensor(f64)
	seed, err = from_slice(f64, []f64{1, 1}, []int{2})
	testing.expect(t, err == .None)
	defer release(&seed)
	testing.expect_value(t, tape_backward_with_seed(&tape, rectified_value, seed), Tensor_Error.Gradient_Not_Tracked)

	total: Tensor(f64)
	total, err = sum_all(input)
	testing.expect(t, err == .None)
	total_value: Tape_Value(f64)
	total_value, err = tape_record_sum_all(&tape, &total, rectified_value)
	testing.expect(t, err == .None)
	testing.expect_value(t, tape_backward(&tape, total_value), Tensor_Error.Gradient_Not_Tracked)
	testing.expect_value(t, tape_backward(&tape, rectified_value), Tensor_Error.Shape_Mismatch)
}

@(test)
test_tape_handles_survive_moves_and_reject_stale_tapes :: proc(t: ^testing.T) {
	left, err := from_slice(f64, []f64{1, 2}, []int{2})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(f64)
	right, err = from_slice(f64, []f64{3, 4}, []int{2})
	testing.expect(t, err == .None)
	defer release(&right)

	original, tape_err := tape_init(f64)
	testing.expect(t, tape_err == .None)
	left_value, left_err := tape_add_leaf_borrow(&original, &left)
	testing.expect(t, left_err == .None)
	right_value, right_err := tape_add_leaf_borrow(&original, &right)
	testing.expect(t, right_err == .None)

	// Moving the Tape value keeps its handles valid.
	tape := original
	original = {}
	defer tape_release(&tape)
	product, product_err := tape_mul(&tape, left_value, right_value)
	testing.expect_value(t, product_err, Tensor_Error.None)
	_, err = tape_value(&original, product)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	total, total_err := tape_sum_all(&tape, product)
	testing.expect_value(t, total_err, Tensor_Error.None)
	testing.expect_value(t, tape_backward(&tape, total), Tensor_Error.None)
	left_grad, left_grad_err := tape_gradient(&tape, left_value)
	testing.expect(t, left_grad_err == .None)
	expect_f64_data(t, left_grad.data, []f64{3, 4})

	// A handle from another tape is rejected even when its id is in range.
	other, other_err := tape_init(f64)
	testing.expect(t, other_err == .None)
	defer tape_release(&other)
	other_value, other_value_err := tape_add_leaf_borrow(&other, &left)
	testing.expect(t, other_value_err == .None)
	_, err = tape_value(&tape, other_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_add(&tape, left_value, other_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)

	// tape_reset invalidates old handles but keeps the tape usable.
	tape_reset(&tape)
	testing.expect(t, len(tape.nodes) == 0)
	_, err = tape_value(&tape, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	testing.expect_value(t, tape_backward(&tape, total), Tensor_Error.Invalid_Tape)
	fresh, fresh_err := tape_add_leaf_borrow(&tape, &right)
	testing.expect(t, fresh_err == .None && fresh.id == 0)
	_, err = tape_value(&tape, fresh)
	testing.expect_value(t, err, Tensor_Error.None)
	// The old handle has the same id as fresh but belongs to the earlier lifetime.
	_, err = tape_value(&tape, left_value)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
}

@(test)
test_tape_nil_tape_and_unsupported_records_leave_output_with_caller :: proc(t: ^testing.T) {
	tape, tape_err := tape_init(f64)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	left, err := from_slice(f64, []f64{1, 2}, []int{2})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(f64)
	right, err = from_slice(f64, []f64{3, 4}, []int{2})
	testing.expect(t, err == .None)
	defer release(&right)
	left_value, left_err := tape_add_leaf_borrow(&tape, &left)
	testing.expect(t, left_err == .None)
	right_value, right_err := tape_add_leaf_borrow(&tape, &right)
	testing.expect(t, right_err == .None)
	output: Tensor(f64)
	output, err = add(left, right)
	testing.expect(t, err == .None)
	defer release(&output)

	_, err = tape_record_unary(&tape, &output, .Add, left_value)
	testing.expect_value(t, err, Tensor_Error.Unsupported_Operation)
	_, err = tape_record_binary(&tape, &output, .Tanh, left_value, right_value)
	testing.expect_value(t, err, Tensor_Error.Unsupported_Operation)
	testing.expect(t, len(tape.nodes) == 2)

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
	testing.expect(t, len(output.data) == 2 && output.data[1] == 6)
}

@(test)
test_tape_broadcast_gradients_through_strided_parent :: proc(t: ^testing.T) {
	storage, err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&storage)
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	// transposed is [[1, 3, 5], [2, 4, 6]] with strides {1, 2}.
	transposed, view_err := transpose_view(as_view(storage), shape_buffer[:], strides_buffer[:])
	testing.expect(t, view_err == .None)
	row: Tensor(f64)
	row, err = from_slice(f64, []f64{10, 20, 30}, []int{3})
	testing.expect(t, err == .None)
	defer release(&row)
	seed: Tensor(f64)
	seed, err = from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&seed)

	tape, tape_err := tape_init(f64, 4)
	testing.expect(t, tape_err == .None)
	defer tape_release(&tape)
	matrix_value, matrix_err := tape_add_leaf_view(&tape, transposed)
	testing.expect(t, matrix_err == .None)
	row_value, row_err := tape_add_leaf_borrow(&tape, &row)
	testing.expect(t, row_err == .None)

	product: Tensor(f64)
	product, err = mul_views(transposed, as_view(row))
	testing.expect(t, err == .None)
	product_value, product_err := tape_record_binary(&tape, &product, .Mul, matrix_value, row_value)
	testing.expect(t, product_err == .None)
	testing.expect_value(t, tape_backward_with_seed(&tape, product_value, seed), Tensor_Error.None)
	matrix_grad, matrix_grad_err := tape_gradient(&tape, matrix_value)
	testing.expect(t, matrix_grad_err == .None)
	expect_f64_data(t, matrix_grad.data, []f64{10, 40, 90, 40, 100, 180})
	row_grad, row_grad_err := tape_gradient(&tape, row_value)
	testing.expect(t, row_grad_err == .None)
	expect_f64_data(t, row_grad.data, []f64{9, 26, 51})

	tape_zero_grad(&tape)
	expect_f64_data(t, matrix_grad.data, []f64{0, 0, 0, 0, 0, 0})
	quotient: Tensor(f64)
	quotient, err = div_views(transposed, as_view(row))
	testing.expect(t, err == .None)
	quotient_value, quotient_err := tape_record_binary(&tape, &quotient, .Div, matrix_value, row_value)
	testing.expect(t, quotient_err == .None)
	testing.expect_value(t, tape_backward_with_seed(&tape, quotient_value, seed), Tensor_Error.None)
	expect_f64_data(t, matrix_grad.data, []f64{0.1, 0.1, 0.1, 0.4, 0.25, 0.2})
	expect_f64_data(t, row_grad.data, []f64{-9.0/100, -26.0/400, -51.0/900})
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
	activated_view := tape_value(&tape, activated_value) or_return
	means := mean_axis_views(activated_view, 1, false, allocator) or_return
	defer release(&means)
	means_value := tape_record_mean_axis(&tape, &means, activated_value, 1) or_return
	means_view := tape_value(&tape, means_value) or_return
	total := sum_all_views(means_view, allocator) or_return
	defer release(&total)
	total_value := tape_record_sum_all(&tape, &total, means_value) or_return
	return tape_backward(&tape, total_value)
}

@(test)
test_tape_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, tape_allocation_scenario)
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
	case .Sum_All:
		return tape_sum_all(tape, a)
	case .Mean_All:
		return tape_mean_all(tape, a)
	case .Sum_Axis:
		return tape_sum_axis(tape, a, -1)
	case .Mean_Axis:
		return tape_mean_axis(tape, a, 0, true)
	}
	return {}, .Unsupported_Operation
}

// helper_objective resets the tape, applies operation, and records
// sum_all(sin-weighted output) so every output element has a distinct weight.
helper_objective :: proc(tape: ^Tape(f64), a, b: ^Tensor(f64), operation: Tape_Operation) -> (loss: Tape_Value(f64), a_value, b_value: Tape_Value(f64), err: Tensor_Error) {
	tape_reset(tape)
	a_value = tape_add_leaf_borrow(tape, a) or_return
	b_value = tape_add_leaf_borrow(tape, b) or_return
	output := apply_tape_helper(tape, a_value, b_value, operation) or_return
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

check_tape_helper :: proc(t: ^testing.T, operation: Tape_Operation, a_values: []f64, a_shape: []int, b_values: []f64, b_shape: []int) {
	a, err := from_slice(f64, a_values, a_shape)
	testing.expect(t, err == .None)
	defer release(&a)
	b: Tensor(f64)
	b, err = from_slice(f64, b_values, b_shape)
	testing.expect(t, err == .None)
	defer release(&b)
	tape: Tape(f64)
	tape, err = tape_init(f64)
	testing.expect(t, err == .None)
	defer tape_release(&tape)

	loss, a_value, b_value, objective_err := helper_objective(&tape, &a, &b, operation)
	testing.expectf(t, objective_err == .None, "%v: %v", operation, objective_err)
	if objective_err != .None {
		return
	}
	testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
	a_grad, _ := tape_gradient(&tape, a_value)
	b_grad, _ := tape_gradient(&tape, b_value)
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
			plus_loss, _, _, plus_err := helper_objective(&tape, &a, &b, operation)
			plus_view, _ := tape_value(&tape, plus_loss)
			plus := plus_view.data[0]
			input.data[i] = original - epsilon
			minus_loss, _, _, minus_err := helper_objective(&tape, &a, &b, operation)
			minus_view, _ := tape_value(&tape, minus_loss)
			minus := minus_view.data[0]
			input.data[i] = original
			testing.expect(t, plus_err == .None && minus_err == .None)
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
	for operation in ([]Tape_Operation{.Relu, .Sigmoid, .Tanh, .Sum_All, .Mean_All, .Sum_Axis, .Mean_Axis}) {
		check_tape_helper(t, operation, matrix_values, []int{2, 3}, row_values, []int{3})
	}
}

@(test)
test_tape_backward_visits_only_reachable_nodes :: proc(t: ^testing.T) {
	x, err := from_slice(f64, []f64{2}, []int{1})
	testing.expect(t, err == .None)
	defer release(&x)
	infinity: Tensor(f64)
	infinity, err = from_slice(f64, []f64{math.inf_f64(1)}, []int{1})
	testing.expect(t, err == .None)
	defer release(&infinity)
	zero: Tensor(f64)
	zero, err = from_slice(f64, []f64{0}, []int{1})
	testing.expect(t, err == .None)
	defer release(&zero)

	tape: Tape(f64)
	tape, err = tape_init(f64)
	testing.expect(t, err == .None)
	defer tape_release(&tape)
	x_value, _ := tape_add_leaf_borrow(&tape, &x)
	infinity_value, _ := tape_add_leaf_borrow(&tape, &infinity, false)
	zero_value, _ := tape_add_leaf_borrow(&tape, &zero, false)

	// An unrelated branch whose local derivative is infinite does not touch x.
	scaled, scaled_err := tape_mul(&tape, x_value, infinity_value)
	testing.expect(t, scaled_err == .None)
	direct, direct_err := tape_sum_all(&tape, x_value)
	testing.expect(t, direct_err == .None)
	testing.expect_value(t, tape_backward(&tape, direct), Tensor_Error.None)
	x_grad, _ := tape_gradient(&tape, x_value)
	expect_f64_data(t, x_grad.data, []f64{1})

	// On a reachable path, 0 * inf gives NaN, as in other autodiff libraries.
	tape_zero_grad(&tape)
	silenced, silenced_err := tape_mul(&tape, scaled, zero_value)
	testing.expect(t, silenced_err == .None)
	loss, loss_err := tape_sum_all(&tape, silenced)
	testing.expect(t, loss_err == .None)
	testing.expect_value(t, tape_backward(&tape, loss), Tensor_Error.None)
	testing.expect(t, math.is_nan(x_grad.data[0]))
}
