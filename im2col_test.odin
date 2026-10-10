package tensor

import "core:math"
import "core:mem"
import "core:testing"

// expect_im2col_matches_gather compares im2col of source with direct indexing
// of every window element.
expect_im2col_matches_gather :: proc(t: ^testing.T, source: Tensor_View(f64), kernel_height, kernel_width, stride: int, loc := #caller_location) {
	result, err := im2col_views(source, kernel_height, kernel_width, stride)
	require_ok(t, err, loc)
	defer release(&result)
	rank := len(source.shape)
	batch_shape := source.shape[:rank-3]
	batch_count, _ := _element_count(batch_shape)
	channels := source.shape[rank-1]
	out_height := (source.shape[rank-3]-kernel_height)/stride + 1
	out_width := (source.shape[rank-2]-kernel_width)/stride + 1
	testing.expectf(t, result.shape[rank-3] == out_height && result.shape[rank-2] == out_width && result.shape[rank-1] == kernel_height*kernel_width*channels, "shape %v", result.shape, loc = loc)

	indices: [MAX_RANK]int
	next := 0
	for batch in 0..<batch_count {
		remaining := batch
		for axis := len(batch_shape)-1; axis >= 0; axis -= 1 {
			indices[axis] = remaining % batch_shape[axis]
			remaining /= batch_shape[axis]
		}
		for oy in 0..<out_height do for ox in 0..<out_width do for dy in 0..<kernel_height do for dx in 0..<kernel_width do for c in 0..<channels {
			indices[rank-3], indices[rank-2], indices[rank-1] = oy*stride+dy, ox*stride+dx, c
			expected, get_err := get_at_view(source, indices[:rank])
			require_ok(t, get_err, loc)
			testing.expectf(t, result.data[next] == expected, "element %d: got %v, want %v", next, result.data[next], expected, loc = loc)
			next += 1
		}
	}
	testing.expect_value(t, next, len(result.data), loc = loc)
}

@(test)
test_im2col_matches_direct_gather :: proc(t: ^testing.T) {
	values := make([]f64, 2*5*4*3)
	defer delete(values)
	for &value, i in values {
		value = f64(i)
	}
	contiguous, err := from_slice(f64, values, []int{2, 5, 4, 3})
	require_ok(t, err)
	defer release(&contiguous)
	expect_im2col_matches_gather(t, as_view(contiguous), 3, 2, 1)
	expect_im2col_matches_gather(t, as_view(contiguous), 2, 2, 2)
	// Stride 2 over 5 rows and kernel 2 leaves the last row unused.
	expect_im2col_matches_gather(t, as_view(contiguous), 2, 3, 2)

	// A permuted view has strided channels, so rows are gathered one element
	// at a time.
	permuted_source, permuted_err := from_slice(f64, values, []int{3, 2, 5, 4})
	require_ok(t, permuted_err)
	defer release(&permuted_source)
	shape_buffer, strides_buffer: [4]int
	permuted := permute_view(as_view(permuted_source), []int{1, 2, 3, 0}, shape_buffer[:], strides_buffer[:]) or_else panic("permute")
	expect_im2col_matches_gather(t, permuted, 3, 2, 1)
	expect_im2col_matches_gather(t, permuted, 2, 2, 2)

	// No batch axes, and several of them.
	require_ok(t, reshape(&contiguous, []int{10, 4, 3}))
	expect_im2col_matches_gather(t, as_view(contiguous), 4, 4, 3)
	require_ok(t, reshape(&contiguous, []int{1, 2, 1, 5, 4, 3}))
	expect_im2col_matches_gather(t, as_view(contiguous), 3, 3, 1)
}

@(test)
test_im2col_validates_arguments :: proc(t: ^testing.T) {
	image, err := new_tensor(f64, []int{1, 4, 4, 2})
	require_ok(t, err)
	defer release(&image)
	matrix_source: Tensor(f64)
	matrix_source, err = new_tensor(f64, []int{4, 4})
	require_ok(t, err)
	defer release(&matrix_source)

	_, err = im2col(matrix_source, 2, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = im2col(image, 0, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = im2col(image, 2, 2, 0)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = im2col(image, 5, 2)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{1, 3, 3, 8})
	require_ok(t, err)
	defer release(&output)
	require_ok(t, im2col_into(&output, image, 2, 2))
	testing.expect_value(t, im2col_into(&output, image, 3, 3), Tensor_Error.Shape_Mismatch)
	small_output, small_err := new_tensor(f64, []int{1, 4, 4, 2})
	require_ok(t, small_err)
	defer release(&small_output)
	self_view := Tensor_View(f64){data = small_output.data, shape = small_output.shape, strides = small_output.strides}
	testing.expect_value(t, im2col_views_into(&small_output, self_view, 1, 1), Tensor_Error.Aliased_Output)
}

// im2col_helper applies im2col with a 3x2 kernel and stride 2.
im2col_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (Tape_Value(f64), Tensor_Error) {
	return tape_im2col(tape, a, 3, 2, 2)
}

// transposed_im2col_helper reads im2col's input through a transposed view, so
// the backward pass must route gradients through the view.
transposed_im2col_helper :: proc(tape: ^Tape(f64), a, b: Tape_Value(f64), operation: Tape_Operation) -> (result: Tape_Value(f64), err: Tensor_Error) {
	transposed := tape_transpose(tape, a) or_return
	return tape_im2col(tape, transposed, 2, 2)
}

@(test)
test_tape_im2col_matches_finite_differences :: proc(t: ^testing.T) {
	values := make([]f64, 5*4*2)
	defer delete(values)
	for &value, i in values {
		value = math.sin(f64(i)*0.7)
	}
	row_values := []f64{0.7, -0.5, 1.3}
	// Overlapping 2x2 windows make most elements feed several rows.
	check_tape_helper(t, .Im2col, values[:18], []int{3, 3, 2}, row_values, []int{3})
	check_tape_helper(t, .Im2col, values, []int{5, 4, 2}, row_values, []int{3}, im2col_helper)
	check_tape_helper(t, .Im2col, values, []int{2, 4, 5}, row_values, []int{3}, transposed_im2col_helper)
}

@(test)
test_tape_im2col_without_gradient_and_invalid_handles :: proc(t: ^testing.T) {
	tape, err := tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	image: Tensor(f64)
	image, err = new_tensor(f64, []int{3, 3, 1})
	require_ok(t, err)
	defer release(&image)
	constant := tape_add_leaf_view(&tape, as_view(image), false) or_else panic("leaf")
	windows := tape_im2col(&tape, constant, 2, 2) or_else panic("im2col")
	_, grad_err := tape_gradient(&tape, windows)
	testing.expect_value(t, grad_err, Tensor_Error.Gradient_Not_Tracked)
	_, err = tape_im2col(&tape, Tape_Value(f64){}, 2, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
	_, err = tape_im2col(&tape, constant, 4, 1)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
}

im2col_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 4, allocator) or_return
	defer tape_release(&tape)
	image := new_tensor(f64, []int{1, 3, 3, 2}, allocator) or_return
	defer release(&image)
	leaf := tape_add_leaf_borrow(&tape, &image) or_return
	windows := tape_im2col(&tape, leaf, 2, 2) or_return
	total := tape_sum_all(&tape, windows) or_return
	direct := im2col(image, 2, 2, allocator = allocator) or_return
	release(&direct)
	return tape_backward(&tape, total)
}

@(test)
test_im2col_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, im2col_allocation_scenario)
}
