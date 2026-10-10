package tensor

import "core:math"
import "core:mem"
import "core:testing"

// reference_pool pools x the long way: slice off rows and columns that do not
// fill a window, reshape into [..., out_h, pool_h, out_w, pool_w, c] blocks,
// and reduce the two window axes with max_axis.
reference_pool :: proc(tape: ^Tape(f64), x: Tape_Value(f64), pool_height, pool_width: int) -> (result: Tape_Value(f64), err: Tensor_Error) {
	value := tape_value(tape, x) or_return
	rank := len(value.shape)
	height_axis := rank - 3
	out_height, out_width := value.shape[height_axis]/pool_height, value.shape[height_axis+1]/pool_width
	region := tape_slice(tape, x, height_axis, 0, out_height*pool_height) or_return
	region = tape_slice(tape, region, height_axis+1, 0, out_width*pool_width) or_return
	shape: [MAX_RANK]int
	copy(shape[:], value.shape[:height_axis])
	blocks := [5]int{out_height, pool_height, out_width, pool_width, value.shape[rank-1]}
	copy(shape[height_axis:], blocks[:])
	reshaped := tape_reshape(tape, region, shape[:height_axis+5]) or_return
	columns := tape_max_axis(tape, reshaped, height_axis+3) or_return
	return tape_max_axis(tape, columns, height_axis+1)
}

// expect_pool_matches_reference checks max_pool2d's values and the gradient
// tape_max_pool2d sends back against reference_pool, exactly, for an upstream
// gradient with a distinct weight per output element.
expect_pool_matches_reference :: proc(t: ^testing.T, values: []f64, shape: []int, pool_height, pool_width: int, loc := #caller_location) {
	source, err := from_slice(f64, values, shape)
	require_ok(t, err, loc)
	defer release(&source)
	results: [2]Tensor(f64)
	gradients: [2]Tensor(f64)
	defer for &r in results do release(&r)
	defer for &g in gradients do release(&g)
	for which in 0..<2 {
		tape: Tape(f64)
		tape, err = tape_init(f64)
		require_ok(t, err, loc)
		defer tape_release(&tape)
		x := tape_add_leaf_borrow(&tape, &source) or_else panic("leaf")
		pooled: Tape_Value(f64)
		if which == 0 {
			pooled, err = tape_max_pool2d(&tape, x, pool_height, pool_width)
		} else {
			pooled, err = reference_pool(&tape, x, pool_height, pool_width)
		}
		require_ok(t, err, loc)
		value := tape_value(&tape, pooled) or_else panic("value")
		results[which] = copy_view(value) or_else panic("copy")
		seed := new_tensor(f64, value.shape) or_else panic("seed")
		defer release(&seed)
		for &s, i in seed.data do s = 1 + f64(i)
		require_ok(t, tape_backward_with_seed(&tape, pooled, seed), loc)
		gradient := tape_gradient(&tape, x) or_else panic("gradient")
		gradients[which] = copy_view(gradient) or_else panic("copy")
	}
	testing.expectf(t, _shapes_match(results[0].shape, results[1].shape), "shape %v, want %v", results[0].shape, results[1].shape, loc = loc)
	for value, i in results[0].data {
		want := results[1].data[i]
		same := value == want || (value != value && want != want)
		testing.expectf(t, same, "output %d: got %v, want %v", i, value, want, loc = loc)
	}
	for value, i in gradients[0].data {
		testing.expectf(t, value == gradients[1].data[i], "gradient %d: got %v, want %v", i, value, gradients[1].data[i], loc = loc)
	}
}

@(test)
test_max_pool2d_matches_reshaped_max_axis :: proc(t: ^testing.T) {
	values := make([]f64, 2*5*7*3)
	defer delete(values)
	for &value, i in values {
		value = math.sin(f64(i) * 1.3)
	}
	// Remainder rows and columns, non-square windows, one batch axis or none.
	expect_pool_matches_reference(t, values, []int{2, 5, 7, 3}, 2, 2)
	expect_pool_matches_reference(t, values, []int{2, 5, 7, 3}, 2, 3)
	expect_pool_matches_reference(t, values, []int{10, 7, 3}, 3, 1)
	expect_pool_matches_reference(t, values, []int{1, 2, 5, 7, 3}, 5, 7)

	// Ties go to the first element in row-major order and NaN wins, in the
	// value and in the gradient.
	ties := []f64{
		1, 3, 3, 0,
		3, 2, 0, 0,
		math.nan_f64(), 5, -1, -1,
		5, math.nan_f64(), -1, -1,
	}
	expect_pool_matches_reference(t, ties, []int{4, 4, 1}, 2, 2)
}

@(test)
test_max_pool2d_reads_strided_views_and_integers :: proc(t: ^testing.T) {
	values := make([]i32, 3*4*6)
	defer delete(values)
	for &value, i in values {
		value = i32((i * 37) % 23)
	}
	// [3, 4, 6] permuted to [4, 6, 3]: channels are strided.
	source, err := from_slice(i32, values, []int{3, 4, 6})
	require_ok(t, err)
	defer release(&source)
	shape_buffer, strides_buffer: [3]int
	permuted := permute_view(as_view(source), []int{1, 2, 0}, shape_buffer[:], strides_buffer[:]) or_else panic("permute")
	pooled := max_pool2d_views(permuted, 2, 3) or_else panic("pool")
	defer release(&pooled)
	testing.expect(t, _shapes_match(pooled.shape, []int{2, 2, 3}))
	for oy in 0..<2 do for ox in 0..<2 do for c in 0..<3 {
		want := min(i32)
		for dy in 0..<2 do for dx in 0..<3 {
			value := get_at_view(permuted, []int{oy*2+dy, ox*3+dx, c}) or_else panic("get")
			want = max(want, value)
		}
		testing.expect_value(t, pooled.data[(oy*2+ox)*3+c], want)
	}
}

@(test)
test_max_pool2d_validates_arguments :: proc(t: ^testing.T) {
	image, err := new_tensor(f64, []int{1, 4, 4, 2})
	require_ok(t, err)
	defer release(&image)
	flat: Tensor(f64)
	flat, err = new_tensor(f64, []int{4, 4})
	require_ok(t, err)
	defer release(&flat)

	_, err = max_pool2d(flat, 2, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = max_pool2d(image, 0, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = max_pool2d(image, 2, 5)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{1, 2, 2, 2})
	require_ok(t, err)
	defer release(&output)
	require_ok(t, max_pool2d_into(&output, image, 2, 2))
	testing.expect_value(t, max_pool2d_into(&output, image, 1, 1), Tensor_Error.Shape_Mismatch)
	self_view := Tensor_View(f64){data = image.data, shape = image.shape, strides = image.strides}
	testing.expect_value(t, max_pool2d_views_into(&image, self_view, 1, 1), Tensor_Error.Aliased_Output)

	tape: Tape(f64)
	tape, err = tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	_, err = tape_max_pool2d(&tape, Tape_Value(f64){}, 2, 2)
	testing.expect_value(t, err, Tensor_Error.Invalid_Tape)
}

@(test)
test_tape_max_pool2d_matches_finite_differences :: proc(t: ^testing.T) {
	values := make([]f64, 5*4*2)
	defer delete(values)
	// Distinct values keep every window's maximum differentiable.
	for &value, i in values {
		value = math.sin(f64(i)*2.1) + 0.01*f64(i)
	}
	row_values := []f64{0.7, -0.5, 1.3}
	check_tape_helper(t, .Max_Pool2D, values, []int{5, 4, 2}, row_values, []int{3})
}

pool_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	tape := tape_init(f64, 4, allocator) or_return
	defer tape_release(&tape)
	image := new_tensor(f64, []int{1, 4, 4, 2}, allocator) or_return
	defer release(&image)
	for &value, i in image.data do value = f64(i % 5)
	leaf := tape_add_leaf_borrow(&tape, &image) or_return
	pooled := tape_max_pool2d(&tape, leaf, 2, 2) or_return
	total := tape_sum_all(&tape, pooled) or_return
	direct := max_pool2d(image, 2, 2, allocator) or_return
	release(&direct)
	return tape_backward(&tape, total)
}

@(test)
test_max_pool2d_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, pool_allocation_scenario)
}
