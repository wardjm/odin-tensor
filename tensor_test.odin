package tensor

import "base:intrinsics"
import "core:math"
import "core:mem"
import "core:testing"

expect_i32_data :: proc(t: ^testing.T, actual, expected: []i32) {
	testing.expect(t, len(actual) == len(expected))
	if len(actual) != len(expected) {
		return
	}
	for i in 0..<len(expected) {
		testing.expect(t, actual[i] == expected[i])
	}
}

@(test)
test_axis_and_global_reductions :: proc(t: ^testing.T) {
	values, err := from_slice(f32, []f32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)

	row_sums: Tensor(f32)
	row_sums, err = sum_axis(values, 1)
	testing.expect(t, err == .None)
	defer release(&row_sums)
	testing.expect(t, len(row_sums.shape) == 1 && row_sums.shape[0] == 2)
	testing.expect(t, row_sums.data[0] == 6 && row_sums.data[1] == 15)

	column_means: Tensor(f32)
	column_means, err = mean_axis(values, 0)
	testing.expect(t, err == .None)
	defer release(&column_means)
	testing.expect(t, column_means.data[0] == 2.5 && column_means.data[1] == 3.5 && column_means.data[2] == 4.5)

	column_max: Tensor(f32)
	column_max, err = max_axis(values, 0, true)
	testing.expect(t, err == .None)
	defer release(&column_max)
	testing.expect(t, len(column_max.shape) == 2 && column_max.shape[0] == 1 && column_max.shape[1] == 3)
	testing.expect(t, column_max.data[0] == 4 && column_max.data[1] == 5 && column_max.data[2] == 6)

	row_min: Tensor(f32)
	row_min, err = min_axis(values, 1)
	testing.expect(t, err == .None)
	defer release(&row_min)
	testing.expect(t, row_min.data[0] == 1 && row_min.data[1] == 4)

	output: Tensor(f32)
	output, err = new_tensor(f32, []int{2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect(t, sum_axis_into(&output, values, 1) == .None)
	testing.expect(t, output.data[0] == 6 && output.data[1] == 15)

	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	transposed_sum: Tensor(f32)
	transposed_sum, err = sum_axis_views(transposed, 1)
	testing.expect(t, err == .None)
	defer release(&transposed_sum)
	testing.expect(t, transposed_sum.data[0] == 5 && transposed_sum.data[1] == 7 && transposed_sum.data[2] == 9)

	all_sum: Tensor(f32)
	all_sum, err = sum_all(values)
	testing.expect(t, err == .None)
	defer release(&all_sum)
	testing.expect(t, len(all_sum.shape) == 0 && all_sum.data[0] == 21)
	all_mean: Tensor(f32)
	all_mean, err = mean_all(values)
	testing.expect(t, err == .None)
	defer release(&all_mean)
	testing.expect(t, all_mean.data[0] == 3.5)
	all_max: Tensor(f32)
	all_max, err = max_all(values)
	testing.expect(t, err == .None)
	defer release(&all_max)
	testing.expect(t, all_max.data[0] == 6)
	all_min: Tensor(f32)
	all_min, err = min_all(values)
	testing.expect(t, err == .None)
	defer release(&all_min)
	testing.expect(t, all_min.data[0] == 1)

	_, err = sum_axis(values, 2)
	testing.expect(t, err == .Out_Of_Bounds)
}

@(test)
test_reductions_on_empty_dimensions :: proc(t: ^testing.T) {
	empty, err := new_tensor(f32, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)

	sums: Tensor(f32)
	sums, err = sum_axis(empty, 1)
	testing.expect(t, err == .None)
	defer release(&sums)
	testing.expect(t, sums.data[0] == 0 && sums.data[1] == 0)
	_, err = mean_axis(empty, 1)
	testing.expect(t, err == .Empty_Reduction)
	_, err = max_axis(empty, 1)
	testing.expect(t, err == .Empty_Reduction)
	_, err = mean_all(empty)
	testing.expect(t, err == .Empty_Reduction)
	_, err = max_all(empty)
	testing.expect(t, err == .Empty_Reduction)
}

@(test)
test_products_along_an_axis_and_over_all_elements :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&values)

	columns: Tensor(f64)
	columns, err = prod_axis(values, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&columns)
	expect_shape(t, columns.shape, []int{2})
	expect_f64_data(t, columns.data, []f64{3, 8})

	kept_rows: Tensor(f64)
	kept_rows, err = prod_axis(values, -1, true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&kept_rows)
	expect_shape(t, kept_rows.shape, []int{2, 1})
	expect_f64_data(t, kept_rows.data, []f64{2, 12})

	// A transposed view multiplies along its own rows: [[1,3],[2,4]].
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	transposed_rows: Tensor(f64)
	transposed_rows, err = prod_axis_views(transposed, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_rows)
	expect_f64_data(t, transposed_rows.data, []f64{3, 8})

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect_value(t, prod_axis_into(&output, values, 1), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{2, 12})
	testing.expect_value(t, prod_axis_views_into(&output, transposed, 0), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{2, 12})

	total: Tensor(f64)
	total, err = prod_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&total)
	expect_shape(t, total.shape, []int{})
	expect_f64_data(t, total.data, []f64{24})
	transposed_total: Tensor(f64)
	transposed_total, err = prod_all_views(transposed)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_total)
	expect_f64_data(t, transposed_total.data, []f64{24})

	// The product of an empty reduction is 1.
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)
	empty_rows: Tensor(f64)
	empty_rows, err = prod_axis(empty, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_rows)
	expect_f64_data(t, empty_rows.data, []f64{1, 1})
	empty_total: Tensor(f64)
	empty_total, err = prod_all(empty)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_total)
	expect_f64_data(t, empty_total.data, []f64{1})

	// Integer products wrap on overflow: 16 * 16 = 256 wraps to 0 in i8, and
	// 2^7 = 128 wraps to -128.
	small: Tensor(i8)
	small, err = from_slice(i8, []i8{16, 16, 2, 2, 2, 2, 2, 2, 2}, []int{9})
	testing.expect(t, err == .None)
	defer release(&small)
	small_total: Tensor(i8)
	small_total, err = prod_all(small)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&small_total)
	testing.expect_value(t, small_total.data[0], i8(0))
	powers: Tensor(i8)
	powers, err = from_slice(i8, []i8{2, 2, 2, 2, 2, 2, 2}, []int{7})
	testing.expect(t, err == .None)
	defer release(&powers)
	powers_total: Tensor(i8)
	powers_total, err = prod_axis(powers, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&powers_total)
	testing.expect_value(t, powers_total.data[0], i8(-128))

	_, err = prod_axis(values, 2)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
}

@(test)
test_variance_and_standard_deviation :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 2, 3, 4}, []int{4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&values)

	variance: Tensor(f64)
	variance, err = var_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&variance)
	expect_shape(t, variance.shape, []int{})
	expect_f64_data(t, variance.data, []f64{1.25})
	deviation: Tensor(f64)
	deviation, err = std_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&deviation)
	expect_f64_data(t, deviation.data, []f64{math.sqrt(f64(1.25))})
	sample_variance: Tensor(f64)
	sample_variance, err = var_all(values, ddof = 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&sample_variance)
	expect_f64_data(t, sample_variance.data, []f64{5.0 / 3.0})

	axis_variance: Tensor(f64)
	axis_variance, err = var_axis(values, -1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&axis_variance)
	expect_shape(t, axis_variance.shape, []int{})
	expect_f64_data(t, axis_variance.data, []f64{1.25})
	axis_deviation: Tensor(f64)
	axis_deviation, err = std_axis(values, 0, ddof = 1, keep_dims = true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&axis_deviation)
	expect_shape(t, axis_deviation.shape, []int{1})
	expect_f64_data(t, axis_deviation.data, []f64{math.sqrt(f64(5.0) / 3.0)})

	// [[1,2],[3,6]]: columns [1,3] and [2,6] have variance 1 and 4; rows [1,2]
	// and [3,6] have variance 0.25 and 2.25.
	grid: Tensor(f32)
	grid, err = from_slice(f32, []f32{1, 2, 3, 6}, []int{2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&grid)
	columns: Tensor(f32)
	columns, err = var_axis(grid, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&columns)
	expect_f32_close(t, columns.data, []f32{1, 4})
	column_deviations: Tensor(f32)
	column_deviations, err = std_axis(grid, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&column_deviations)
	expect_f32_close(t, column_deviations.data, []f32{1, 2})
	sample_columns: Tensor(f32)
	sample_columns, err = var_axis(grid, 0, ddof = 1, keep_dims = true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&sample_columns)
	expect_shape(t, sample_columns.shape, []int{1, 2})
	expect_f32_close(t, sample_columns.data, []f32{2, 8})

	// A transposed view reduces along its own rows, which are the columns.
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(grid), transposed_shape[:], transposed_strides[:])
	testing.expect_value(t, view_err, Tensor_Error.None)
	transposed_rows: Tensor(f32)
	transposed_rows, err = var_axis_views(transposed, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_rows)
	expect_f32_close(t, transposed_rows.data, []f32{1, 4})
	transposed_deviations: Tensor(f32)
	transposed_deviations, err = std_axis_views(transposed, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_deviations)
	expect_f32_close(t, transposed_deviations.data, []f32{0.5, 1.5})

	output: Tensor(f32)
	output, err = new_tensor(f32, []int{2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&output)
	testing.expect_value(t, var_axis_into(&output, grid, 1), Tensor_Error.None)
	expect_f32_close(t, output.data, []f32{0.25, 2.25})
	testing.expect_value(t, std_axis_into(&output, grid, 1, 1), Tensor_Error.None)
	expect_f32_close(t, output.data, []f32{math.sqrt(f32(0.5)), math.sqrt(f32(4.5))})
	testing.expect_value(t, var_axis_views_into(&output, transposed, 0), Tensor_Error.None)
	expect_f32_close(t, output.data, []f32{0.25, 2.25})
	testing.expect_value(t, std_axis_views_into(&output, transposed, 1), Tensor_Error.None)
	expect_f32_close(t, output.data, []f32{1, 2})

	// The whole-tensor forms follow the view's elements, whatever their order:
	// the mean of [1,2,3,6] is 3, so the variance is (4+1+0+9)/4 = 3.5.
	transposed_total: Tensor(f32)
	transposed_total, err = var_all_views(transposed)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_total)
	expect_f32_close(t, transposed_total.data, []f32{3.5})
	transposed_total_deviation: Tensor(f32)
	transposed_total_deviation, err = std_all_views(transposed, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_total_deviation)
	expect_f32_close(t, transposed_total_deviation.data, []f32{math.sqrt(f32(14.0) / 3.0)})

	// ddof must leave at least one degree of freedom.
	_, err = var_all(values, ddof = 4)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = std_all(values, ddof = 5)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = var_axis(grid, 0, ddof = 2)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, std_axis_into(&output, grid, 1, 2), Tensor_Error.Empty_Reduction)
	_, err = var_axis_views(transposed, 0, ddof = 2)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = std_all_views(transposed, ddof = 4)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, var_axis_views_into(&output, transposed, 1, 2), Tensor_Error.Empty_Reduction)

	// A negative ddof divides by more than n, as in NumPy: 5/(4+1) = 1.
	widened: Tensor(f64)
	widened, err = var_all(values, ddof = -1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&widened)
	expect_f64_data(t, widened.data, []f64{1})
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty)
	_, err = var_all(empty)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	// An empty reduction is an error even when a negative ddof would leave
	// degrees of freedom.
	_, err = var_all(empty, ddof = -1)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = std_all_views(as_view(empty), ddof = -1)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	empty_rows: Tensor(f64)
	empty_rows, err = new_tensor(f64, []int{2, 0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_rows)
	_, err = var_axis(empty_rows, 1, ddof = -1)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	empty_output: Tensor(f64)
	empty_output, err = new_tensor(f64, []int{2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_output)
	testing.expect_value(t, std_axis_into(&empty_output, empty_rows, 1, -1), Tensor_Error.Empty_Reduction)

	_, err = var_axis(grid, 2)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
}

@(test)
test_reductions_over_several_axes :: proc(t: ^testing.T) {
	// values[i,j,k] = 12i + 4j + k over shape [2,3,4].
	counting: [24]f64
	for i in 0..<len(counting) {
		counting[i] = f64(i)
	}
	values, err := from_slice(f64, counting[:], []int{2, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&values)

	// Over axes {0,2}: sum_{i,k} (12i + 4j + k) = 32j + 60.
	sums: Tensor(f64)
	sums, err = sum_axes(values, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&sums)
	expect_shape(t, sums.shape, []int{3})
	expect_f64_data(t, sums.data, []f64{60, 92, 124})
	inner: Tensor(f64)
	inner, err = sum_axis(values, 2)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&inner)
	nested: Tensor(f64)
	nested, err = sum_axis(inner, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nested)
	expect_f64_data(t, sums.data, nested.data)

	means: Tensor(f64)
	means, err = mean_axes(values, []int{2, 0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&means)
	expect_f64_data(t, means.data, []f64{7.5, 11.5, 15.5})
	largest: Tensor(f64)
	largest, err = max_axes(values, []int{0, 2}, true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&largest)
	expect_shape(t, largest.shape, []int{1, 3, 1})
	expect_f64_data(t, largest.data, []f64{15, 19, 23})
	smallest: Tensor(f64)
	smallest, err = min_axes(values, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&smallest)
	expect_f64_data(t, smallest.data, []f64{0, 4, 8})

	// A negative axis counts from the end, so {-1} is {2}.
	last: Tensor(f64)
	last, err = sum_axes(values, []int{-1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&last)
	third: Tensor(f64)
	third, err = sum_axes(values, []int{2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&third)
	expect_shape(t, last.shape, third.shape)
	expect_f64_data(t, last.data, third.data)
	expect_f64_data(t, last.data, inner.data)

	// Every axis matches the whole-tensor reduction; no axes copies the source.
	everything: Tensor(f64)
	everything, err = sum_axes(values, []int{2, 1, 0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&everything)
	expect_shape(t, everything.shape, []int{})
	expect_f64_data(t, everything.data, []f64{276})
	nothing: Tensor(f64)
	nothing, err = max_axes(values, []int{})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nothing)
	expect_shape(t, nothing.shape, []int{2, 3, 4})
	expect_f64_data(t, nothing.data, counting[:])

	// Repeated and out-of-range axes are errors.
	_, err = sum_axes(values, []int{0, 0})
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = mean_axes(values, []int{2, -1})
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = max_axes(values, []int{0, 3})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = min_axes_views(as_view(values), []int{-4})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	// Every axis is bounds-checked before repeats, so the order does not matter.
	_, err = sum_axes(values, []int{0, 0, 5})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = sum_axes(values, []int{5, 0, 0})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)

	// A permuted view reduces along its own axes: view[k,i,j] = values[i,j,k].
	permuted_shape, permuted_strides := [3]int{}, [3]int{}
	permuted, view_err := permute_view(as_view(values), []int{2, 0, 1}, permuted_shape[:], permuted_strides[:])
	testing.expect_value(t, view_err, Tensor_Error.None)
	permuted_sums: Tensor(f64)
	permuted_sums, err = sum_axes_views(permuted, []int{0, 1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&permuted_sums)
	expect_f64_data(t, permuted_sums.data, []f64{60, 92, 124})
	permuted_max: Tensor(f64)
	permuted_max, err = max_axes_views(permuted, []int{1, 2}, true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&permuted_max)
	expect_shape(t, permuted_max.shape, []int{4, 1, 1})
	expect_f64_data(t, permuted_max.data, []f64{20, 21, 22, 23})
	permuted_mean: Tensor(f64)
	permuted_mean, err = mean_axes_views(permuted, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&permuted_mean)
	expect_f64_data(t, permuted_mean.data, []f64{5.5, 17.5})

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{3})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&output)
	testing.expect_value(t, sum_axes_into(&output, values, []int{0, -1}), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{60, 92, 124})
	testing.expect_value(t, mean_axes_into(&output, values, []int{0, 2}), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{7.5, 11.5, 15.5})
	testing.expect_value(t, min_axes_views_into(&output, permuted, []int{0, 1}), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{0, 4, 8})
	testing.expect_value(t, max_axes_views_into(&output, permuted, []int{1, 0}), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{15, 19, 23})
	kept: Tensor(f64)
	kept, err = new_tensor(f64, []int{1, 3, 1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&kept)
	testing.expect_value(t, max_axes_into(&kept, values, []int{0, 2}, true), Tensor_Error.None)
	expect_f64_data(t, kept.data, []f64{15, 19, 23})
	copied: Tensor(f64)
	copied, err = new_tensor(f64, []int{2, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&copied)
	testing.expect_value(t, mean_axes_into(&copied, values, []int{}), Tensor_Error.None)
	expect_f64_data(t, copied.data, counting[:])

	// Wrong output shapes, bad axes, and overlapping outputs are rejected without writing.
	testing.expect_value(t, sum_axes_into(&kept, values, []int{0, 2}), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, sum_axes_into(&output, values, []int{0, 2}, true), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, sum_axes_into(&output, values, []int{1}), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, sum_axes_into(&output, values, []int{0, 0}), Tensor_Error.Invalid_Shape)
	testing.expect_value(t, sum_axes_views_into(&output, permuted, []int{3}), Tensor_Error.Out_Of_Bounds)
	folded := Tensor_View(f64){data = output.data, shape = []int{3, 1, 1}, strides = []int{1, 1, 1}}
	testing.expect_value(t, sum_axes_views_into(&output, folded, []int{1, 2}), Tensor_Error.Aliased_Output)
	expect_f64_data(t, output.data, []f64{15, 19, 23})

	// Empty reductions follow the single-axis forms.
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0, 3})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty)
	empty_sums: Tensor(f64)
	empty_sums, err = sum_axes(empty, []int{1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_sums)
	expect_shape(t, empty_sums.shape, []int{2, 3})
	expect_f64_data(t, empty_sums.data, []f64{0, 0, 0, 0, 0, 0})
	_, err = mean_axes(empty, []int{0, 1})
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = max_axes_views(as_view(empty), []int{1})
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, min_axes_into(&output, empty, []int{0, 1}), Tensor_Error.Empty_Reduction)
	nothing_kept: Tensor(f64)
	nothing_kept, err = max_axes(empty, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nothing_kept)
	expect_shape(t, nothing_kept.shape, []int{0})
	empty_products: Tensor(f64)
	empty_products, err = prod_axes(empty, []int{1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_products)
	expect_f64_data(t, empty_products.data, []f64{1, 1, 1, 1, 1, 1})

	// A rank-zero source has no axes to reduce.
	scalar: Tensor(f64)
	scalar, err = from_slice(f64, []f64{5}, []int{})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&scalar)
	scalar_copy: Tensor(f64)
	scalar_copy, err = sum_axes(scalar, []int{})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&scalar_copy)
	expect_shape(t, scalar_copy.shape, []int{})
	expect_f64_data(t, scalar_copy.data, []f64{5})
	_, err = sum_axes(scalar, []int{0})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = max_axes(scalar, []int{-1})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)

	// A NaN anywhere in a block wins for max and min, including across rows.
	for position in 0..<4 {
		grid := []f64{1, 2, 3, 4}
		grid[position] = math.nan_f64()
		source: Tensor(f64)
		source, err = from_slice(f64, grid, []int{2, 2})
		testing.expect_value(t, err, Tensor_Error.None)
		largest_nan: Tensor(f64)
		largest_nan, err = max_axes(source, []int{0, 1})
		testing.expect_value(t, err, Tensor_Error.None)
		testing.expectf(t, math.is_nan(largest_nan.data[0]), "max_axes with NaN at %d gave %v", position, largest_nan.data[0])
		smallest_nan: Tensor(f64)
		smallest_nan, err = min_axes(source, []int{1, 0})
		testing.expect_value(t, err, Tensor_Error.None)
		testing.expectf(t, math.is_nan(smallest_nan.data[0]), "min_axes with NaN at %d gave %v", position, smallest_nan.data[0])
		release(&largest_nan)
		release(&smallest_nan)
		release(&source)
	}

	// Integer sums accumulate in the element type.
	integers: Tensor(i32)
	integers, err = from_slice(i32, []i32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integers)
	integer_sums: Tensor(i32)
	integer_sums, err = sum_axes(integers, []int{0, 1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_sums)
	expect_i32_data(t, integer_sums.data, []i32{16, 20})
	integer_products: Tensor(i32)
	integer_products, err = prod_axes(integers, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_products)
	expect_i32_data(t, integer_products.data, []i32{60, 672})
	integer_max: Tensor(i32)
	integer_max, err = max_axes(integers, []int{0, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_max)
	expect_i32_data(t, integer_max.data, []i32{6, 8})
	integer_min: Tensor(i32)
	integer_min, err = min_axes(integers, []int{2, 0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_min)
	expect_i32_data(t, integer_min.data, []i32{1, 3})

	// prod_axes has the same _views, _into, and _views_into forms.
	product_view: Tensor(i32)
	product_view, err = prod_axes_views(as_view(integers), []int{1, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&product_view)
	expect_i32_data(t, product_view.data, []i32{24, 1680})
	product_output: Tensor(i32)
	product_output, err = new_tensor(i32, []int{2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&product_output)
	testing.expect_value(t, prod_axes_into(&product_output, integers, []int{0, 1}), Tensor_Error.None)
	expect_i32_data(t, product_output.data, []i32{105, 384})
	testing.expect_value(t, prod_axes_views_into(&product_output, as_view(integers), []int{-1, 0}), Tensor_Error.None)
	expect_i32_data(t, product_output.data, []i32{60, 672})
}

@(test)
test_dot_vector_and_matrix_semantics :: proc(t: ^testing.T) {
	vector, err := from_slice(f32, []f32{1, 2, 3}, []int{3})
	testing.expect(t, err == .None)
	defer release(&vector)
	other_vector: Tensor(f32)
	other_vector, err = from_slice(f32, []f32{4, 5, 6}, []int{3})
	testing.expect(t, err == .None)
	defer release(&other_vector)

	scalar_dot: Tensor(f32)
	scalar_dot, err = dot(vector, other_vector)
	testing.expect(t, err == .None)
	defer release(&scalar_dot)
	testing.expect(t, len(scalar_dot.shape) == 0 && scalar_dot.data[0] == 32)

	left_matrix: Tensor(f32)
	left_matrix, err = from_slice(f32, []f32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&left_matrix)
	matrix_vector: Tensor(f32)
	matrix_vector, err = dot(left_matrix, vector)
	testing.expect(t, err == .None)
	defer release(&matrix_vector)
	testing.expect(t, len(matrix_vector.shape) == 1 && matrix_vector.shape[0] == 2)
	testing.expect(t, matrix_vector.data[0] == 14 && matrix_vector.data[1] == 32)

	other_matrix: Tensor(f32)
	other_matrix, err = from_slice(f32, []f32{1, 4, 2, 5, 3, 6}, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&other_matrix)
	vector_matrix: Tensor(f32)
	vector_matrix, err = dot(vector, other_matrix)
	testing.expect(t, err == .None)
	defer release(&vector_matrix)
	testing.expect(t, len(vector_matrix.shape) == 1 && vector_matrix.shape[0] == 2)
	testing.expect(t, vector_matrix.data[0] == 14 && vector_matrix.data[1] == 32)

	matrix_product: Tensor(f32)
	matrix_product, err = dot(left_matrix, other_matrix)
	testing.expect(t, err == .None)
	defer release(&matrix_product)
	testing.expect(t, len(matrix_product.shape) == 2)
	testing.expect(t, matrix_product.data[0] == 14 && matrix_product.data[1] == 32)
	testing.expect(t, matrix_product.data[2] == 32 && matrix_product.data[3] == 77)

	scalar_output: Tensor(f32)
	scalar_output, err = new_tensor(f32, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar_output)
	testing.expect(t, dot_into(&scalar_output, vector, other_vector) == .None)
	testing.expect(t, scalar_output.data[0] == 32)

	shape_buffer := [1]int{}
	stride_buffer := [1]int{}
	reversed, view_err := slice_view(as_view(vector), 0, 2, 3, -1, shape_buffer[:], stride_buffer[:])
	testing.expect(t, view_err == .None)
	reversed_dot: Tensor(f32)
	reversed_dot, err = dot_views(reversed, reversed)
	testing.expect(t, err == .None)
	defer release(&reversed_dot)
	testing.expect(t, reversed_dot.data[0] == 14)
}

@(test)
test_dot_rejects_incompatible_dimensions :: proc(t: ^testing.T) {
	left, err := new_tensor(f64, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(f64)
	right, err = new_tensor(f64, []int{4})
	testing.expect(t, err == .None)
	defer release(&right)
	_, err = dot(left, right)
	testing.expect(t, err == .Shape_Mismatch)

	invalid_rank: Tensor(f64)
	invalid_rank, err = new_tensor(f64, []int{1, 2, 3})
	testing.expect(t, err == .None)
	defer release(&invalid_rank)
	_, err = dot(invalid_rank, left)
	testing.expect(t, err == .Invalid_Shape)
}

@(test)
test_matrix_multiplication_and_strided_views :: proc(t: ^testing.T) {
	left, err := from_slice(f32, []f32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(f32)
	right, err = from_slice(f32, []f32{7, 8, 9, 10, 11, 12}, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&right)

	product: Tensor(f32)
	product, err = matmul(left, right)
	testing.expect(t, err == .None)
	defer release(&product)
	testing.expect(t, product.shape[0] == 2 && product.shape[1] == 2)
	testing.expect(t, product.data[0] == 58 && product.data[1] == 64)
	testing.expect(t, product.data[2] == 139 && product.data[3] == 154)

	output: Tensor(f32)
	output, err = new_tensor(f32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	fill(&output, 999)
	testing.expect(t, matmul_into(&output, left, right) == .None)
	testing.expect(t, output.data[0] == 58 && output.data[1] == 64)
	testing.expect(t, output.data[2] == 139 && output.data[3] == 154)

	// Transposing both source matrices exercises non-contiguous input strides.
	left_transpose_storage, left_transpose_err := from_slice(f32, []f32{1, 4, 2, 5, 3, 6}, []int{3, 2})
	testing.expect(t, left_transpose_err == .None)
	defer release(&left_transpose_storage)
	right_transpose_storage, right_transpose_err := from_slice(f32, []f32{7, 9, 11, 8, 10, 12}, []int{2, 3})
	testing.expect(t, right_transpose_err == .None)
	defer release(&right_transpose_storage)
	left_shape_buffer := [2]int{}
	left_stride_buffer := [2]int{}
	left_transposed, left_view_err := transpose_view(as_view(left_transpose_storage), left_shape_buffer[:], left_stride_buffer[:])
	testing.expect(t, left_view_err == .None)
	right_shape_buffer := [2]int{}
	right_stride_buffer := [2]int{}
	right_transposed, right_view_err := transpose_view(as_view(right_transpose_storage), right_shape_buffer[:], right_stride_buffer[:])
	testing.expect(t, right_view_err == .None)
	view_product: Tensor(f32)
	view_product, err = matmul_views(left_transposed, right_transposed)
	testing.expect(t, err == .None)
	defer release(&view_product)
	testing.expect(t, view_product.data[0] == 58 && view_product.data[1] == 64)
	testing.expect(t, view_product.data[2] == 139 && view_product.data[3] == 154)
	testing.expect(t, matmul_views_into(&output, left_transposed, right_transposed) == .None)
	testing.expect(t, output.data[0] == 58 && output.data[1] == 64)
	testing.expect(t, output.data[2] == 139 && output.data[3] == 154)
}

@(test)
test_matrix_multiplication_errors_and_zero_inner_dimension :: proc(t: ^testing.T) {
	left, err := new_tensor(f64, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(f64)
	right, err = new_tensor(f64, []int{4, 2})
	testing.expect(t, err == .None)
	defer release(&right)
	_, err = matmul(left, right)
	testing.expect(t, err == .Shape_Mismatch)

	vector: Tensor(f64)
	vector, err = new_tensor(f64, []int{3})
	testing.expect(t, err == .None)
	defer release(&vector)
	// As in NumPy, matmul of two vectors is their inner product.
	inner: Tensor(f64)
	inner, err = matmul(vector, vector)
	testing.expect(t, err == .None && len(inner.shape) == 0)
	release(&inner)

	wrong_output: Tensor(f64)
	wrong_output, err = new_tensor(f64, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&wrong_output)
	matching_right: Tensor(f64)
	matching_right, err = new_tensor(f64, []int{3, 1})
	testing.expect(t, err == .None)
	defer release(&matching_right)
	testing.expect(t, matmul_into(&wrong_output, left, matching_right) == .Shape_Mismatch)

	empty_left: Tensor(f32)
	empty_left, err = new_tensor(f32, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty_left)
	empty_right: Tensor(f32)
	empty_right, err = new_tensor(f32, []int{0, 3})
	testing.expect(t, err == .None)
	defer release(&empty_right)
	empty_product: Tensor(f32)
	empty_product, err = matmul(empty_left, empty_right)
	testing.expect(t, err == .None)
	defer release(&empty_product)
	testing.expect(t, empty_product.shape[0] == 2 && empty_product.shape[1] == 3)
	for value in empty_product.data {
		testing.expect(t, value == 0)
	}
}

// expect_batch_matches_matmul checks that output[batch] equals the 2-D matmul
// of a[a_batch] and b[b_batch], each a contiguous row-major matrix.
expect_batch_matches_matmul :: proc(t: ^testing.T, output: Tensor(f64), batch: int, a: Tensor(f64), a_batch: int, b: Tensor(f64), b_batch: int, loc := #caller_location) {
	m, k, n := a.shape[len(a.shape)-2], a.shape[len(a.shape)-1], b.shape[len(b.shape)-1]
	a_matrix, err := from_slice(f64, a.data[a_batch*m*k:][:m*k], []int{m, k})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&a_matrix)
	b_matrix: Tensor(f64)
	b_matrix, err = from_slice(f64, b.data[b_batch*k*n:][:k*n], []int{k, n})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&b_matrix)
	expected: Tensor(f64)
	expected, err = matmul(a_matrix, b_matrix)
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&expected)
	expect_f64_data(t, output.data[batch*m*n:][:m*n], expected.data, loc = loc)
}

// counting_values returns 1, 2, ..., count scaled by factor, for test operands.
counting_values :: proc(count: int, factor: f64) -> []f64 {
	values := make([]f64, count)
	for &value, i in values {
		value = f64(i+1) * factor
	}
	return values
}

@(test)
test_batched_matmul_stacks_matrices :: proc(t: ^testing.T) {
	a_values := counting_values(24, 1)
	defer delete(a_values)
	b_values := counting_values(20, 0.5)
	defer delete(b_values)
	a, err := from_slice(f64, a_values, []int{2, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&a)
	b: Tensor(f64)
	b, err = from_slice(f64, b_values, []int{4, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&b)

	product: Tensor(f64)
	product, err = matmul(a, b)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&product)
	expect_shape(t, product.shape, []int{2, 3, 5})
	expect_batch_matches_matmul(t, product, 0, a, 0, b, 0)
	expect_batch_matches_matmul(t, product, 1, a, 1, b, 0)

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{2, 3, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&output)
	testing.expect_value(t, matmul_into(&output, a, b), Tensor_Error.None)
	expect_f64_data(t, output.data, product.data)

	// A vector operand is promoted and its axis dropped, as in NumPy.
	row: Tensor(f64)
	row, err = from_slice(f64, []f64{1, 0, -1, 2}, []int{4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&row)
	row_products: Tensor(f64)
	row_products, err = matmul(a, row)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&row_products)
	expect_shape(t, row_products.shape, []int{2, 3})
	expect_f64_data(t, row_products.data, []f64{6, 14, 22, 30, 38, 46})
	column: Tensor(f64)
	column, err = from_slice(f64, []f64{1, 0, -1}, []int{3})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&column)
	column_products: Tensor(f64)
	column_products, err = matmul(column, a)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&column_products)
	expect_shape(t, column_products.shape, []int{2, 4})
	expect_f64_data(t, column_products.data, []f64{-8, -8, -8, -8, -8, -8, -8, -8})
}

@(test)
test_batched_matmul_broadcasts_batch_axes :: proc(t: ^testing.T) {
	a_values := counting_values(24, 1)
	defer delete(a_values)
	b_values := counting_values(40, -0.25)
	defer delete(b_values)
	a, err := from_slice(f64, a_values, []int{2, 1, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&a)
	b: Tensor(f64)
	b, err = from_slice(f64, b_values, []int{5, 4, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&b)

	product: Tensor(f64)
	product, err = matmul(a, b)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&product)
	expect_shape(t, product.shape, []int{2, 5, 3, 2})
	for i in 0..<2 {
		for j in 0..<5 {
			expect_batch_matches_matmul(t, product, i*5+j, a, i, b, j)
		}
	}

	// The same values as b through a strided view: copy b with its first two
	// axes swapped, then swap them back in a view.
	shape_buffer, strides_buffer: [3]int
	swapped: Tensor_View(f64)
	swapped, err = permute_view(as_view(b), []int{1, 0, 2}, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.None)
	swapped_copy: Tensor(f64)
	swapped_copy, err = copy_view(swapped)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&swapped_copy)
	restored_shape, restored_strides: [3]int
	restored: Tensor_View(f64)
	restored, err = permute_view(as_view(swapped_copy), []int{1, 0, 2}, restored_shape[:], restored_strides[:])
	testing.expect_value(t, err, Tensor_Error.None)
	testing.expect(t, !_is_contiguous_view(restored), "the restored view should be strided")
	view_product: Tensor(f64)
	view_product, err = matmul_views(as_view(a), restored)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&view_product)
	expect_f64_data(t, view_product.data, product.data)
}

@(test)
test_batched_matmul_edge_cases :: proc(t: ^testing.T) {
	// k = 0: every batch is a sum over nothing, so the output must be zeroed
	// even when it starts out holding other values.
	a, err := new_tensor(f64, []int{2, 3, 0})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&a)
	b: Tensor(f64)
	b, err = new_tensor(f64, []int{0, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&b)
	output: Tensor(f64)
	output, err = new_tensor(f64, []int{2, 3, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&output)
	for &value in output.data {
		value = 7
	}
	testing.expect_value(t, matmul_into(&output, a, b), Tensor_Error.None)
	for value in output.data {
		testing.expect_value(t, value, 0)
	}

	// Integers, with a size-1 batch axis on each side: [3,1,1,2] @ [4,2,1] is
	// [3,4,1,1], and element [i,j] is row i of a times column j of b.
	ints_a: Tensor(int)
	ints_a, err = from_slice(int, []int{1, 2, 3, 4, 5, 6}, []int{3, 1, 1, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&ints_a)
	ints_b: Tensor(int)
	ints_b, err = from_slice(int, []int{1, 0, 0, 1, 1, 1, 2, -1}, []int{4, 2, 1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&ints_b)
	ints_product: Tensor(int)
	ints_product, err = matmul(ints_a, ints_b)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&ints_product)
	expect_shape(t, ints_product.shape, []int{3, 4, 1, 1})
	expect_int_data(t, ints_product.data, []int{1, 2, 3, 0, 3, 4, 7, 2, 5, 6, 11, 4})

	// A batch axis from broadcast_view has zero stride, so every batch reads
	// the same matrix.
	base: Tensor(f64)
	base, err = from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&base)
	shape_buffer, strides_buffer: [3]int
	repeated: Tensor_View(f64)
	repeated, err = broadcast_view(as_view(base), []int{3, 2, 2}, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.None)
	stack_values := counting_values(12, 1)
	defer delete(stack_values)
	stack: Tensor(f64)
	stack, err = from_slice(f64, stack_values, []int{3, 2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&stack)
	repeated_product: Tensor(f64)
	repeated_product, err = matmul_views(repeated, as_view(stack))
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&repeated_product)
	expect_shape(t, repeated_product.shape, []int{3, 2, 2})
	for batch in 0..<3 {
		expect_batch_matches_matmul(t, repeated_product, batch, base, 0, stack, batch)
	}
}

// fill_product_operand fills data with operand values. Float values are not
// small integers, so their sums round differently when the products are added
// in another order; integer types get small values.
fill_product_operand :: proc(data: []$T, seed: int) {
	for &value, i in data {
		when intrinsics.type_is_float(T) {
			value = T(0.1*f64((i+seed)%7) + 1e-3*f64(i) - 0.3)
		} else {
			value = T((i+seed)%7 - 3)
		}
	}
}

// naive_product_element sums a[i, :] * b[:, j] from zero in index order, the
// order matmul keeps, treating a rank-one a as a row and a rank-one b as a
// column.
naive_product_element :: proc(a, b: Tensor_View($T), i, j: int) -> (sum: T) {
	a_row_stride := a.strides[0] if len(a.shape) == 2 else 0
	a_inner_stride := a.strides[len(a.shape)-1]
	b_column_stride := b.strides[1] if len(b.shape) == 2 else 0
	for p in 0..<a.shape[len(a.shape)-1] {
		sum += a.data[a.offset + i*a_row_stride + p*a_inner_stride] * b.data[b.offset + p*b.strides[0] + j*b_column_stride]
	}
	return
}

// expect_product_matches_naive compares output, the m×n product of a and b,
// with naive_product_element. The comparison is exact, so a kernel that adds
// float products in another order fails it.
expect_product_matches_naive :: proc(t: ^testing.T, output: []$T, a, b: Tensor_View(T), loc := #caller_location) {
	m := a.shape[0] if len(a.shape) == 2 else 1
	n := b.shape[1] if len(b.shape) == 2 else 1
	k := a.shape[len(a.shape)-1]
	mismatches := 0
	for i in 0..<m {
		for j in 0..<n {
			if output[i*n + j] != naive_product_element(a, b, i, j) {
				mismatches += 1
			}
		}
	}
	testing.expectf(t, mismatches == 0, "%v %v @ %v (k=%d, a strides %v, b strides %v): %d elements differ from the naive product", typeid_of(T), a.shape, b.shape, k, a.strides, b.strides, mismatches, loc = loc)
}

// expect_matmul_matches_naive multiplies an m×k by a k×n matrix and compares
// the result exactly with the naive product. With transpose_a or transpose_b
// that operand is a transposed view, so its rows are not contiguous.
expect_matmul_matches_naive :: proc(t: ^testing.T, $T: typeid, m, k, n: int, transpose_a, transpose_b: bool, loc := #caller_location) {
	a_storage, err := new_tensor(T, []int{k, m} if transpose_a else []int{m, k})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&a_storage)
	fill_product_operand(a_storage.data, 0)
	b_storage: Tensor(T)
	b_storage, err = new_tensor(T, []int{n, k} if transpose_b else []int{k, n})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&b_storage)
	fill_product_operand(b_storage.data, 3)
	a, b := as_view(a_storage), as_view(b_storage)
	a_shape, a_strides, b_shape, b_strides: [2]int
	if transpose_a {
		a, err = transpose_view(a, a_shape[:], a_strides[:])
		testing.expect_value(t, err, Tensor_Error.None, loc)
	}
	if transpose_b {
		b, err = transpose_view(b, b_shape[:], b_strides[:])
		testing.expect_value(t, err, Tensor_Error.None, loc)
	}

	product: Tensor(T)
	product, err = matmul_views(a, b)
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&product)
	expect_shape(t, product.shape, []int{m, n})
	expect_product_matches_naive(t, product.data, a, b, loc = loc)
}

// expect_vector_products_match_naive multiplies a k-vector by a k×n matrix and
// an n×k matrix by a k-vector, comparing both exactly with the naive product.
expect_vector_products_match_naive :: proc(t: ^testing.T, $T: typeid, k, n: int, loc := #caller_location) {
	vector, err := new_tensor(T, []int{k})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&vector)
	fill_product_operand(vector.data, 0)
	right: Tensor(T)
	right, err = new_tensor(T, []int{k, n})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&right)
	fill_product_operand(right.data, 3)
	left: Tensor(T)
	left, err = new_tensor(T, []int{n, k})
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&left)
	fill_product_operand(left.data, 5)

	row_product: Tensor(T)
	row_product, err = matmul(vector, right)
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&row_product)
	expect_shape(t, row_product.shape, []int{n})
	expect_product_matches_naive(t, row_product.data, as_view(vector), as_view(right), loc = loc)
	column_product: Tensor(T)
	column_product, err = matmul(left, vector)
	testing.expect_value(t, err, Tensor_Error.None, loc)
	defer release(&column_product)
	expect_shape(t, column_product.shape, []int{n})
	expect_product_matches_naive(t, column_product.data, as_view(left), as_view(vector), loc = loc)
}

@(test)
test_matmul_blocking_matches_naive_product :: proc(t: ^testing.T) {
	// Sizes straddle the register tiles and the k blocks, leaving partial row
	// and column tiles and a partial last block.
	expect_matmul_matches_naive(t, f32, 5, 300, 37, false, false)
	expect_matmul_matches_naive(t, f32, 5, 300, 37, false, true)
	expect_matmul_matches_naive(t, f32, 5, 300, 37, true, false)
	expect_matmul_matches_naive(t, f32, 6, 600, 33, true, true)
	expect_matmul_matches_naive(t, f64, 7, 513, 19, false, false)
	expect_matmul_matches_naive(t, f64, 3, 513, 7, false, false)
	expect_matmul_matches_naive(t, i64, 4, 256, 16, false, true)
	expect_matmul_matches_naive(t, i8, 3, 9, 70, false, false)
	expect_matmul_matches_naive(t, f32, 1, 300, 37, false, false)
	expect_matmul_matches_naive(t, f32, 6, 300, 1, false, false)
	expect_matmul_matches_naive(t, f32, 8, 3, 40, false, false)
	expect_vector_products_match_naive(t, f32, 600, 37)
	expect_vector_products_match_naive(t, f64, 257, 8)
}

@(test)
test_batched_matmul_blocking_matches_naive_product :: proc(t: ^testing.T) {
	m, k, n :: 3, 600, 21
	a, err := new_tensor(f32, []int{2, m, k})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&a)
	fill_product_operand(a.data, 0)
	b: Tensor(f32)
	b, err = new_tensor(f32, []int{k, n})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&b)
	fill_product_operand(b.data, 3)

	product: Tensor(f32)
	product, err = matmul(a, b)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&product)
	expect_shape(t, product.shape, []int{2, m, n})
	a_shape, a_strides := [2]int{m, k}, [2]int{k, 1}
	for batch in 0..<2 {
		a_matrix := Tensor_View(f32){data = a.data, shape = a_shape[:], strides = a_strides[:], offset = batch*m*k}
		expect_product_matches_naive(t, product.data[batch*m*n:][:m*n], a_matrix, as_view(b))
	}
}

@(test)
test_batched_matmul_errors_and_aliasing :: proc(t: ^testing.T) {
	a, err := new_tensor(f64, []int{2, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&a)
	b: Tensor(f64)
	b, err = new_tensor(f64, []int{3, 4, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&b)
	_, err = matmul(a, b)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	output: Tensor(f64)
	output, err = new_tensor(f64, []int{3, 3, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&output)
	testing.expect_value(t, matmul_into(&output, a, b), Tensor_Error.Shape_Mismatch)

	wrong_inner: Tensor(f64)
	wrong_inner, err = new_tensor(f64, []int{2, 3, 5})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&wrong_inner)
	_, err = matmul(a, wrong_inner)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)

	// An empty batch axis gives an empty result, and a batch axis of one
	// broadcasts against it.
	empty_batch: Tensor(f64)
	empty_batch, err = new_tensor(f64, []int{0, 4, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_batch)
	single_batch: Tensor(f64)
	single_batch, err = new_tensor(f64, []int{1, 3, 4})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&single_batch)
	empty_product: Tensor(f64)
	empty_product, err = matmul(single_batch, empty_batch)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_product)
	expect_shape(t, empty_product.shape, []int{0, 3, 2})
	testing.expect_value(t, len(empty_product.data), 0)

	// Batched outputs follow the same overlap rule as 2-D products: any input
	// that shares memory with the output is rejected and the output kept.
	stacked_values := counting_values(8, 1)
	defer delete(stacked_values)
	stacked: Tensor(f64)
	stacked, err = from_slice(f64, stacked_values, []int{2, 2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&stacked)
	identity: Tensor(f64)
	identity, err = from_slice(f64, []f64{1, 0, 0, 1}, []int{2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&identity)
	testing.expect_value(t, matmul_into(&stacked, stacked, identity), Tensor_Error.Aliased_Output)
	second_matrix := Tensor_View(f64){data = stacked.data, shape = []int{2, 2}, strides = []int{2, 1}, offset = 4}
	testing.expect_value(t, matmul_views_into(&stacked, as_view(stacked), second_matrix), Tensor_Error.Aliased_Output)
	testing.expect_value(t, matmul_views_into(&stacked, as_view(identity), as_view(stacked)), Tensor_Error.Aliased_Output)
	expect_f64_data(t, stacked.data, stacked_values)
	separate: Tensor(f64)
	separate, err = new_tensor(f64, []int{2, 2, 2})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&separate)
	testing.expect_value(t, matmul_into(&separate, stacked, identity), Tensor_Error.None)
	expect_f64_data(t, separate.data, stacked_values)
}

@(test)
test_create_reshape_and_add :: proc(t: ^testing.T) {
	shape := []int{2, 2}
	left, err := from_slice(i32, []i32{1, 2, 3, 4}, shape)
	testing.expect(t, err == .None)
	defer release(&left)

	right: Tensor(i32)
	right, err = new_tensor(i32, shape)
	testing.expect(t, err == .None)
	defer release(&right)
	fill(&right, 10)

	sum: Tensor(i32)
	sum, err = add(left, right)
	testing.expect(t, err == .None)
	defer release(&sum)
	testing.expect(t, sum.shape[0] == 2 && sum.shape[1] == 2)
	testing.expect(t, sum.data[0] == 11 && sum.data[3] == 14)
	value, access_err := get_at(sum, []int{1, 0})
	testing.expect(t, access_err == .None && value == 13)
	testing.expect(t, set_at(&sum, []int{0, 1}, 50) == .None)
	testing.expect(t, sum.data[1] == 50)

	new_shape := []int{4}
	testing.expect(t, reshape(&sum, new_shape) == .None)
	testing.expect(t, sum.shape[0] == 4 && len(sum.data) == 4)
	testing.expect(t, reshape(&sum, []int{3}) == .Shape_Mismatch)
}

@(test)
test_transpose_copy_owns_reversed_axis_result :: proc(t: ^testing.T) {
	owner, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&owner)

	copied, copy_err := transpose_copy_tensor(owner)
	testing.expect(t, copy_err == .None)
	defer release(&copied)
	testing.expect(t, _shapes_match(copied.shape, []int{3, 2}))
	testing.expect(t, copied.data[0] == 1 && copied.data[1] == 4)
	testing.expect(t, copied.data[2] == 2 && copied.data[3] == 5)
	testing.expect(t, copied.data[4] == 3 && copied.data[5] == 6)

	copied.data[0] = 99
	testing.expect(t, owner.data[0] == 1)

	reversed_shape := [2]int{}
	reversed_strides := [2]int{}
	reversed, reverse_err := slice_view(as_view(owner), 1, 2, 3, -1, reversed_shape[:], reversed_strides[:])
	testing.expect(t, reverse_err == .None)
	transposed, transposed_err := transpose_copy(reversed)
	testing.expect(t, transposed_err == .None)
	defer release(&transposed)
	testing.expect(t, transposed.data[0] == 3 && transposed.data[1] == 6)
	testing.expect(t, transposed.data[2] == 2 && transposed.data[3] == 5)
	testing.expect(t, transposed.data[4] == 1 && transposed.data[5] == 4)
}

@(test)
test_optimizer_in_place_scaling_helpers :: proc(t: ^testing.T) {
	parameters, parameter_err := from_slice(f64, []f64{1, -2, 0.5}, []int{3})
	testing.expect(t, parameter_err == .None)
	defer release(&parameters)
	gradient, gradient_err := from_slice(f64, []f64{0.25, 2, -1}, []int{3})
	testing.expect(t, gradient_err == .None)
	defer release(&gradient)

	testing.expect(t, add_scaled_in_place(&parameters, gradient, -0.1) == .None)
	testing.expect(t, abs(parameters.data[0]-0.975) < 1e-12)
	testing.expect(t, abs(parameters.data[1]+2.2) < 1e-12)
	testing.expect(t, abs(parameters.data[2]-0.6) < 1e-12)
	scale_in_place(&parameters, 2)
	testing.expect(t, abs(parameters.data[0]-1.95) < 1e-12)
	testing.expect(t, abs(parameters.data[1]+4.4) < 1e-12)
	testing.expect(t, abs(parameters.data[2]-1.2) < 1e-12)

	wrong_shape, wrong_shape_err := new_tensor(f64, []int{1, 3})
	testing.expect(t, wrong_shape_err == .None)
	defer release(&wrong_shape)
	testing.expect(t, add_scaled_in_place(&parameters, wrong_shape, 1) == .Shape_Mismatch)
}

@(test)
test_transpose_borrowed_view :: proc(t: ^testing.T) {
	owner, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&owner)

	shape_buffer := [2]int{}
	strides_buffer := [2]int{}
	transposed: Tensor_View(i32)
	transposed, err = transpose_view(as_view(owner), shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	testing.expect(t, transposed.shape[0] == 3 && transposed.shape[1] == 2)

	value, access_err := get_at_view(transposed, []int{2, 1})
	testing.expect(t, access_err == .None && value == 6)
	testing.expect(t, set_at_view(transposed, []int{0, 1}, 40) == .None)
	testing.expect(t, owner.data[3] == 40)

	sum: Tensor(i32)
	sum, err = add_views(transposed, transposed)
	testing.expect(t, err == .None)
	defer release(&sum)
	testing.expect(t, sum.shape[0] == 3 && sum.shape[1] == 2)
	testing.expect(t, sum.data[1] == 80)
}

@(test)
test_slice_view_reverse_and_mutate :: proc(t: ^testing.T) {
	owner, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&owner)

	shape_buffer := [2]int{}
	strides_buffer := [2]int{}
	reversed: Tensor_View(i32)
	reversed, err = slice_view(as_view(owner), 1, 2, 3, -1, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	testing.expect(t, reversed.shape[1] == 3 && reversed.strides[1] == -1)

	value, access_err := get_at_view(reversed, []int{1, 0})
	testing.expect(t, access_err == .None && value == 6)
	testing.expect(t, set_at_view(reversed, []int{0, 0}, 30) == .None)
	testing.expect(t, owner.data[2] == 30)

	_, err = slice_view(as_view(owner), 1, 0, 4, 1, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .Out_Of_Bounds)
}

@(test)
test_broadcast_views_and_addition :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)
	row: Tensor(i32)
	row, err = from_slice(i32, []i32{10, 20, 30}, []int{1, 3})
	testing.expect(t, err == .None)
	defer release(&row)

	shape_buffer := [2]int{}
	strides_buffer := [2]int{}
	row_matrix: Tensor_View(i32)
	row_matrix, err = broadcast_view(as_view(row), []int{2, 3}, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	testing.expect(t, row_matrix.shape[0] == 2 && row_matrix.strides[0] == 0 && row_matrix.strides[1] == 1)
	value, access_err := get_at_view(row_matrix, []int{1, 2})
	testing.expect(t, access_err == .None && value == 30)

	sum: Tensor(i32)
	sum, err = add(values, row)
	testing.expect(t, err == .None)
	defer release(&sum)
	testing.expect(t, sum.shape[0] == 2 && sum.shape[1] == 3)
	testing.expect(t, sum.data[0] == 11 && sum.data[2] == 33 && sum.data[3] == 14 && sum.data[5] == 36)

	scalar: Tensor(i32)
	scalar, err = from_slice(i32, []i32{5}, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)
	scalar_sum: Tensor(i32)
	scalar_sum, err = add(values, scalar)
	testing.expect(t, err == .None)
	defer release(&scalar_sum)
	testing.expect(t, scalar_sum.data[0] == 6 && scalar_sum.data[5] == 11)

	product: Tensor(i32)
	product, err = mul(values, row)
	testing.expect(t, err == .None)
	defer release(&product)
	testing.expect(t, product.data[0] == 10 && product.data[2] == 90 && product.data[3] == 40 && product.data[5] == 180)

	testing.expect(t, add_in_place(&values, row) == .None)
	testing.expect(t, values.data[0] == 11 && values.data[2] == 33 && values.data[3] == 14 && values.data[5] == 36)
}

@(test)
test_all_binary_operations_have_matching_apis :: proc(t: ^testing.T) {
	left, err := from_slice(i32, []i32{20, 30, 40, 50}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&left)
	right: Tensor(i32)
	right, err = from_slice(i32, []i32{2, 5}, []int{1, 2})
	testing.expect(t, err == .None)
	defer release(&right)

	difference: Tensor(i32)
	difference, err = sub(left, right)
	testing.expect(t, err == .None)
	defer release(&difference)
	expect_i32_data(t, difference.data, []i32{18, 25, 38, 45})

	quotient: Tensor(i32)
	quotient, err = div(left, right)
	testing.expect(t, err == .None)
	defer release(&quotient)
	expect_i32_data(t, quotient.data, []i32{10, 6, 20, 10})

	output: Tensor(i32)
	output, err = new_tensor(i32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect(t, add_into(&output, left, right) == .None)
	expect_i32_data(t, output.data, []i32{22, 35, 42, 55})
	testing.expect(t, sub_into(&output, left, right) == .None)
	expect_i32_data(t, output.data, []i32{18, 25, 38, 45})
	testing.expect(t, mul_into(&output, left, right) == .None)
	expect_i32_data(t, output.data, []i32{40, 150, 80, 250})
	testing.expect(t, div_into(&output, left, right) == .None)
	expect_i32_data(t, output.data, []i32{10, 6, 20, 10})
	wrong_output: Tensor(i32)
	wrong_output, err = new_tensor(i32, []int{2})
	testing.expect(t, err == .None)
	defer release(&wrong_output)
	testing.expect(t, add_into(&wrong_output, left, right) == .Shape_Mismatch)

	left_view := as_view(left)
	right_view := as_view(right)
	view_sum: Tensor(i32)
	view_sum, err = add_views(left_view, right_view)
	testing.expect(t, err == .None)
	defer release(&view_sum)
	expect_i32_data(t, view_sum.data, []i32{22, 35, 42, 55})
	view_product: Tensor(i32)
	view_product, err = mul_views(left_view, right_view)
	testing.expect(t, err == .None)
	defer release(&view_product)
	expect_i32_data(t, view_product.data, []i32{40, 150, 80, 250})
	view_difference: Tensor(i32)
	view_difference, err = sub_views(left_view, right_view)
	testing.expect(t, err == .None)
	defer release(&view_difference)
	expect_i32_data(t, view_difference.data, []i32{18, 25, 38, 45})
	view_quotient: Tensor(i32)
	view_quotient, err = div_views(left_view, right_view)
	testing.expect(t, err == .None)
	defer release(&view_quotient)
	expect_i32_data(t, view_quotient.data, []i32{10, 6, 20, 10})

	in_place: Tensor(i32)
	in_place, err = from_slice(i32, []i32{20, 30, 40, 50}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&in_place)
	testing.expect(t, sub_in_place(&in_place, right) == .None)
	expect_i32_data(t, in_place.data, []i32{18, 25, 38, 45})
	testing.expect(t, add_in_place(&in_place, right) == .None)
	expect_i32_data(t, in_place.data, []i32{20, 30, 40, 50})
	testing.expect(t, mul_in_place(&in_place, right) == .None)
	expect_i32_data(t, in_place.data, []i32{40, 150, 80, 250})
	testing.expect(t, div_in_place(&in_place, right) == .None)
	expect_i32_data(t, in_place.data, []i32{20, 30, 40, 50})
}

@(test)
test_binary_operations_handle_scalars_empty_tensors_and_strided_views :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{8, 12, 16, 20}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&values)
	scalar: Tensor(i32)
	scalar, err = from_slice(i32, []i32{2}, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)

	sum: Tensor(i32)
	sum, err = add(values, scalar)
	testing.expect(t, err == .None)
	defer release(&sum)
	expect_i32_data(t, sum.data, []i32{10, 14, 18, 22})
	difference: Tensor(i32)
	difference, err = sub(values, scalar)
	testing.expect(t, err == .None)
	defer release(&difference)
	expect_i32_data(t, difference.data, []i32{6, 10, 14, 18})
	product: Tensor(i32)
	product, err = mul(values, scalar)
	testing.expect(t, err == .None)
	defer release(&product)
	expect_i32_data(t, product.data, []i32{16, 24, 32, 40})
	quotient: Tensor(i32)
	quotient, err = div(values, scalar)
	testing.expect(t, err == .None)
	defer release(&quotient)
	expect_i32_data(t, quotient.data, []i32{4, 6, 8, 10})

	output: Tensor(i32)
	output, err = new_tensor(i32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect(t, add_into(&output, values, scalar) == .None)
	expect_i32_data(t, output.data, []i32{10, 14, 18, 22})
	testing.expect(t, sub_into(&output, values, scalar) == .None)
	expect_i32_data(t, output.data, []i32{6, 10, 14, 18})
	testing.expect(t, mul_into(&output, values, scalar) == .None)
	expect_i32_data(t, output.data, []i32{16, 24, 32, 40})
	testing.expect(t, div_into(&output, values, scalar) == .None)
	expect_i32_data(t, output.data, []i32{4, 6, 8, 10})

	testing.expect(t, sub_in_place(&values, scalar) == .None)
	expect_i32_data(t, values.data, []i32{6, 10, 14, 18})
	testing.expect(t, add_in_place(&values, scalar) == .None)
	expect_i32_data(t, values.data, []i32{8, 12, 16, 20})
	testing.expect(t, mul_in_place(&values, scalar) == .None)
	expect_i32_data(t, values.data, []i32{16, 24, 32, 40})
	testing.expect(t, div_in_place(&values, scalar) == .None)
	expect_i32_data(t, values.data, []i32{8, 12, 16, 20})

	empty: Tensor(i32)
	empty, err = new_tensor(i32, []int{0, 3})
	testing.expect(t, err == .None)
	defer release(&empty)
	row: Tensor(i32)
	row, err = from_slice(i32, []i32{2, 3, 4}, []int{1, 3})
	testing.expect(t, err == .None)
	defer release(&row)
	empty_sum: Tensor(i32)
	empty_sum, err = add(empty, row)
	testing.expect(t, err == .None)
	defer release(&empty_sum)
	testing.expect(t, len(empty_sum.data) == 0 && empty_sum.shape[0] == 0 && empty_sum.shape[1] == 3)
	empty_difference: Tensor(i32)
	empty_difference, err = sub(empty, row)
	testing.expect(t, err == .None)
	defer release(&empty_difference)
	testing.expect(t, len(empty_difference.data) == 0)
	empty_product: Tensor(i32)
	empty_product, err = mul(empty, row)
	testing.expect(t, err == .None)
	defer release(&empty_product)
	testing.expect(t, len(empty_product.data) == 0)
	empty_quotient: Tensor(i32)
	empty_quotient, err = div(empty, row)
	testing.expect(t, err == .None)
	defer release(&empty_quotient)
	testing.expect(t, len(empty_quotient.data) == 0)
	empty_output: Tensor(i32)
	empty_output, err = new_tensor(i32, []int{0, 3})
	testing.expect(t, err == .None)
	defer release(&empty_output)
	testing.expect(t, add_into(&empty_output, empty, row) == .None)
	testing.expect(t, sub_into(&empty_output, empty, row) == .None)
	testing.expect(t, mul_into(&empty_output, empty, row) == .None)
	testing.expect(t, div_into(&empty_output, empty, row) == .None)
	testing.expect(t, add_in_place(&empty, row) == .None)
	testing.expect(t, sub_in_place(&empty, row) == .None)
	testing.expect(t, mul_in_place(&empty, row) == .None)
	testing.expect(t, div_in_place(&empty, row) == .None)

	source: Tensor(i32)
	source, err = from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&source)
	shape_buffer := [2]int{}
	strides_buffer := [2]int{}
	reversed: Tensor_View(i32)
	reversed, err = slice_view(as_view(source), 1, 2, 3, -1, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	scalar_view := as_view(scalar)
	view_sum: Tensor(i32)
	view_sum, err = add_views(reversed, scalar_view)
	testing.expect(t, err == .None)
	defer release(&view_sum)
	expect_i32_data(t, view_sum.data, []i32{5, 4, 3, 8, 7, 6})
	view_difference: Tensor(i32)
	view_difference, err = sub_views(reversed, scalar_view)
	testing.expect(t, err == .None)
	defer release(&view_difference)
	expect_i32_data(t, view_difference.data, []i32{1, 0, -1, 4, 3, 2})
	view_product: Tensor(i32)
	view_product, err = mul_views(reversed, scalar_view)
	testing.expect(t, err == .None)
	defer release(&view_product)
	expect_i32_data(t, view_product.data, []i32{6, 4, 2, 12, 10, 8})
	view_quotient: Tensor(i32)
	view_quotient, err = div_views(reversed, scalar_view)
	testing.expect(t, err == .None)
	defer release(&view_quotient)
	expect_i32_data(t, view_quotient.data, []i32{1, 1, 0, 3, 2, 2})
}

@(test)
test_division_by_zero_returns_an_error_without_writing :: proc(t: ^testing.T) {
	dividend, err := from_slice(i32, []i32{8, 12, 16, 20}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&dividend)
	divisors: Tensor(i32)
	divisors, err = from_slice(i32, []i32{2, 0}, []int{1, 2})
	testing.expect(t, err == .None)
	defer release(&divisors)

	_, err = div(dividend, divisors)
	testing.expect(t, err == .Division_By_Zero)
	_, err = div_views(as_view(dividend), as_view(divisors))
	testing.expect(t, err == .Division_By_Zero)

	output: Tensor(i32)
	output, err = from_slice(i32, []i32{99, 99, 99, 99}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect(t, div_into(&output, dividend, divisors) == .Division_By_Zero)
	expect_i32_data(t, output.data, []i32{99, 99, 99, 99})

	testing.expect(t, div_in_place(&dividend, divisors) == .Division_By_Zero)
	expect_i32_data(t, dividend.data, []i32{8, 12, 16, 20})
}

@(test)
test_all_binary_operations_reject_shape_mismatches :: proc(t: ^testing.T) {
	a, err := new_tensor(i32, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&a)
	b: Tensor(i32)
	b, err = new_tensor(i32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&b)
	output: Tensor(i32)
	output, err = new_tensor(i32, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&output)

	_, err = add(a, b)
	testing.expect(t, err == .Shape_Mismatch)
	_, err = sub(a, b)
	testing.expect(t, err == .Shape_Mismatch)
	_, err = mul(a, b)
	testing.expect(t, err == .Shape_Mismatch)
	_, err = div(a, b)
	testing.expect(t, err == .Shape_Mismatch)
	_, err = add_views(as_view(a), as_view(b))
	testing.expect(t, err == .Shape_Mismatch)
	_, err = sub_views(as_view(a), as_view(b))
	testing.expect(t, err == .Shape_Mismatch)
	_, err = mul_views(as_view(a), as_view(b))
	testing.expect(t, err == .Shape_Mismatch)
	_, err = div_views(as_view(a), as_view(b))
	testing.expect(t, err == .Shape_Mismatch)
	testing.expect(t, add_into(&output, a, b) == .Shape_Mismatch)
	testing.expect(t, sub_into(&output, a, b) == .Shape_Mismatch)
	testing.expect(t, mul_into(&output, a, b) == .Shape_Mismatch)
	testing.expect(t, div_into(&output, a, b) == .Shape_Mismatch)
	testing.expect(t, add_in_place(&a, b) == .Shape_Mismatch)
	testing.expect(t, sub_in_place(&a, b) == .Shape_Mismatch)
	testing.expect(t, mul_in_place(&a, b) == .Shape_Mismatch)
	testing.expect(t, div_in_place(&a, b) == .Shape_Mismatch)
}

@(test)
test_broadcast_shape_errors :: proc(t: ^testing.T) {
	a, err := new_tensor(i32, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&a)
	b: Tensor(i32)
	b, err = new_tensor(i32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&b)
	_, err = add(a, b)
	testing.expect(t, err == .Shape_Mismatch)

	shape_buffer := [2]int{}
	result_rank, shape_err := broadcast_shape([]int{2, 1}, []int{1, 3}, shape_buffer[:])
	testing.expect(t, shape_err == .None && result_rank == 2 && shape_buffer[0] == 2 && shape_buffer[1] == 3)

	small_shape_buffer := [1]int{}
	strides_buffer := [1]int{}
	_, err = broadcast_view(as_view(a), []int{3}, small_shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .Shape_Mismatch)
	_, err = broadcast_view(as_view(a), []int{2, 3}, small_shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .Insufficient_Buffer)
}

@(test)
test_shape_and_bounds_errors :: proc(t: ^testing.T) {
	_, err := new_tensor(f32, []int{2, -1})
	testing.expect(t, err == .Invalid_Shape)

	value: Tensor(i32)
	value, err = from_slice(i32, []i32{1}, []int{1})
	testing.expect(t, err == .None)
	defer release(&value)
	testing.expect(t, set_linear(&value, 1, 2) == .Out_Of_Bounds)
}
@(test)
test_unary_math_and_activations :: proc(t: ^testing.T) {
	values, err := from_slice(f32, []f32{-2, 0, 1, 2}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&values)

	exponentials, exp_err := exp(values)
	testing.expect(t, exp_err == .None)
	defer release(&exponentials)
	testing.expect(t, exponentials.data[0] > 0.1353 && exponentials.data[0] < 0.1354)
	testing.expect(t, exponentials.data[1] == 1)
	testing.expect(t, exponentials.data[2] > 2.718 && exponentials.data[2] < 2.719)

	logs, log_err := log(exponentials)
	testing.expect(t, log_err == .None)
	defer release(&logs)
	for i in 0..<len(values.data) {
		testing.expect(t, logs.data[i] - values.data[i] < 0.0001 && values.data[i] - logs.data[i] < 0.0001)
	}

	squares, square_err := from_slice(f32, []f32{0, 1, 4, 9}, []int{2, 2})
	testing.expect(t, square_err == .None)
	defer release(&squares)
	square_roots, sqrt_err := sqrt(squares)
	testing.expect(t, sqrt_err == .None)
	defer release(&square_roots)
	testing.expect(t, square_roots.data[0] == 0 && square_roots.data[1] == 1 && square_roots.data[2] == 2 && square_roots.data[3] == 3)

	rectified, relu_err := relu(values)
	testing.expect(t, relu_err == .None)
	defer release(&rectified)
	testing.expect(t, rectified.data[0] == 0 && rectified.data[1] == 0 && rectified.data[2] == 1 && rectified.data[3] == 2)

	probabilities, sigmoid_err := sigmoid(values)
	testing.expect(t, sigmoid_err == .None)
	defer release(&probabilities)
	testing.expect(t, probabilities.data[0] > 0.1191 && probabilities.data[0] < 0.1193)
	testing.expect(t, probabilities.data[1] == 0.5)
	testing.expect(t, probabilities.data[2] > 0.7310 && probabilities.data[2] < 0.7311)
	testing.expect(t, probabilities.data[3] > 0.8807 && probabilities.data[3] < 0.8809)

	hyperbolic, tanh_err := tanh(values)
	testing.expect(t, tanh_err == .None)
	defer release(&hyperbolic)
	testing.expect(t, hyperbolic.data[0] < -0.964 && hyperbolic.data[0] > -0.965)
	testing.expect(t, hyperbolic.data[1] == 0)
	testing.expect(t, hyperbolic.data[2] > 0.761 && hyperbolic.data[2] < 0.762)

	output: Tensor(f32)
	output, err = new_tensor(f32, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect(t, sigmoid_into(&output, values) == .None)
	testing.expect(t, output.data[1] == 0.5)

	wrong_shape: Tensor(f32)
	wrong_shape, err = new_tensor(f32, []int{4})
	testing.expect(t, err == .None)
	defer release(&wrong_shape)
	testing.expect(t, relu_into(&wrong_shape, values) == .Shape_Mismatch)

	in_place, in_place_err := from_slice(f32, []f32{-1000, 1000}, []int{2})
	testing.expect(t, in_place_err == .None)
	defer release(&in_place)
	testing.expect(t, sigmoid_in_place(&in_place) == .None)
	testing.expect(t, in_place.data[0] >= 0 && in_place.data[0] < 0.0001)
	testing.expect(t, in_place.data[1] <= 1 && in_place.data[1] > 0.9999)

	transposed_shape := [2]int{}
	transposed_strides := [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	view_result, view_result_err := relu_views(transposed)
	testing.expect(t, view_result_err == .None)
	defer release(&view_result)
	testing.expect(t, view_result.data[0] == 0 && view_result.data[1] == 1 && view_result.data[2] == 0 && view_result.data[3] == 2)
}

expect_f64_data :: proc(t: ^testing.T, actual, expected: []f64, tolerance: f64 = 1e-12, loc := #caller_location) {
	testing.expect_value(t, len(actual), len(expected), loc)
	if len(actual) != len(expected) {
		return
	}
	for i in 0..<len(expected) {
		testing.expectf(t, abs(actual[i]-expected[i]) <= tolerance, "index %d: got %v, want %v", i, actual[i], expected[i], loc = loc)
	}
}

expect_f32_close :: proc(t: ^testing.T, actual, expected: []f32, tolerance: f32 = 1e-6) {
	testing.expect_value(t, len(actual), len(expected))
	if len(actual) != len(expected) {
		return
	}
	for i in 0..<len(expected) {
		testing.expectf(t, abs(actual[i]-expected[i]) <= tolerance, "index %d: got %v, want %v", i, actual[i], expected[i])
	}
}

expect_int_data :: proc(t: ^testing.T, actual, expected: []int) {
	testing.expect_value(t, len(actual), len(expected))
	if len(actual) != len(expected) {
		return
	}
	for i in 0..<len(expected) {
		testing.expect_value(t, actual[i], expected[i])
	}
}

expect_shape :: proc(t: ^testing.T, actual, expected: []int) {
	testing.expectf(t, _shapes_match(actual, expected), "shape %v, want %v", actual, expected)
}

// Failing_Allocator forwards to backing until remaining allocations run out,
// then reports Out_Of_Memory. Frees always reach backing so leaks stay visible.
Failing_Allocator :: struct {
	backing:   mem.Allocator,
	remaining: int,
}

failing_allocator_proc :: proc(allocator_data: rawptr, mode: mem.Allocator_Mode, size, alignment: int, old_memory: rawptr, old_size: int, location := #caller_location) -> ([]byte, mem.Allocator_Error) {
	state := (^Failing_Allocator)(allocator_data)
	#partial switch mode {
	case .Alloc, .Alloc_Non_Zeroed, .Resize, .Resize_Non_Zeroed:
		if state.remaining <= 0 {
			return nil, .Out_Of_Memory
		}
		state.remaining -= 1
	}
	return state.backing.procedure(state.backing.data, mode, size, alignment, old_memory, old_size, location)
}

failing_allocator :: proc(state: ^Failing_Allocator) -> mem.Allocator {
	return mem.Allocator{procedure = failing_allocator_proc, data = state}
}

// expect_clean_allocation_failures runs scenario with an allocation budget of
// 0, 1, 2, ... until it succeeds. Every shorter budget must fail with
// Allocation_Failed; the test runner's tracking allocator reports any leak.
expect_clean_allocation_failures :: proc(t: ^testing.T, scenario: proc(allocator: mem.Allocator) -> Tensor_Error) {
	for budget in 0..<256 {
		state := Failing_Allocator{backing = context.allocator, remaining = budget}
		err := scenario(failing_allocator(&state))
		if err == .None {
			testing.expect(t, budget > 0)
			return
		}
		testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	}
	testing.fail_now(t, "scenario never succeeded")
}

tensor_allocation_scenario :: proc(allocator: mem.Allocator) -> Tensor_Error {
	a := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3}, allocator) or_return
	defer release(&a)
	row := from_slice(f64, []f64{1, 2, 3}, []int{3}, allocator) or_return
	defer release(&row)
	sum := add(a, row) or_return
	defer release(&sum)
	reshape(&sum, []int{6}) or_return
	axis_sum := sum_axis(a, 1, true) or_return
	defer release(&axis_sum)
	transposed := transpose_copy(as_view(a), allocator) or_return
	defer release(&transposed)
	product := matmul(a, transposed) or_return
	defer release(&product)
	vector := dot(a, row) or_return
	defer release(&vector)
	exponentials := exp(a) or_return
	defer release(&exponentials)
	mean := mean_all(a) or_return
	defer release(&mean)
	indices := argmax_axis(a, 1, true, allocator) or_return
	defer release(&indices)
	smallest := argmin_axis_views(as_view(a), 0, false, allocator) or_return
	defer release(&smallest)
	largest_index := argmax_all(a, allocator) or_return
	defer release(&largest_index)
	smallest_index := argmin_all_views(as_view(a), allocator) or_return
	defer release(&smallest_index)
	column_products := prod_axis_views(as_view(a), 0, false, allocator) or_return
	defer release(&column_products)
	product_total := prod_all(a, allocator) or_return
	defer release(&product_total)
	column_variances := var_axis_views(as_view(a), 0, 1, false, allocator) or_return
	defer release(&column_variances)
	row_deviations := std_axis(a, 1, 0, true, allocator) or_return
	defer release(&row_deviations)
	variance_total := var_all_views(as_view(a), 0, allocator) or_return
	defer release(&variance_total)
	deviation_total := std_all(a, 1, allocator) or_return
	defer release(&deviation_total)
	plane_sums := sum_axes(a, []int{0, 1}, true, allocator) or_return
	defer release(&plane_sums)
	column_means := mean_axes_views(as_view(a), []int{0}, false, allocator) or_return
	defer release(&column_means)
	row_max := max_axes(a, []int{-1}, false, allocator) or_return
	defer release(&row_max)
	smallest_total := min_axes_views(as_view(a), []int{1, 0}, false, allocator) or_return
	defer release(&smallest_total)
	return .None
}

@(test)
test_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	expect_clean_allocation_failures(t, tensor_allocation_scenario)
}

@(test)
test_into_outputs_reject_overlapping_inputs :: proc(t: ^testing.T) {
	a, err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&a)
	b: Tensor(f64)
	b, err = from_slice(f64, []f64{5, 6, 7, 8}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&b)

	// Matrix products read each input element several times.
	testing.expect_value(t, matmul_into(&a, a, b), Tensor_Error.Aliased_Output)
	testing.expect_value(t, matmul_into(&b, a, b), Tensor_Error.Aliased_Output)
	expect_f64_data(t, a.data, []f64{1, 2, 3, 4})
	expect_f64_data(t, b.data, []f64{5, 6, 7, 8})
	product: Tensor(f64)
	product, err = new_tensor(f64, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&product)
	testing.expect_value(t, matmul_into(&product, a, b), Tensor_Error.None)
	expect_f64_data(t, product.data, []f64{19, 22, 43, 50})

	ones: Tensor(f64)
	ones, err = from_slice(f64, []f64{1, 1}, []int{2})
	testing.expect(t, err == .None)
	defer release(&ones)
	testing.expect_value(t, dot_into(&ones, a, ones), Tensor_Error.Aliased_Output)
	testing.expect_value(t, dot_views_into(&ones, as_view(a), as_view(ones)), Tensor_Error.Aliased_Output)
	expect_f64_data(t, ones.data, []f64{1, 1})
	row_sums: Tensor(f64)
	row_sums, err = new_tensor(f64, []int{2})
	testing.expect(t, err == .None)
	defer release(&row_sums)
	testing.expect_value(t, dot_views_into(&row_sums, as_view(a), as_view(ones)), Tensor_Error.None)
	expect_f64_data(t, row_sums.data, []f64{3, 7})

	// A unary op may run over the output's own storage, but not a permutation of it.
	squares: Tensor(f64)
	squares, err = from_slice(f64, []f64{1, 4, 9, 16}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&squares)
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(squares), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	testing.expect_value(t, sqrt_views_into(&squares, transposed), Tensor_Error.Aliased_Output)
	reversed_shape, reversed_strides := [2]int{}, [2]int{}
	reversed, reverse_err := slice_view(as_view(squares), 0, 1, 2, -1, reversed_shape[:], reversed_strides[:])
	testing.expect(t, reverse_err == .None)
	testing.expect_value(t, sqrt_views_into(&squares, reversed), Tensor_Error.Aliased_Output)
	expect_f64_data(t, squares.data, []f64{1, 4, 9, 16})
	testing.expect_value(t, sqrt_views_into(&squares, as_view(squares)), Tensor_Error.None)
	expect_f64_data(t, squares.data, []f64{1, 2, 3, 4})
	testing.expect_value(t, exp_into(&squares, squares), Tensor_Error.None)

	// An axis reduction writes early outputs before reading later inputs.
	folded := Tensor_View(f64){data = row_sums.data, shape = []int{2, 1}, strides = []int{1, 1}}
	testing.expect_value(t, sum_axis_views_into(&row_sums, folded, 1), Tensor_Error.Aliased_Output)
	expect_f64_data(t, row_sums.data, []f64{3, 7})

	// Element-wise ops read each element before writing it, so self-aliasing is fine.
	testing.expect_value(t, add_into(&a, a, a), Tensor_Error.None)
	expect_f64_data(t, a.data, []f64{2, 4, 6, 8})
	testing.expect_value(t, mul_in_place(&a, a), Tensor_Error.None)
	expect_f64_data(t, a.data, []f64{4, 16, 36, 64})
}

@(test)
test_signed_integer_division_overflow_returns_error :: proc(t: ^testing.T) {
	dividend, err := from_slice(i32, []i32{6, min(i32)}, []int{2})
	testing.expect(t, err == .None)
	defer release(&dividend)
	divisors: Tensor(i32)
	divisors, err = from_slice(i32, []i32{2, -1}, []int{2})
	testing.expect(t, err == .None)
	defer release(&divisors)

	_, err = div(dividend, divisors)
	testing.expect_value(t, err, Tensor_Error.Integer_Overflow)
	_, err = div_views(as_view(dividend), as_view(divisors))
	testing.expect_value(t, err, Tensor_Error.Integer_Overflow)
	output: Tensor(i32)
	output, err = from_slice(i32, []i32{99, 99}, []int{2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect_value(t, div_into(&output, dividend, divisors), Tensor_Error.Integer_Overflow)
	expect_i32_data(t, output.data, []i32{99, 99})
	testing.expect_value(t, div_in_place(&dividend, divisors), Tensor_Error.Integer_Overflow)
	expect_i32_data(t, dividend.data, []i32{6, min(i32)})

	// A broadcast -1 divisor only fails when it meets min(T).
	negative_one: Tensor(i32)
	negative_one, err = from_slice(i32, []i32{-1}, []int{})
	testing.expect(t, err == .None)
	defer release(&negative_one)
	_, err = div(dividend, negative_one)
	testing.expect_value(t, err, Tensor_Error.Integer_Overflow)
	safe: Tensor(i32)
	safe, err = from_slice(i32, []i32{6, -8}, []int{2})
	testing.expect(t, err == .None)
	defer release(&safe)
	testing.expect_value(t, div_in_place(&safe, negative_one), Tensor_Error.None)
	expect_i32_data(t, safe.data, []i32{-6, 8})

	small: Tensor(i8)
	small, err = from_slice(i8, []i8{min(i8)}, []int{1})
	testing.expect(t, err == .None)
	defer release(&small)
	small_divisor: Tensor(i8)
	small_divisor, err = from_slice(i8, []i8{-1}, []int{1})
	testing.expect(t, err == .None)
	defer release(&small_divisor)
	testing.expect_value(t, div_in_place(&small, small_divisor), Tensor_Error.Integer_Overflow)

	unsigned: Tensor(u8)
	unsigned, err = from_slice(u8, []u8{255, 10}, []int{2})
	testing.expect(t, err == .None)
	defer release(&unsigned)
	unsigned_divisor: Tensor(u8)
	unsigned_divisor, err = from_slice(u8, []u8{255, 5}, []int{2})
	testing.expect(t, err == .None)
	defer release(&unsigned_divisor)
	testing.expect_value(t, div_in_place(&unsigned, unsigned_divisor), Tensor_Error.None)
	testing.expect(t, unsigned.data[0] == 1 && unsigned.data[1] == 2)
}

@(test)
test_float_division_by_zero_follows_ieee :: proc(t: ^testing.T) {
	dividend, err := from_slice(f64, []f64{1, -2, 0}, []int{3})
	testing.expect(t, err == .None)
	defer release(&dividend)
	zero: Tensor(f64)
	zero, err = from_slice(f64, []f64{0}, []int{})
	testing.expect(t, err == .None)
	defer release(&zero)

	quotient: Tensor(f64)
	quotient, err = div(dividend, zero)
	testing.expect(t, err == .None)
	defer release(&quotient)
	testing.expect(t, math.is_inf(quotient.data[0], 1) && math.is_inf(quotient.data[1], -1) && math.is_nan(quotient.data[2]))

	zero.data[0] = -0.0
	testing.expect_value(t, div_in_place(&dividend, zero), Tensor_Error.None)
	testing.expect(t, math.is_inf(dividend.data[0], -1) && math.is_inf(dividend.data[1], 1))
}

@(test)
test_max_and_min_propagate_nan :: proc(t: ^testing.T) {
	nan := math.nan_f64()
	for position in 0..<3 {
		values := []f64{1, 2, 3}
		values[position] = nan
		source, err := from_slice(f64, values, []int{3})
		testing.expect(t, err == .None)
		largest: Tensor(f64)
		largest, err = max_all(source)
		testing.expect(t, err == .None)
		testing.expectf(t, math.is_nan(largest.data[0]), "max_all with NaN at %d gave %v", position, largest.data[0])
		smallest: Tensor(f64)
		smallest, err = min_axis(source, 0)
		testing.expect(t, err == .None)
		testing.expectf(t, math.is_nan(smallest.data[0]), "min_axis with NaN at %d gave %v", position, smallest.data[0])
		release(&largest)
		release(&smallest)
		release(&source)
	}
}

unary_reference :: proc(value: f64, operation: _Unary_Operation) -> f64 {
	switch operation {
	case .Exp:
		return math.exp(value)
	case .Log:
		return math.ln(value)
	case .Sqrt:
		return math.sqrt(value)
	case .ReLU:
		return max(value, 0)
	case .Sigmoid:
		return 1 / (1 + math.exp(-value))
	case .Tanh:
		return math.tanh(value)
	}
	return 0
}

// The dispatch helpers below call each public unary procedure by name so every
// variant is exercised, not just the shared implementation.
unary_allocating :: proc(source: Tensor(f64), operation: _Unary_Operation) -> (Tensor(f64), Tensor_Error) {
	switch operation {
	case .Exp:
		return exp(source)
	case .Log:
		return log(source)
	case .Sqrt:
		return sqrt(source)
	case .ReLU:
		return relu(source)
	case .Sigmoid:
		return sigmoid(source)
	case .Tanh:
		return tanh(source)
	}
	return {}, .Unsupported_Operation
}

unary_views :: proc(source: Tensor_View(f64), operation: _Unary_Operation) -> (Tensor(f64), Tensor_Error) {
	switch operation {
	case .Exp:
		return exp_views(source)
	case .Log:
		return log_views(source)
	case .Sqrt:
		return sqrt_views(source)
	case .ReLU:
		return relu_views(source)
	case .Sigmoid:
		return sigmoid_views(source)
	case .Tanh:
		return tanh_views(source)
	}
	return {}, .Unsupported_Operation
}

unary_views_into :: proc(output: ^Tensor(f64), source: Tensor_View(f64), operation: _Unary_Operation) -> Tensor_Error {
	switch operation {
	case .Exp:
		return exp_views_into(output, source)
	case .Log:
		return log_views_into(output, source)
	case .Sqrt:
		return sqrt_views_into(output, source)
	case .ReLU:
		return relu_views_into(output, source)
	case .Sigmoid:
		return sigmoid_views_into(output, source)
	case .Tanh:
		return tanh_views_into(output, source)
	}
	return .Unsupported_Operation
}

unary_into :: proc(output: ^Tensor(f64), source: Tensor(f64), operation: _Unary_Operation) -> Tensor_Error {
	switch operation {
	case .Exp:
		return exp_into(output, source)
	case .Log:
		return log_into(output, source)
	case .Sqrt:
		return sqrt_into(output, source)
	case .ReLU:
		return relu_into(output, source)
	case .Sigmoid:
		return sigmoid_into(output, source)
	case .Tanh:
		return tanh_into(output, source)
	}
	return .Unsupported_Operation
}

unary_in_place :: proc(value: ^Tensor(f64), operation: _Unary_Operation) -> Tensor_Error {
	switch operation {
	case .Exp:
		return exp_in_place(value)
	case .Log:
		return log_in_place(value)
	case .Sqrt:
		return sqrt_in_place(value)
	case .ReLU:
		return relu_in_place(value)
	case .Sigmoid:
		return sigmoid_in_place(value)
	case .Tanh:
		return tanh_in_place(value)
	}
	return .Unsupported_Operation
}

check_unary_variants :: proc(t: ^testing.T, operation: _Unary_Operation, values: []f64) {
	source, err := from_slice(f64, values, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&source)
	expected := make([]f64, 6)
	defer delete(expected)
	transposed_expected := make([]f64, 6)
	defer delete(transposed_expected)
	for row in 0..<2 {
		for column in 0..<3 {
			value := unary_reference(values[row*3+column], operation)
			expected[row*3+column] = value
			transposed_expected[column*2+row] = value
		}
	}
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(source), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)

	allocated: Tensor(f64)
	allocated, err = unary_allocating(source, operation)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&allocated)
	expect_shape(t, allocated.shape, []int{2, 3})
	expect_f64_data(t, allocated.data, expected)

	from_view: Tensor(f64)
	from_view, err = unary_views(transposed, operation)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&from_view)
	expect_shape(t, from_view.shape, []int{3, 2})
	expect_f64_data(t, from_view.data, transposed_expected)

	output: Tensor(f64)
	output, err = new_tensor(f64, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&output)
	testing.expect_value(t, unary_views_into(&output, transposed, operation), Tensor_Error.None)
	expect_f64_data(t, output.data, transposed_expected)
	testing.expect_value(t, unary_views_into(&output, as_view(source), operation), Tensor_Error.Shape_Mismatch)

	same_shape: Tensor(f64)
	same_shape, err = new_tensor(f64, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&same_shape)
	testing.expect_value(t, unary_into(&same_shape, source, operation), Tensor_Error.None)
	expect_f64_data(t, same_shape.data, expected)
	testing.expect_value(t, unary_into(&output, source, operation), Tensor_Error.Shape_Mismatch)

	testing.expect_value(t, unary_in_place(&source, operation), Tensor_Error.None)
	expect_f64_data(t, source.data, expected)

	// Rank-zero and empty tensors keep their shapes.
	scalar: Tensor(f64)
	scalar, err = from_slice(f64, values[:1], []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)
	scalar_result: Tensor(f64)
	scalar_result, err = unary_allocating(scalar, operation)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&scalar_result)
	testing.expect(t, len(scalar_result.shape) == 0)
	expect_f64_data(t, scalar_result.data, expected[:1])
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{0, 3})
	testing.expect(t, err == .None)
	defer release(&empty)
	empty_result: Tensor(f64)
	empty_result, err = unary_allocating(empty, operation)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_result)
	expect_shape(t, empty_result.shape, []int{0, 3})
}

@(test)
test_every_unary_variant_matches_reference_math :: proc(t: ^testing.T) {
	positive := []f64{0.25, 1, 2, 4, 9, 0.5}
	signed := []f64{-2, -0.5, 0, 0.5, 1, 3}
	check_unary_variants(t, .Exp, signed)
	check_unary_variants(t, .Log, positive)
	check_unary_variants(t, .Sqrt, positive)
	check_unary_variants(t, .ReLU, signed)
	check_unary_variants(t, .Sigmoid, signed)
	check_unary_variants(t, .Tanh, signed)
}

@(test)
test_linear_and_coordinate_indexing_bounds :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)

	value: i32
	value, err = get_linear(values, 4)
	testing.expect(t, err == .None && value == 5)
	_, err = get_linear(values, -1)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = get_linear(values, 6)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	testing.expect_value(t, set_linear(&values, -1, 0), Tensor_Error.Out_Of_Bounds)
	testing.expect_value(t, set_linear(&values, 5, 60), Tensor_Error.None)

	value, err = get_at(values, []int{1, 2})
	testing.expect(t, err == .None && value == 60)
	// Negative indices count from the end, as in NumPy.
	value, err = get_at(values, []int{-1, -3})
	testing.expect(t, err == .None && value == 4)
	_, err = get_at(values, []int{-3, 0})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = get_at(values, []int{0, 3})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = get_at(values, []int{0})
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, set_at(&values, []int{2, 0}, 0), Tensor_Error.Out_Of_Bounds)
	testing.expect_value(t, set_at(&values, []int{0, 0, 0}, 0), Tensor_Error.Shape_Mismatch)

	scalar: Tensor(i32)
	scalar, err = from_slice(i32, []i32{7}, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)
	testing.expect_value(t, set_at(&scalar, []int{}, 8), Tensor_Error.None)
	value, err = get_at(scalar, []int{})
	testing.expect(t, err == .None && value == 8)

	// View indexing checks the view's own bounds and its backing storage.
	backing := []i32{1, 2, 3}
	short_view := Tensor_View(i32){data = backing, shape = []int{4}, strides = []int{1}}
	_, err = get_at_view(short_view, []int{3})
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	testing.expect_value(t, set_at_view(short_view, []int{3}, 0), Tensor_Error.Out_Of_Bounds)
	testing.expect_value(t, set_at_view(short_view, []int{2}, 30), Tensor_Error.None)
	testing.expect(t, backing[2] == 30)
}

@(test)
test_reshape_changes_rank_and_rejects_invalid_shapes :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)

	testing.expect_value(t, reshape(&values, []int{1, 6, 1}), Tensor_Error.None)
	expect_shape(t, values.shape, []int{1, 6, 1})
	expect_shape(t, values.strides, []int{6, 1, 1})
	value: i32
	value, err = get_at(values, []int{0, 4, 0})
	testing.expect(t, err == .None && value == 5)

	testing.expect_value(t, reshape(&values, []int{6}), Tensor_Error.None)
	expect_shape(t, values.shape, []int{6})
	testing.expect_value(t, reshape(&values, []int{4}), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, reshape(&values, []int{3, -2}), Tensor_Error.Invalid_Shape)
	testing.expect_value(t, reshape(&values, []int{-1, -1}), Tensor_Error.Invalid_Shape)
	testing.expect_value(t, reshape(&values, []int{4, -1}), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, reshape(&values, []int{0, -1}), Tensor_Error.Shape_Mismatch)
	expect_shape(t, values.shape, []int{6})

	// One dimension may be -1 and is inferred, as in NumPy.
	testing.expect_value(t, reshape(&values, []int{-1, 2}), Tensor_Error.None)
	expect_shape(t, values.shape, []int{3, 2})
	testing.expect_value(t, reshape(&values, []int{-1}), Tensor_Error.None)
	expect_shape(t, values.shape, []int{6})
	expect_i32_data(t, values.data, []i32{1, 2, 3, 4, 5, 6})

	scalar: Tensor(i32)
	scalar, err = from_slice(i32, []i32{9}, []int{1, 1})
	testing.expect(t, err == .None)
	defer release(&scalar)
	testing.expect_value(t, reshape(&scalar, []int{}), Tensor_Error.None)
	testing.expect(t, len(scalar.shape) == 0 && len(scalar.strides) == 0)

	empty: Tensor(i32)
	empty, err = new_tensor(i32, []int{0, 4})
	testing.expect(t, err == .None)
	defer release(&empty)
	testing.expect_value(t, reshape(&empty, []int{2, 0, 3}), Tensor_Error.None)
	expect_shape(t, empty.shape, []int{2, 0, 3})
}

@(test)
test_axis_reduction_view_and_into_variants :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 5, 3, 4, 2, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)
	view := as_view(values)
	rows: Tensor(f64)
	rows, err = new_tensor(f64, []int{2})
	testing.expect(t, err == .None)
	defer release(&rows)
	columns: Tensor(f64)
	columns, err = new_tensor(f64, []int{3})
	testing.expect(t, err == .None)
	defer release(&columns)
	kept_columns: Tensor(f64)
	kept_columns, err = new_tensor(f64, []int{1, 3})
	testing.expect(t, err == .None)
	defer release(&kept_columns)
	kept_rows: Tensor(f64)
	kept_rows, err = new_tensor(f64, []int{2, 1})
	testing.expect(t, err == .None)
	defer release(&kept_rows)

	testing.expect_value(t, sum_axis_views_into(&rows, view, 1), Tensor_Error.None)
	expect_f64_data(t, rows.data, []f64{9, 12})
	testing.expect_value(t, mean_axis_views_into(&columns, view, 0), Tensor_Error.None)
	expect_f64_data(t, columns.data, []f64{2.5, 3.5, 4.5})
	testing.expect_value(t, max_axis_views_into(&rows, view, 1), Tensor_Error.None)
	expect_f64_data(t, rows.data, []f64{5, 6})
	testing.expect_value(t, min_axis_views_into(&columns, view, 0), Tensor_Error.None)
	expect_f64_data(t, columns.data, []f64{1, 2, 3})
	testing.expect_value(t, mean_axis_into(&rows, values, 1), Tensor_Error.None)
	expect_f64_data(t, rows.data, []f64{3, 4})
	testing.expect_value(t, max_axis_into(&kept_columns, values, 0, true), Tensor_Error.None)
	expect_f64_data(t, kept_columns.data, []f64{4, 5, 6})
	testing.expect_value(t, min_axis_into(&kept_rows, values, 1, true), Tensor_Error.None)
	expect_f64_data(t, kept_rows.data, []f64{1, 2})

	// Wrong output shapes and axes are rejected without writing.
	testing.expect_value(t, sum_axis_into(&columns, values, 1), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, max_axis_into(&kept_rows, values, 1), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, min_axis_into(&rows, values, 1, true), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, sum_axis_views_into(&rows, view, 2), Tensor_Error.Out_Of_Bounds)
	expect_f64_data(t, kept_rows.data, []f64{1, 2})
	_, err = sum_axis(values, -3)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = max_axis(values, 2)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)

	// Negative axes count from the end, as in NumPy.
	last_axis: Tensor(f64)
	last_axis, err = sum_axis(values, -1)
	testing.expect(t, err == .None)
	defer release(&last_axis)
	expect_f64_data(t, last_axis.data, []f64{9, 12})
	testing.expect_value(t, max_axis_into(&columns, values, -2), Tensor_Error.None)
	expect_f64_data(t, columns.data, []f64{4, 5, 6})

	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(view, transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	column_max: Tensor(f64)
	column_max, err = max_axis_views(transposed, 0)
	testing.expect(t, err == .None)
	defer release(&column_max)
	expect_f64_data(t, column_max.data, []f64{5, 6})
	row_min: Tensor(f64)
	row_min, err = min_axis_views(transposed, 1, true)
	testing.expect(t, err == .None)
	defer release(&row_min)
	expect_shape(t, row_min.shape, []int{3, 1})
	expect_f64_data(t, row_min.data, []f64{1, 2, 3})
	all_max: Tensor(f64)
	all_max, err = max_all_views(transposed)
	testing.expect(t, err == .None)
	defer release(&all_max)
	expect_f64_data(t, all_max.data, []f64{6})
	all_min: Tensor(f64)
	all_min, err = min_all_views(transposed)
	testing.expect(t, err == .None)
	defer release(&all_min)
	expect_f64_data(t, all_min.data, []f64{1})

	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)
	testing.expect_value(t, max_axis_into(&rows, empty, 1), Tensor_Error.Empty_Reduction)
	testing.expect_value(t, min_axis_views_into(&rows, as_view(empty), 1), Tensor_Error.Empty_Reduction)
	_, err = min_all_views(as_view(empty))
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	testing.expect_value(t, sum_axis_into(&rows, empty, 1), Tensor_Error.None)
	expect_f64_data(t, rows.data, []f64{0, 0})
}

@(test)
test_argmax_and_argmin_along_an_axis :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 5, 5, 7, 2, 7}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)

	// Ties pick the first index.
	rows: Tensor(int)
	rows, err = argmax_axis(values, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&rows)
	expect_shape(t, rows.shape, []int{2})
	expect_int_data(t, rows.data, []int{1, 0})
	columns: Tensor(int)
	columns, err = argmax_axis(values, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&columns)
	expect_int_data(t, columns.data, []int{1, 0, 1})
	smallest: Tensor(int)
	smallest, err = argmin_axis(values, -1, true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&smallest)
	expect_shape(t, smallest.shape, []int{2, 1})
	expect_int_data(t, smallest.data, []int{0, 1})

	// Views report indices in their own logical order.
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	transposed_rows: Tensor(int)
	transposed_rows, err = argmax_axis_views(transposed, 1)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_rows)
	expect_int_data(t, transposed_rows.data, []int{1, 0, 1})
	transposed_columns: Tensor(int)
	transposed_columns, err = argmin_axis_views(transposed, 0, true)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_columns)
	expect_shape(t, transposed_columns.shape, []int{1, 2})
	expect_int_data(t, transposed_columns.data, []int{0, 1})

	// The first NaN wins for both argmax and argmin.
	with_nan: Tensor(f64)
	with_nan, err = from_slice(f64, []f64{1, 9, math.nan_f64(), 0, math.nan_f64()}, []int{5})
	testing.expect(t, err == .None)
	defer release(&with_nan)
	nan_max: Tensor(int)
	nan_max, err = argmax_axis(with_nan, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nan_max)
	expect_shape(t, nan_max.shape, []int{})
	expect_int_data(t, nan_max.data, []int{2})
	nan_min: Tensor(int)
	nan_min, err = argmin_axis(with_nan, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nan_min)
	expect_int_data(t, nan_min.data, []int{2})

	integers: Tensor(i32)
	integers, err = from_slice(i32, []i32{4, -2, 4, -2}, []int{4})
	testing.expect(t, err == .None)
	defer release(&integers)
	integer_min: Tensor(int)
	integer_min, err = argmin_axis(integers, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_min)
	expect_int_data(t, integer_min.data, []int{1})

	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)
	_, err = argmax_axis(empty, 1)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = argmin_axis_views(as_view(empty), -1)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	// Reducing the non-empty axis of an empty tensor gives an empty result.
	empty_columns: Tensor(int)
	empty_columns, err = argmax_axis(empty, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&empty_columns)
	expect_shape(t, empty_columns.shape, []int{0})
	_, err = argmax_axis(values, 2)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
}

@(test)
test_argmax_and_argmin_over_all_elements :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 5, 5, 7, 2, 7}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)

	// Ties pick the first flat index.
	largest: Tensor(int)
	largest, err = argmax_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&largest)
	expect_shape(t, largest.shape, []int{})
	expect_int_data(t, largest.data, []int{3})
	smallest: Tensor(int)
	smallest, err = argmin_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&smallest)
	expect_int_data(t, smallest.data, []int{0})

	// A transposed view counts in its own row-major order: [[1,7],[5,2],[5,7]]
	// has its first 7 at flat index 1, where storage order would give 3, and
	// [[9,2],[0,1]] has its 0 at index 2, where storage order would give 1.
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	transposed_largest: Tensor(int)
	transposed_largest, err = argmax_all_views(transposed)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_largest)
	expect_int_data(t, transposed_largest.data, []int{1})
	transposed_values: Tensor(f64)
	transposed_values, err = from_slice(f64, []f64{9, 0, 2, 1}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&transposed_values)
	small_shape, small_strides := [2]int{}, [2]int{}
	small_transposed, small_err := transpose_view(as_view(transposed_values), small_shape[:], small_strides[:])
	testing.expect(t, small_err == .None)
	transposed_smallest: Tensor(int)
	transposed_smallest, err = argmin_all_views(small_transposed)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_smallest)
	expect_int_data(t, transposed_smallest.data, []int{2})

	// The first NaN wins for both argmax and argmin.
	with_nan: Tensor(f64)
	with_nan, err = from_slice(f64, []f64{1, 9, math.nan_f64(), 0, math.nan_f64(), -1}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&with_nan)
	nan_max: Tensor(int)
	nan_max, err = argmax_all(with_nan)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nan_max)
	expect_int_data(t, nan_max.data, []int{2})
	nan_min: Tensor(int)
	nan_min, err = argmin_all(with_nan)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&nan_min)
	expect_int_data(t, nan_min.data, []int{2})

	integers: Tensor(i32)
	integers, err = from_slice(i32, []i32{4, -2, 4, -2}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&integers)
	integer_max: Tensor(int)
	integer_max, err = argmax_all(integers)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&integer_max)
	expect_int_data(t, integer_max.data, []int{0})

	scalar: Tensor(f64)
	scalar, err = from_slice(f64, []f64{3}, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)
	scalar_index: Tensor(int)
	scalar_index, err = argmax_all(scalar)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&scalar_index)
	expect_int_data(t, scalar_index.data, []int{0})

	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)
	_, err = argmax_all(empty)
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
	_, err = argmin_all_views(as_view(empty))
	testing.expect_value(t, err, Tensor_Error.Empty_Reduction)
}

@(test)
test_argmax_and_argmin_into_outputs :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 5, 5, 7, 2, 7}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&values)
	rows: Tensor(int)
	rows, err = new_tensor(int, []int{2})
	testing.expect(t, err == .None)
	defer release(&rows)
	kept_columns: Tensor(int)
	kept_columns, err = new_tensor(int, []int{1, 3})
	testing.expect(t, err == .None)
	defer release(&kept_columns)

	testing.expect_value(t, argmax_axis_into(&rows, values, 1), Tensor_Error.None)
	expect_int_data(t, rows.data, []int{1, 0})
	testing.expect_value(t, argmin_axis_into(&kept_columns, values, 0, true), Tensor_Error.None)
	expect_int_data(t, kept_columns.data, []int{0, 1, 0})
	testing.expect_value(t, argmin_axis_views_into(&rows, as_view(values), -1), Tensor_Error.None)
	expect_int_data(t, rows.data, []int{0, 1})
	transposed_shape, transposed_strides := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(values), transposed_shape[:], transposed_strides[:])
	testing.expect(t, view_err == .None)
	testing.expect_value(t, argmax_axis_views_into(&rows, transposed, 0), Tensor_Error.None)
	expect_int_data(t, rows.data, []int{1, 0})

	// Errors leave the output unchanged.
	testing.expect_value(t, argmax_axis_into(&rows, values, 0), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, argmax_axis_into(&kept_columns, values, 0), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, argmin_axis_into(&rows, values, 2), Tensor_Error.Out_Of_Bounds)
	empty: Tensor(f64)
	empty, err = new_tensor(f64, []int{2, 0})
	testing.expect(t, err == .None)
	defer release(&empty)
	testing.expect_value(t, argmax_axis_into(&rows, empty, 1), Tensor_Error.Empty_Reduction)
	expect_int_data(t, rows.data, []int{1, 0})

	// An output that overlaps the source is rejected, even for an int source.
	folded := Tensor_View(int){data = rows.data, shape = []int{2, 1}, strides = []int{1, 1}}
	testing.expect_value(t, argmax_axis_views_into(&rows, folded, 1), Tensor_Error.Aliased_Output)
	expect_int_data(t, rows.data, []int{1, 0})
}

@(test)
test_integer_sum_max_and_min :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&values)

	total: Tensor(i32)
	total, err = sum_all(values)
	testing.expect(t, err == .None)
	defer release(&total)
	expect_i32_data(t, total.data, []i32{10})
	column_max: Tensor(i32)
	column_max, err = max_axis(values, 0)
	testing.expect(t, err == .None)
	defer release(&column_max)
	expect_i32_data(t, column_max.data, []i32{3, 4})
	smallest: Tensor(i32)
	smallest, err = min_all(values)
	testing.expect(t, err == .None)
	defer release(&smallest)
	expect_i32_data(t, smallest.data, []i32{1})
}

@(test)
test_float_sums_are_pairwise :: proc(t: ^testing.T) {
	// Adding 0.1 to an f32 accumulator ten million times one by one stalls once
	// the total is large enough, finishing about 10% off; pairwise sums stay
	// within rounding of the exact total.
	COUNT :: 10_000_000
	values, err := new_tensor(f32, []int{COUNT})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&values)
	fill(&values, 0.1)
	expect_near :: proc(t: ^testing.T, actual: f32, expected: f64, location := #caller_location) {
		testing.expectf(t, abs(f64(actual)-expected) <= 1e-3*expected, "got %v, want %v", actual, expected, loc = location)
	}

	total: Tensor(f32)
	total, err = sum_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&total)
	expect_near(t, total.data[0], 1e6)
	mean: Tensor(f32)
	mean, err = mean_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&mean)
	expect_near(t, mean.data[0], 0.1)
	along_axis: Tensor(f32)
	along_axis, err = sum_axis(values, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&along_axis)
	expect_near(t, along_axis.data[0], 1e6)

	// Many short rows: the rows are combined pairwise too, not just each row.
	short_rows := Tensor_View(f32){data = values.data, shape = []int{COUNT/4, 2}, strides = []int{4, 1}}
	short_rows_total: Tensor(f32)
	short_rows_total, err = sum_all_views(short_rows)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&short_rows_total)
	expect_near(t, short_rows_total.data[0], 5e5)

	// A few long strided rows.
	columns := Tensor_View(f32){data = values.data, shape = []int{COUNT/4, 4}, strides = []int{4, 1}}
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	rows, view_err := transpose_view(columns, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, view_err, Tensor_Error.None)
	transposed_total: Tensor(f32)
	transposed_total, err = sum_all_views(rows)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_total)
	expect_near(t, transposed_total.data[0], 1e6)
	over_axes: Tensor(f32)
	over_axes, err = mean_axes_views(columns, []int{0, 1})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&over_axes)
	expect_near(t, over_axes.data[0], 0.1)
	strided_axis: Tensor(f32)
	strided_axis, err = sum_axis_views(columns, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&strided_axis)
	for column in strided_axis.data {
		expect_near(t, column, 2.5e5)
	}

	// Integer sums are exact and still wrap in T.
	bytes: Tensor(i8)
	bytes, err = new_tensor(i8, []int{100, 100})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&bytes)
	fill(&bytes, 1)
	byte_total: Tensor(i8)
	byte_total, err = sum_all(bytes)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&byte_total)
	testing.expect_value(t, byte_total.data[0], i8(16))
	counts: Tensor(i32)
	counts, err = new_tensor(i32, []int{1000, 1000})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&counts)
	fill(&counts, 3)
	count_total: Tensor(i32)
	count_total, err = sum_all(counts)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&count_total)
	testing.expect_value(t, count_total.data[0], i32(3_000_000))
}

@(test)
test_float_variances_are_pairwise :: proc(t: ^testing.T) {
	// Every squared deviation of 0.9, 1.1, 0.9, ... is about 0.01. Adding ten
	// million of them to an f32 accumulator one by one finishes about 4% low;
	// summed pairwise they stay within rounding of the exact variance.
	COUNT :: 10_000_000
	values, err := new_tensor(f32, []int{COUNT})
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&values)
	for &value, i in values.data {
		value = 0.9 if i % 2 == 0 else 1.1
	}
	expect_near :: proc(t: ^testing.T, actual: f32, expected: f64, location := #caller_location) {
		testing.expectf(t, abs(f64(actual)-expected) <= 1e-3*expected, "got %v, want %v", actual, expected, loc = location)
	}

	variance: Tensor(f32)
	variance, err = var_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&variance)
	expect_near(t, variance.data[0], 0.01)
	deviation: Tensor(f32)
	deviation, err = std_all(values)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&deviation)
	expect_near(t, deviation.data[0], 0.1)
	along_axis: Tensor(f32)
	along_axis, err = var_axis(values, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&along_axis)
	expect_near(t, along_axis.data[0], 0.01)

	// Many short rows: the rows are combined pairwise too, not just each row.
	// Each row is one 0.9 and one 1.1.
	short_rows := Tensor_View(f32){data = values.data, shape = []int{COUNT/4, 2}, strides = []int{4, 1}}
	short_rows_variance: Tensor(f32)
	short_rows_variance, err = var_all_views(short_rows)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&short_rows_variance)
	expect_near(t, short_rows_variance.data[0], 0.01)

	// A few long strided rows.
	columns := Tensor_View(f32){data = values.data, shape = []int{COUNT/4, 4}, strides = []int{4, 1}}
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	rows, view_err := transpose_view(columns, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, view_err, Tensor_Error.None)
	transposed_variance: Tensor(f32)
	transposed_variance, err = var_all_views(rows)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&transposed_variance)
	expect_near(t, transposed_variance.data[0], 0.01)

	// A strided axis: with an odd stride each column still alternates.
	fifths := Tensor_View(f32){data = values.data, shape = []int{COUNT/5, 5}, strides = []int{5, 1}}
	strided_axis: Tensor(f32)
	strided_axis, err = var_axis_views(fifths, 0)
	testing.expect_value(t, err, Tensor_Error.None)
	defer release(&strided_axis)
	for column in strided_axis.data {
		expect_near(t, column, 0.01)
	}
}

@(test)
test_float_sums_across_block_boundaries :: proc(t: ^testing.T) {
	// Whole numbers sum exactly in f64, so any element dropped or added twice
	// around the eight-lane blocks, their leftovers or a split shows up.
	for count in ([]int{0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 257, 1000}) {
		expected := f64(count*(count+1)/2)
		values, err := new_tensor(f64, []int{count})
		testing.expect_value(t, err, Tensor_Error.None)
		defer release(&values)
		for &value, i in values.data {
			value = f64(i+1)
		}
		total: Tensor(f64)
		total, err = sum_all(values)
		testing.expect_value(t, err, Tensor_Error.None)
		defer release(&total)
		testing.expectf(t, total.data[0] == expected, "sum_all of %v: got %v, want %v", count, total.data[0], expected)
		along_axis: Tensor(f64)
		along_axis, err = sum_axis(values, 0)
		testing.expect_value(t, err, Tensor_Error.None)
		defer release(&along_axis)
		testing.expectf(t, along_axis.data[0] == expected, "sum_axis of %v: got %v, want %v", count, along_axis.data[0], expected)
		if count > 0 {
			mean: Tensor(f64)
			mean, err = mean_all(values)
			testing.expect_value(t, err, Tensor_Error.None)
			defer release(&mean)
			testing.expectf(t, mean.data[0] == f64(count+1)/2, "mean_all of %v: got %v", count, mean.data[0])
		}

		// Axis 0 of [count, 3] walks each column with stride 3.
		columns: Tensor(f64)
		columns, err = new_tensor(f64, []int{count, 3})
		testing.expect_value(t, err, Tensor_Error.None)
		defer release(&columns)
		for &value, i in columns.data {
			value = f64(i/3+1)
		}
		column_totals: Tensor(f64)
		column_totals, err = sum_axis(columns, 0)
		testing.expect_value(t, err, Tensor_Error.None)
		defer release(&column_totals)
		for column in column_totals.data {
			testing.expectf(t, column == expected, "strided sum of %v: got %v, want %v", count, column, expected)
		}
	}
}

@(test)
test_zero_stride_broadcast_views_feed_every_operation :: proc(t: ^testing.T) {
	row, err := from_slice(f64, []f64{1, 2, 3}, []int{3})
	testing.expect(t, err == .None)
	defer release(&row)
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	repeated, view_err := broadcast_view(as_view(row), []int{2, 3}, shape_buffer[:], strides_buffer[:])
	testing.expect(t, view_err == .None)
	testing.expect(t, repeated.strides[0] == 0 && repeated.strides[1] == 1)

	column_sums: Tensor(f64)
	column_sums, err = sum_axis_views(repeated, 0)
	testing.expect(t, err == .None)
	defer release(&column_sums)
	expect_f64_data(t, column_sums.data, []f64{2, 4, 6})
	total: Tensor(f64)
	total, err = sum_all_views(repeated)
	testing.expect(t, err == .None)
	defer release(&total)
	expect_f64_data(t, total.data, []f64{12})

	doubled: Tensor(f64)
	doubled, err = add_views(repeated, repeated)
	testing.expect(t, err == .None)
	defer release(&doubled)
	expect_f64_data(t, doubled.data, []f64{2, 4, 6, 2, 4, 6})
	roots: Tensor(f64)
	roots, err = sqrt_views(repeated)
	testing.expect(t, err == .None)
	defer release(&roots)
	expect_f64_data(t, roots.data, []f64{1, math.SQRT_TWO, math.sqrt(f64(3)), 1, math.SQRT_TWO, math.sqrt(f64(3))})

	weights: Tensor(f64)
	weights, err = from_slice(f64, []f64{1, 0, 0, 1, 1, 1}, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&weights)
	product: Tensor(f64)
	product, err = matmul_views(repeated, as_view(weights))
	testing.expect(t, err == .None)
	defer release(&product)
	expect_f64_data(t, product.data, []f64{4, 5, 4, 5})
	vector: Tensor(f64)
	vector, err = dot_views(repeated, as_view(row))
	testing.expect(t, err == .None)
	defer release(&vector)
	expect_f64_data(t, vector.data, []f64{14, 14})
}

slice_values_i32 :: proc(view: Tensor_View(i32)) -> [dynamic]i32 {
	values := make([dynamic]i32, 0, view.shape[0])
	for i in 0..<view.shape[0] {
		value, _ := get_at_view(view, []int{i})
		append(&values, value)
	}
	return values
}

@(test)
test_slice_view_steps_and_errors :: proc(t: ^testing.T) {
	source, err := from_slice(i32, []i32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{10})
	testing.expect(t, err == .None)
	defer release(&source)
	shape_buffer, strides_buffer := [1]int{}, [1]int{}

	view: Tensor_View(i32)
	view, err = slice_view(as_view(source), 0, 1, 4, 2, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	stepped := slice_values_i32(view)
	defer delete(stepped)
	expect_i32_data(t, stepped[:], []i32{1, 3, 5, 7})

	view, err = slice_view(as_view(source), 0, 9, 4, -3, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	reversed := slice_values_i32(view)
	defer delete(reversed)
	expect_i32_data(t, reversed[:], []i32{9, 6, 3, 0})

	view, err = slice_view(as_view(source), 0, 10, 0, 1, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None && view.shape[0] == 0)

	_, err = slice_view(as_view(source), 0, 11, 0, 1, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = slice_view(as_view(source), 0, 8, 2, 2, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = slice_view(as_view(source), 0, -11, 1, 1, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	// A negative start counts from the end: source[-3:] has 3 elements.
	view, err = slice_view(as_view(source), 0, -3, 3, 1, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	tail_values := slice_values_i32(view)
	defer delete(tail_values)
	expect_i32_data(t, tail_values[:], []i32{7, 8, 9})
	_, err = slice_view(as_view(source), 0, 0, 2, 0, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = slice_view(as_view(source), 0, 0, -1, 1, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = slice_view(as_view(source), 1, 0, 1, 1, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = slice_view(as_view(source), -2, 0, 1, 1, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = slice_view(as_view(source), 0, 0, 1, 1, shape_buffer[:0], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Insufficient_Buffer)

	// Slicing a matrix's columns produces a strided view other operations accept.
	grid: Tensor(i32)
	grid, err = from_slice(i32, []i32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&grid)
	matrix_shape, matrix_strides := [2]int{}, [2]int{}
	columns: Tensor_View(i32)
	columns, err = slice_view(as_view(grid), 1, 0, 2, 2, matrix_shape[:], matrix_strides[:])
	testing.expect(t, err == .None)
	expect_shape(t, columns.shape, []int{2, 2})
	sums: Tensor(i32)
	sums, err = sum_axis_views(columns, 1)
	testing.expect(t, err == .None)
	defer release(&sums)
	expect_i32_data(t, sums.data, []i32{4, 10})

	_, err = transpose_view(as_view(grid), matrix_shape[:1], matrix_strides[:])
	testing.expect_value(t, err, Tensor_Error.Insufficient_Buffer)
}

@(test)
test_rank_zero_tensors_in_arithmetic_and_reductions :: proc(t: ^testing.T) {
	scalar, err := from_slice(f64, []f64{4}, []int{})
	testing.expect(t, err == .None)
	defer release(&scalar)
	grid: Tensor(f64)
	grid, err = from_slice(f64, []f64{2, 4, 6, 8}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&grid)

	scaled: Tensor(f64)
	scaled, err = mul(scalar, grid)
	testing.expect(t, err == .None)
	defer release(&scaled)
	expect_shape(t, scaled.shape, []int{2, 2})
	expect_f64_data(t, scaled.data, []f64{8, 16, 24, 32})
	quotient: Tensor(f64)
	quotient, err = div(grid, scalar)
	testing.expect(t, err == .None)
	defer release(&quotient)
	expect_f64_data(t, quotient.data, []f64{0.5, 1, 1.5, 2})
	testing.expect_value(t, sub_in_place(&grid, scalar), Tensor_Error.None)
	expect_f64_data(t, grid.data, []f64{-2, 0, 2, 4})

	largest: Tensor(f64)
	largest, err = max_all(scalar)
	testing.expect(t, err == .None)
	defer release(&largest)
	expect_f64_data(t, largest.data, []f64{4})
	_, err = sum_axis(scalar, 0)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = dot(scalar, scalar)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = matmul(scalar, grid)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
}

@(test)
test_permute_view_and_copy_view :: proc(t: ^testing.T) {
	values, err := from_slice(i32, []i32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, []int{2, 3, 2})
	testing.expect(t, err == .None)
	defer release(&values)
	shape_buffer, strides_buffer := [3]int{}, [3]int{}

	// Like np.transpose(values, (1, -1, 0)).
	permuted: Tensor_View(i32)
	permuted, err = permute_view(as_view(values), []int{1, -1, 0}, shape_buffer[:], strides_buffer[:])
	testing.expect(t, err == .None)
	expect_shape(t, permuted.shape, []int{3, 2, 2})
	copied: Tensor(i32)
	copied, err = copy_view(permuted)
	testing.expect(t, err == .None)
	defer release(&copied)
	expect_shape(t, copied.strides, []int{4, 2, 1})
	expect_i32_data(t, copied.data, []i32{0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11})

	_, err = permute_view(as_view(values), []int{0, 0, 1}, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = permute_view(as_view(values), []int{0, 1}, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = permute_view(as_view(values), []int{0, 1, 3}, shape_buffer[:], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
	_, err = permute_view(as_view(values), []int{0, 1, 2}, shape_buffer[:2], strides_buffer[:])
	testing.expect_value(t, err, Tensor_Error.Insufficient_Buffer)

	backing := []i32{1, 2}
	invalid := Tensor_View(i32){data = backing, shape = []int{3}, strides = []int{1}}
	_, err = copy_view(invalid)
	testing.expect_value(t, err, Tensor_Error.Out_Of_Bounds)
}

@(test)
test_binary_views_into_variants :: proc(t: ^testing.T) {
	grid, err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&grid)
	row: Tensor(f64)
	row, err = from_slice(f64, []f64{1, 2, 4}, []int{3})
	testing.expect(t, err == .None)
	defer release(&row)
	output: Tensor(f64)
	output, err = new_tensor(f64, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&output)

	testing.expect_value(t, add_views_into(&output, as_view(grid), as_view(row)), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{2, 4, 7, 5, 7, 10})
	testing.expect_value(t, sub_views_into(&output, as_view(grid), as_view(row)), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{0, 0, -1, 3, 3, 2})
	testing.expect_value(t, mul_views_into(&output, as_view(grid), as_view(row)), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{1, 4, 12, 4, 10, 24})
	testing.expect_value(t, div_views_into(&output, as_view(grid), as_view(row)), Tensor_Error.None)
	expect_f64_data(t, output.data, []f64{1, 1, 0.75, 4, 2.5, 1.5})

	// Transposed operands use the strided path.
	shape_buffer, strides_buffer := [2]int{}, [2]int{}
	transposed, view_err := transpose_view(as_view(grid), shape_buffer[:], strides_buffer[:])
	testing.expect(t, view_err == .None)
	column: Tensor(f64)
	column, err = from_slice(f64, []f64{10, 20}, []int{2})
	testing.expect(t, err == .None)
	defer release(&column)
	transposed_output: Tensor(f64)
	transposed_output, err = new_tensor(f64, []int{3, 2})
	testing.expect(t, err == .None)
	defer release(&transposed_output)
	testing.expect_value(t, add_views_into(&transposed_output, transposed, as_view(column)), Tensor_Error.None)
	expect_f64_data(t, transposed_output.data, []f64{11, 24, 12, 25, 13, 26})

	testing.expect_value(t, add_views_into(&output, as_view(grid), as_view(column)), Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, add_views_into(&transposed_output, as_view(grid), as_view(row)), Tensor_Error.Shape_Mismatch)

	// The output's own storage is a safe operand; a shifted window of it is not.
	testing.expect_value(t, add_views_into(&output, as_view(output), as_view(row)), Tensor_Error.None)
	first_row := Tensor_View(f64){data = output.data, shape = []int{3}, strides = []int{1}}
	testing.expect_value(t, add_views_into(&output, as_view(grid), first_row), Tensor_Error.Aliased_Output)
	snapshot := make([]f64, len(output.data))
	defer delete(snapshot)
	copy(snapshot, output.data)
	first_row_tensor := Tensor(f64){data = output.data[:3], shape = []int{3}, strides = []int{1}}
	testing.expect_value(t, add_in_place(&output, first_row_tensor), Tensor_Error.Aliased_Output)
	expect_f64_data(t, output.data, snapshot)
}

@(test)
test_allocating_operations_use_the_given_allocator :: proc(t: ^testing.T) {
	values, err := from_slice(f64, []f64{1, 2, 3, 4}, []int{2, 2})
	testing.expect(t, err == .None)
	defer release(&values)

	// An allocator with no budget makes every allocating form fail, which
	// shows the result does not come from values' allocator.
	state := Failing_Allocator{backing = context.allocator, remaining = 0}
	empty := failing_allocator(&state)
	_, err = add(values, values, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = exp(values, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = sum_axis(values, 0, false, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = mean_all(values, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = matmul(values, values, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = dot(values, values, empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)
	_, err = copy_view(as_view(values), empty)
	testing.expect_value(t, err, Tensor_Error.Allocation_Failed)

	// The default is context.allocator, not the operand's allocator.
	operand_state := Failing_Allocator{backing = context.allocator, remaining = 3}
	operand: Tensor(f64)
	operand, err = from_slice(f64, []f64{1, 2}, []int{2}, failing_allocator(&operand_state))
	testing.expect(t, err == .None)
	defer release(&operand)
	doubled: Tensor(f64)
	doubled, err = add(operand, operand)
	testing.expect(t, err == .None)
	defer release(&doubled)
	expect_f64_data(t, doubled.data, []f64{2, 4})
}

@(test)
test_matmul_promotes_vectors_like_numpy :: proc(t: ^testing.T) {
	grid, err := from_slice(f64, []f64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	testing.expect(t, err == .None)
	defer release(&grid)
	row: Tensor(f64)
	row, err = from_slice(f64, []f64{1, 0, -1}, []int{3})
	testing.expect(t, err == .None)
	defer release(&row)
	column: Tensor(f64)
	column, err = from_slice(f64, []f64{1, 1}, []int{2})
	testing.expect(t, err == .None)
	defer release(&column)

	matrix_vector: Tensor(f64)
	matrix_vector, err = matmul(grid, row)
	testing.expect(t, err == .None)
	defer release(&matrix_vector)
	expect_shape(t, matrix_vector.shape, []int{2})
	expect_f64_data(t, matrix_vector.data, []f64{-2, -2})
	vector_matrix: Tensor(f64)
	vector_matrix, err = matmul(column, grid)
	testing.expect(t, err == .None)
	defer release(&vector_matrix)
	expect_shape(t, vector_matrix.shape, []int{3})
	expect_f64_data(t, vector_matrix.data, []f64{5, 7, 9})
	_, err = matmul(row, grid)
	testing.expect_value(t, err, Tensor_Error.Shape_Mismatch)
}

@(test)
test_shapes_above_max_rank_are_rejected :: proc(t: ^testing.T) {
	shape: [MAX_RANK+1]int
	for &dim in shape {
		dim = 1
	}
	_, err := new_tensor(f64, shape[:])
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	values, values_err := new_tensor(f64, shape[:MAX_RANK])
	testing.expect(t, values_err == .None)
	defer release(&values)

	backing := []f64{1}
	strides: [MAX_RANK+1]int
	too_deep := Tensor_View(f64){data = backing, shape = shape[:], strides = strides[:]}
	_, err = sum_axis_views(too_deep, 0)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = add_views(too_deep, as_view(values))
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
	_, err = copy_view(too_deep)
	testing.expect_value(t, err, Tensor_Error.Invalid_Shape)
}
