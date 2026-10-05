package tensor

// Axis and whole-tensor reductions.

import "base:intrinsics"

// _Reduction_Operation selects the aggregation used by _reduce_axis_views* and
// _reduce_all_views*.
_Reduction_Operation :: enum {
	Sum,
	Mean,
	Max,
	Min,
}

// _combine folds element into an accumulator. Max and min propagate NaN, as in
// NumPy: once the accumulator is NaN it stays NaN.
_combine :: #force_inline proc(accumulator, element: $T, $operation: _Reduction_Operation) -> T {
	when operation == .Sum || operation == .Mean {
		return accumulator + element
	} else {
		when intrinsics.type_is_float(T) {
			if element != element || accumulator != accumulator {
				return element if element != element else accumulator
			}
		}
		when operation == .Max {
			return element if element > accumulator else accumulator
		} else {
			return element if element < accumulator else accumulator
		}
	}
}

// _reduce_run folds count elements starting at offset and advancing by step.
// Sums start from zero; max/min start from the first element.
_reduce_run :: #force_inline proc(data: []$T, offset, step, count: int, $operation: _Reduction_Operation) -> T {
	offset := offset
	when operation == .Sum || operation == .Mean {
		value: T = 0
		for _ in 0..<count {
			value += data[offset]
			offset += step
		}
	} else {
		value := data[offset]
		for _ in 1..<count {
			offset += step
			value = _combine(value, data[offset], operation)
		}
	}
	when operation == .Mean {
		value /= T(count)
	}
	return value
}

// _reduction_shape_matches validates an exactly shaped contiguous output.
_reduction_shape_matches :: proc(output: Tensor($T), source_shape: []int, axis: int, keep_dims: bool) -> bool {
	expected_rank := len(source_shape) if keep_dims else len(source_shape)-1
	if len(output.shape) != expected_rank || _validate_output(output) != .None {
		return false
	}
	out_axis := 0
	for source_axis in 0..<len(source_shape) {
		if source_axis == axis {
			if keep_dims {
				if output.shape[out_axis] != 1 {
					return false
				}
				out_axis += 1
			}
			continue
		}
		if output.shape[out_axis] != source_shape[source_axis] {
			return false
		}
		out_axis += 1
	}
	return true
}

// _reduce_axis_views_into reduces source along axis into an exactly shaped,
// contiguous output. A negative axis counts from the end. keep_dims retains the
// reduced dimension as a singleton. Output must not overlap source; an overlap
// returns Aliased_Output.
_reduce_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, $operation: _Reduction_Operation, keep_dims := false) -> Tensor_Error {
	_validate_view(source) or_return
	reduced_axis := _normalize_axis(axis, len(source.shape)) or_return
	if !_reduction_shape_matches(output^, source.shape, reduced_axis, keep_dims) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}
	reduced_count := source.shape[reduced_axis]
	when operation != .Sum {
		if reduced_count == 0 {
			return .Empty_Reduction
		}
	}

	// Output elements map to source positions with the reduced reduced_axis at zero.
	// Treat the source as [outer, reduced, inner] around that reduced_axis.
	inner := 1
	for dim in source.shape[reduced_axis+1:] {
		inner *= dim
	}
	reduced_stride := source.strides[reduced_axis]
	for output_index in 0..<len(output.data) {
		outer_index := output_index / inner if inner > 0 else 0
		inner_index := output_index - outer_index*inner
		base_offset := source.offset
		remaining := inner_index
		for source_axis := len(source.shape)-1; source_axis > reduced_axis; source_axis -= 1 {
			dim := source.shape[source_axis]
			base_offset += (remaining % dim) * source.strides[source_axis]
			remaining /= dim
		}
		remaining = outer_index
		for source_axis := reduced_axis-1; source_axis >= 0; source_axis -= 1 {
			dim := source.shape[source_axis]
			base_offset += (remaining % dim) * source.strides[source_axis]
			remaining /= dim
		}
		output.data[output_index] = _reduce_run(source.data, base_offset, reduced_stride, reduced_count, operation)
	}
	return .None
}

// _reduce_axis_views returns a contiguous tensor reduced along axis.
_reduce_axis_views :: proc(source: Tensor_View($T), axis: int, $operation: _Reduction_Operation, keep_dims := false, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	normalized := _normalize_axis(axis, len(source.shape)) or_return
	when operation != .Sum {
		if source.shape[normalized] == 0 {
			return result, .Empty_Reduction
		}
	}
	shape_buffer: [MAX_RANK]int
	rank := 0
	for dim, source_axis in source.shape {
		if source_axis == normalized && !keep_dims {
			continue
		}
		shape_buffer[rank] = 1 if source_axis == normalized else dim
		rank += 1
	}
	result = new_tensor(T, shape_buffer[:rank], allocator) or_return
	if err = _reduce_axis_views_into(&result, source, normalized, operation, keep_dims); err != .None {
		release(&result)
	}
	return
}

// sum_axis, mean_axis, max_axis, and min_axis reduce one axis. A negative axis
// counts from the end, and keep_dims is NumPy's keepdims. Mean/max/min of an
// empty axis return Empty_Reduction. mean requires a floating-point type.
sum_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Sum, keep_dims, allocator)
}
mean_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axis_views(as_view(source), axis, .Mean, keep_dims, allocator)
}
max_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Max, keep_dims, allocator)
}
min_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Min, keep_dims, allocator)
}

sum_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(source, axis, .Sum, keep_dims, allocator)
}
mean_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axis_views(source, axis, .Mean, keep_dims, allocator)
}
max_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(source, axis, .Max, keep_dims, allocator)
}
min_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(source, axis, .Min, keep_dims, allocator)
}

sum_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Sum, keep_dims)
}
mean_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Mean, keep_dims)
}
max_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Max, keep_dims)
}
min_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Min, keep_dims)
}

sum_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Sum, keep_dims)
}
mean_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axis_views_into(output, source, axis, .Mean, keep_dims)
}
max_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Max, keep_dims)
}
min_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Min, keep_dims)
}

// _reduce_all_views aggregates every element into a new rank-zero tensor.
// Summing an empty tensor yields zero; mean/max/min return Empty_Reduction.
// Rows of the source are folded with a fixed step, then combined.
_reduce_all_views :: proc(source: Tensor_View($T), $operation: _Reduction_Operation, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	count, _ := _element_count(source.shape)
	when operation != .Sum {
		if count == 0 {
			return result, .Empty_Reduction
		}
	}
	result = new_tensor(T, []int{}, allocator) or_return
	if count == 0 {
		return result, .None
	}
	row := _row_length(source.shape)
	step := _inner_step(source, source.shape)
	when operation == .Sum || operation == .Mean {
		value: T = 0
		for start := 0; start < count; start += row {
			value += _reduce_run(source.data, _broadcast_offset(source, source.shape, start), step, row, .Sum)
		}
		when operation == .Mean {
			value /= T(count)
		}
	} else {
		value := _reduce_run(source.data, source.offset, step, row, operation)
		for start := row; start < count; start += row {
			value = _combine(value, _reduce_run(source.data, _broadcast_offset(source, source.shape, start), step, row, operation), operation)
		}
	}
	result.data[0] = value
	return result, .None
}

// sum_all, mean_all, max_all, and min_all reduce every element to a rank-zero
// tensor. Max and min propagate NaN. mean requires a floating-point type.
sum_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(as_view(source), .Sum, allocator)
}
mean_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_all_views(as_view(source), .Mean, allocator)
}
max_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(as_view(source), .Max, allocator)
}
min_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(as_view(source), .Min, allocator)
}

sum_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(source, .Sum, allocator)
}
mean_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_all_views(source, .Mean, allocator)
}
max_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(source, .Max, allocator)
}
min_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(source, .Min, allocator)
}
