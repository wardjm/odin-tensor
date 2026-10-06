package tensor

// Axis and whole-tensor reductions.

import "base:intrinsics"
import "core:math"

// _Reduction_Operation selects the aggregation used by _reduce_axis_views*,
// _reduce_axes_views*, and _reduce_all_views*.
_Reduction_Operation :: enum {
	Sum,
	Mean,
	Prod,
	Max,
	Min,
}

// _combine folds element into an accumulator. Max and min propagate NaN, as in
// NumPy: once the accumulator is NaN it stays NaN.
_combine :: #force_inline proc(accumulator, element: $T, $operation: _Reduction_Operation) -> T {
	when operation == .Sum || operation == .Mean {
		return accumulator + element
	} else when operation == .Prod {
		return accumulator * element
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
// Sums start from zero, products from one; max/min start from the first element.
// Float sums are pairwise (see _pairwise_sum); integer sums add in order.
_reduce_run :: #force_inline proc(data: []$T, offset, step, count: int, $operation: _Reduction_Operation) -> T {
	offset := offset
	when (operation == .Sum || operation == .Mean) && intrinsics.type_is_float(T) {
		value := _pairwise_sum(data, offset, step, count, 0, false)
	} else when operation == .Sum || operation == .Mean {
		value: T = 0
		for _ in 0..<count {
			value += data[offset]
			offset += step
		}
	} else when operation == .Prod {
		value: T = 1
		for _ in 0..<count {
			value *= data[offset]
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

// PAIRWISE_BLOCK is the run length below which _pairwise_sum stops splitting,
// as in NumPy.
PAIRWISE_BLOCK :: 128

// _pairwise_sum adds count elements starting at offset and advancing by step.
// Runs longer than PAIRWISE_BLOCK are split in half and the halves summed
// recursively, so rounding error grows with log(count) rather than count. Within
// a block, eight interleaved accumulators keep each partial sum short, as
// NumPy's pairwise_sum does. With squared, it adds (element - center)² instead.
_pairwise_sum :: proc(data: []$T, offset, step, count: int, center: T, $squared: bool) -> T {
	if count < 8 {
		value: T = 0
		for i in 0..<count {
			value += _pairwise_term(data[offset + i*step], center, squared)
		}
		return value
	}
	if count <= PAIRWISE_BLOCK {
		partial: [8]T
		for lane in 0..<8 {
			partial[lane] = _pairwise_term(data[offset + lane*step], center, squared)
		}
		i := 8
		for ; i <= count-8; i += 8 {
			base := offset + i*step
			for lane in 0..<8 {
				partial[lane] += _pairwise_term(data[base + lane*step], center, squared)
			}
		}
		value := ((partial[0]+partial[1]) + (partial[2]+partial[3])) + ((partial[4]+partial[5]) + (partial[6]+partial[7]))
		for ; i < count; i += 1 {
			value += _pairwise_term(data[offset + i*step], center, squared)
		}
		return value
	}
	half := count/2
	half -= half % 8
	return _pairwise_sum(data, offset, step, half, center, squared) + _pairwise_sum(data, offset + half*step, step, count-half, center, squared)
}

// _pairwise_term is the element _pairwise_sum adds: element itself, or with
// squared, its squared deviation from center.
_pairwise_term :: #force_inline proc(element, center: $T, $squared: bool) -> T {
	when squared {
		deviation := element - center
		return deviation * deviation
	} else {
		return element
	}
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
	when operation != .Sum && operation != .Prod {
		if reduced_count == 0 {
			return .Empty_Reduction
		}
	}

	inner := _reduction_inner_count(source.shape, reduced_axis)
	reduced_stride := source.strides[reduced_axis]
	for output_index in 0..<len(output.data) {
		base_offset := _reduction_base_offset(source, reduced_axis, inner, output_index)
		output.data[output_index] = _reduce_run(source.data, base_offset, reduced_stride, reduced_count, operation)
	}
	return .None
}

// _reduction_inner_count is the number of elements after axis, treating the
// source as [outer, reduced, inner] around that axis.
_reduction_inner_count :: proc(shape: []int, axis: int) -> int {
	inner := 1
	for dim in shape[axis+1:] {
		inner *= dim
	}
	return inner
}

// _reduction_base_offset maps an output element of an axis reduction to the
// source position with the reduced axis at zero.
_reduction_base_offset :: proc(source: Tensor_View($T), reduced_axis, inner, output_index: int) -> int {
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
	return base_offset
}

// _reduced_shape writes the shape of an axis reduction into buffer.
_reduced_shape :: proc(shape: []int, axis: int, keep_dims: bool, buffer: []int) -> []int {
	rank := 0
	for dim, source_axis in shape {
		if source_axis == axis && !keep_dims {
			continue
		}
		buffer[rank] = 1 if source_axis == axis else dim
		rank += 1
	}
	return buffer[:rank]
}

// _reduce_axis_views returns a contiguous tensor reduced along axis.
_reduce_axis_views :: proc(source: Tensor_View($T), axis: int, $operation: _Reduction_Operation, keep_dims := false, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	normalized := _normalize_axis(axis, len(source.shape)) or_return
	when operation != .Sum && operation != .Prod {
		if source.shape[normalized] == 0 {
			return result, .Empty_Reduction
		}
	}
	shape_buffer: [MAX_RANK]int
	result = new_tensor(T, _reduced_shape(source.shape, normalized, keep_dims, shape_buffer[:]), allocator) or_return
	if err = _reduce_axis_views_into(&result, source, normalized, operation, keep_dims); err != .None {
		release(&result)
	}
	return
}

// sum_axis, mean_axis, prod_axis, max_axis, and min_axis reduce one axis. A
// negative axis counts from the end, and keep_dims keeps the reduced axis with
// size 1. The product of an empty axis is 1; mean/max/min of an empty axis
// return Empty_Reduction. mean requires a floating-point type. Integer sums and
// products wrap on overflow.
sum_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Sum, keep_dims, allocator)
}
mean_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axis_views(as_view(source), axis, .Mean, keep_dims, allocator)
}
prod_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Prod, keep_dims, allocator)
}
max_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Max, keep_dims, allocator)
}
min_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views(as_view(source), axis, .Min, keep_dims, allocator)
}

sum_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(source, axis, .Sum, keep_dims, allocator)
}
mean_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axis_views(source, axis, .Mean, keep_dims, allocator)
}
prod_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views(source, axis, .Prod, keep_dims, allocator)
}
max_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views(source, axis, .Max, keep_dims, allocator)
}
min_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views(source, axis, .Min, keep_dims, allocator)
}

sum_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Sum, keep_dims)
}
mean_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Mean, keep_dims)
}
prod_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Prod, keep_dims)
}
max_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Max, keep_dims)
}
min_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views_into(output, as_view(source), axis, .Min, keep_dims)
}

sum_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Sum, keep_dims)
}
mean_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axis_views_into(output, source, axis, .Mean, keep_dims)
}
prod_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Prod, keep_dims)
}
max_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Max, keep_dims)
}
min_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axis_views_into(output, source, axis, .Min, keep_dims)
}

// _reduced_axes_mask marks each axis in axes, normalizing negative ones. An
// out-of-range axis returns Out_Of_Bounds and a repeated one Invalid_Shape.
// Every axis is bounds-checked before any is checked for repeats, as NumPy's
// normalize_axis_tuple does, so the error does not depend on the axes' order.
_reduced_axes_mask :: proc(axes: []int, rank: int) -> (mask: [MAX_RANK]bool, err: Tensor_Error) {
	for axis in axes {
		_normalize_axis(axis, rank) or_return
	}
	for axis in axes {
		normalized, _ := _normalize_axis(axis, rank)
		if mask[normalized] {
			return mask, .Invalid_Shape
		}
		mask[normalized] = true
	}
	return mask, .None
}

// _reduced_axes_shape writes the shape of a reduction over the masked axes into
// buffer.
_reduced_axes_shape :: proc(shape: []int, mask: ^[MAX_RANK]bool, keep_dims: bool, buffer: []int) -> []int {
	rank := 0
	for dim, axis in shape {
		if mask[axis] && !keep_dims {
			continue
		}
		buffer[rank] = 1 if mask[axis] else dim
		rank += 1
	}
	return buffer[:rank]
}

// _reduced_axes_count is the number of source elements folded into each output
// element of a reduction over the masked axes.
_reduced_axes_count :: proc(shape: []int, mask: ^[MAX_RANK]bool) -> int {
	count := 1
	for dim, axis in shape {
		if mask[axis] {
			count *= dim
		}
	}
	return count
}

// _reduce_axes_views_into reduces source over every axis in axes into an
// exactly shaped, contiguous output. Negative axes count from the end; a
// repeated axis returns Invalid_Shape and an out-of-range one Out_Of_Bounds.
// keep_dims retains each reduced
// dimension as a singleton. Output must not overlap source; an overlap returns
// Aliased_Output.
_reduce_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, $operation: _Reduction_Operation, keep_dims := false) -> Tensor_Error {
	_validate_view(source) or_return
	mask := _reduced_axes_mask(axes, len(source.shape)) or_return
	shape_buffer: [MAX_RANK]int
	if _validate_output(output^) != .None || !_shapes_match(output.shape, _reduced_axes_shape(source.shape, &mask, keep_dims, shape_buffer[:])) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}

	// Split the source into a view over the kept axes, which addresses the
	// start of each output's block, and a view over the reduced axes, which
	// walks that block.
	kept_shape, kept_strides, reduced_shape, reduced_strides: [MAX_RANK]int
	kept_rank, reduced_rank := 0, 0
	for dim, axis in source.shape {
		if mask[axis] {
			reduced_shape[reduced_rank], reduced_strides[reduced_rank] = dim, source.strides[axis]
			reduced_rank += 1
		} else {
			kept_shape[kept_rank], kept_strides[kept_rank] = dim, source.strides[axis]
			kept_rank += 1
		}
	}
	kept := Tensor_View(T){data = source.data, shape = kept_shape[:kept_rank], strides = kept_strides[:kept_rank], offset = source.offset}
	reduced := Tensor_View(T){data = source.data, shape = reduced_shape[:reduced_rank], strides = reduced_strides[:reduced_rank]}
	reduced_count, _ := _element_count(reduced.shape)
	when operation != .Sum && operation != .Prod {
		if reduced_count == 0 {
			return .Empty_Reduction
		}
	}
	for output_index in 0..<len(output.data) {
		reduced.offset = _broadcast_offset(kept, kept.shape, output_index)
		output.data[output_index] = _fold_view(reduced, reduced_count, operation)
	}
	return .None
}

// _reduce_axes_views returns a contiguous tensor reduced over every axis in axes.
_reduce_axes_views :: proc(source: Tensor_View($T), axes: []int, $operation: _Reduction_Operation, keep_dims := false, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	mask := _reduced_axes_mask(axes, len(source.shape)) or_return
	when operation != .Sum && operation != .Prod {
		if _reduced_axes_count(source.shape, &mask) == 0 {
			return result, .Empty_Reduction
		}
	}
	shape_buffer: [MAX_RANK]int
	result = new_tensor(T, _reduced_axes_shape(source.shape, &mask, keep_dims, shape_buffer[:]), allocator) or_return
	if err = _reduce_axes_views_into(&result, source, axes, operation, keep_dims); err != .None {
		release(&result)
	}
	return
}

// sum_axes, mean_axes, prod_axes, max_axes, and min_axes reduce over several
// axes at once, like NumPy's axis=(0, 2). Negative axes count from the end; a
// repeated axis returns Invalid_Shape and an out-of-range one Out_Of_Bounds.
// An empty axes list reduces nothing and copies the source. Empty reductions
// follow the single-axis forms.
sum_axes :: proc(source: Tensor($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views(as_view(source), axes, .Sum, keep_dims, allocator)
}
mean_axes :: proc(source: Tensor($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axes_views(as_view(source), axes, .Mean, keep_dims, allocator)
}
prod_axes :: proc(source: Tensor($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views(as_view(source), axes, .Prod, keep_dims, allocator)
}
max_axes :: proc(source: Tensor($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views(as_view(source), axes, .Max, keep_dims, allocator)
}
min_axes :: proc(source: Tensor($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views(as_view(source), axes, .Min, keep_dims, allocator)
}

sum_axes_views :: proc(source: Tensor_View($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views(source, axes, .Sum, keep_dims, allocator)
}
mean_axes_views :: proc(source: Tensor_View($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_axes_views(source, axes, .Mean, keep_dims, allocator)
}
prod_axes_views :: proc(source: Tensor_View($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views(source, axes, .Prod, keep_dims, allocator)
}
max_axes_views :: proc(source: Tensor_View($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views(source, axes, .Max, keep_dims, allocator)
}
min_axes_views :: proc(source: Tensor_View($T), axes: []int, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views(source, axes, .Min, keep_dims, allocator)
}

sum_axes_into :: proc(output: ^Tensor($T), source: Tensor(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views_into(output, as_view(source), axes, .Sum, keep_dims)
}
mean_axes_into :: proc(output: ^Tensor($T), source: Tensor(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axes_views_into(output, as_view(source), axes, .Mean, keep_dims)
}
prod_axes_into :: proc(output: ^Tensor($T), source: Tensor(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views_into(output, as_view(source), axes, .Prod, keep_dims)
}
max_axes_into :: proc(output: ^Tensor($T), source: Tensor(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views_into(output, as_view(source), axes, .Max, keep_dims)
}
min_axes_into :: proc(output: ^Tensor($T), source: Tensor(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views_into(output, as_view(source), axes, .Min, keep_dims)
}

sum_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views_into(output, source, axes, .Sum, keep_dims)
}
mean_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _reduce_axes_views_into(output, source, axes, .Mean, keep_dims)
}
prod_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _reduce_axes_views_into(output, source, axes, .Prod, keep_dims)
}
max_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views_into(output, source, axes, .Max, keep_dims)
}
min_axes_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axes: []int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_axes_views_into(output, source, axes, .Min, keep_dims)
}

// _arg_reduce_axis_views_into writes the index of the largest (.Max) or
// smallest (.Min) element along axis into an exactly shaped, contiguous output.
// Ties pick the first index and the first NaN wins, as in NumPy. Output must
// not overlap source; an overlap returns Aliased_Output.
_arg_reduce_axis_views_into :: proc(output: ^Tensor(int), source: Tensor_View($T), axis: int, $operation: _Reduction_Operation, keep_dims := false) -> Tensor_Error {
	#assert(operation == .Max || operation == .Min)
	_validate_view(source) or_return
	reduced_axis := _normalize_axis(axis, len(source.shape)) or_return
	if !_reduction_shape_matches(output^, source.shape, reduced_axis, keep_dims) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}
	reduced_count := source.shape[reduced_axis]
	if reduced_count == 0 {
		return .Empty_Reduction
	}
	inner := _reduction_inner_count(source.shape, reduced_axis)
	reduced_stride := source.strides[reduced_axis]
	for output_index in 0..<len(output.data) {
		offset := _reduction_base_offset(source, reduced_axis, inner, output_index)
		output.data[output_index] = _arg_run(source.data, offset, reduced_stride, reduced_count, operation)
	}
	return .None
}

// _arg_run returns the position, among count > 0 elements starting at offset
// and advancing by step, of the largest (.Max) or smallest (.Min) one. Ties
// pick the first and the first NaN wins.
_arg_run :: #force_inline proc(data: []$T, offset, step, count: int, $operation: _Reduction_Operation) -> int {
	offset := offset
	best := data[offset]
	best_index := 0
	for index in 1..<count {
		when intrinsics.type_is_float(T) {
			if best != best {
				break
			}
		}
		offset += step
		element := data[offset]
		if _arg_is_better(element, best, operation) {
			best, best_index = element, index
		}
	}
	return best_index
}

// _arg_is_better reports whether element should replace best in an argmax
// (.Max) or argmin (.Min) scan. A NaN element always replaces a number.
_arg_is_better :: #force_inline proc(element, best: $T, $operation: _Reduction_Operation) -> bool {
	when operation == .Max {
		is_better := element > best
	} else {
		is_better := element < best
	}
	when intrinsics.type_is_float(T) {
		is_better = is_better || element != element
	}
	return is_better
}

// _arg_reduce_axis_views returns a new Tensor(int) of indices along axis.
_arg_reduce_axis_views :: proc(source: Tensor_View($T), axis: int, $operation: _Reduction_Operation, keep_dims := false, allocator := context.allocator) -> (result: Tensor(int), err: Tensor_Error) {
	_validate_view(source) or_return
	normalized := _normalize_axis(axis, len(source.shape)) or_return
	if source.shape[normalized] == 0 {
		return result, .Empty_Reduction
	}
	shape_buffer: [MAX_RANK]int
	result = new_tensor(int, _reduced_shape(source.shape, normalized, keep_dims, shape_buffer[:]), allocator) or_return
	if err = _arg_reduce_axis_views_into(&result, source, normalized, operation, keep_dims); err != .None {
		release(&result)
	}
	return
}

// argmax_axis and argmin_axis return the index of the largest or smallest
// element along one axis as a Tensor(int). A negative axis counts from the end,
// and keep_dims keeps the reduced axis with size 1. Ties pick the first index, a NaN wins, and
// an empty axis returns Empty_Reduction.
argmax_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views(as_view(source), axis, .Max, keep_dims, allocator)
}
argmin_axis :: proc(source: Tensor($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views(as_view(source), axis, .Min, keep_dims, allocator)
}

argmax_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views(source, axis, .Max, keep_dims, allocator)
}
argmin_axis_views :: proc(source: Tensor_View($T), axis: int, keep_dims := false, allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views(source, axis, .Min, keep_dims, allocator)
}

argmax_axis_into :: proc(output: ^Tensor(int), source: Tensor($T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views_into(output, as_view(source), axis, .Max, keep_dims)
}
argmin_axis_into :: proc(output: ^Tensor(int), source: Tensor($T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views_into(output, as_view(source), axis, .Min, keep_dims)
}

argmax_axis_views_into :: proc(output: ^Tensor(int), source: Tensor_View($T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views_into(output, source, axis, .Max, keep_dims)
}
argmin_axis_views_into :: proc(output: ^Tensor(int), source: Tensor_View($T), axis: int, keep_dims := false) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_axis_views_into(output, source, axis, .Min, keep_dims)
}

// _arg_reduce_all_views returns the flat row-major index, in the view's logical
// order, of the largest (.Max) or smallest (.Min) element as a rank-zero
// Tensor(int). Ties pick the first index, the first NaN wins, and an empty
// source returns Empty_Reduction.
_arg_reduce_all_views :: proc(source: Tensor_View($T), $operation: _Reduction_Operation, allocator := context.allocator) -> (result: Tensor(int), err: Tensor_Error) {
	#assert(operation == .Max || operation == .Min)
	_validate_view(source) or_return
	count, _ := _element_count(source.shape)
	if count == 0 {
		return result, .Empty_Reduction
	}
	result = new_tensor(int, []int{}, allocator) or_return
	result.data[0] = _arg_all_index(source, count, operation)
	return result, .None
}

// _arg_all_index returns the flat logical index of the largest (.Max) or
// smallest (.Min) of the count > 0 elements of a validated view. Ties pick the
// first index and the first NaN wins.
_arg_all_index :: proc(source: Tensor_View($T), count: int, $operation: _Reduction_Operation) -> int {
	row := _row_length(source.shape)
	step := _inner_step(source, source.shape)
	best := source.data[source.offset]
	best_index := 0
	search: for start := 0; start < count; start += row {
		offset := _broadcast_offset(source, source.shape, start)
		for column in 0..<row {
			when intrinsics.type_is_float(T) {
				if best != best {
					break search
				}
			}
			element := source.data[offset]
			if _arg_is_better(element, best, operation) {
				best, best_index = element, start+column
			}
			offset += step
		}
	}
	return best_index
}

// argmax_all and argmin_all return the flat row-major index of the largest or
// smallest element as a rank-zero Tensor(int). For a view the index follows the
// view's logical order, not its storage. Ties pick the first index, a NaN wins,
// and an empty source returns Empty_Reduction.
argmax_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_all_views(as_view(source), .Max, allocator)
}
argmin_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_all_views(as_view(source), .Min, allocator)
}

argmax_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_all_views(source, .Max, allocator)
}
argmin_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(int), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _arg_reduce_all_views(source, .Min, allocator)
}

// _fold_view aggregates the count elements of a validated view. Sums start
// from zero and products from one; max/min require count > 0. Rows of the view
// are folded with a fixed step, then combined. A contiguous view is one row.
_fold_view :: proc(source: Tensor_View($T), count: int, $operation: _Reduction_Operation) -> T {
	row, row_count, step := _fold_layout(source, count)
	when operation == .Sum || operation == .Mean {
		value := _sum_rows(source, 0, row_count, row, step, 0, false)
		when operation == .Mean {
			value /= T(count)
		}
	} else when operation == .Prod {
		value: T = 1
		for r in 0..<row_count {
			value *= _reduce_run(source.data, _broadcast_offset(source, source.shape, r*row), step, row, .Prod)
		}
	} else {
		value := _reduce_run(source.data, source.offset, step, row, operation)
		for r in 1..<row_count {
			value = _combine(value, _reduce_run(source.data, _broadcast_offset(source, source.shape, r*row), step, row, operation), operation)
		}
	}
	return value
}

// _fold_layout splits the count elements of a validated view into row_count
// rows of row elements, each advancing by step. A contiguous view is one row.
_fold_layout :: proc(source: Tensor_View($T), count: int) -> (row, row_count, step: int) {
	row = _row_length(source.shape)
	step = _inner_step(source, source.shape)
	if _is_contiguous_view(source) {
		row, step = count, 1
	}
	row_count = count/row if row > 0 else 0
	return
}

// _sum_rows adds row_count rows of a validated view, starting at first_row. For
// floats the row sums are themselves combined pairwise, so a view of many short
// rows keeps the accuracy of one long run. With squared (floats only), it adds
// squared deviations from center instead, as _pairwise_sum does.
_sum_rows :: proc(source: Tensor_View($T), first_row, row_count, row, step: int, center: T, $squared: bool) -> T {
	when intrinsics.type_is_float(T) {
		if row_count > 8 {
			half := row_count/2
			return _sum_rows(source, first_row, half, row, step, center, squared) + _sum_rows(source, first_row+half, row_count-half, row, step, center, squared)
		}
	}
	value: T = 0
	for r in first_row..<first_row+row_count {
		offset := _broadcast_offset(source, source.shape, r*row)
		when squared {
			value += _pairwise_sum(source.data, offset, step, row, center, true)
		} else {
			value += _reduce_run(source.data, offset, step, row, .Sum)
		}
	}
	return value
}

// _reduce_all_views aggregates every element into a new rank-zero tensor.
// Summing an empty tensor yields zero and its product one; mean/max/min return
// Empty_Reduction.
_reduce_all_views :: proc(source: Tensor_View($T), $operation: _Reduction_Operation, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	count, _ := _element_count(source.shape)
	when operation != .Sum && operation != .Prod {
		if count == 0 {
			return result, .Empty_Reduction
		}
	}
	result = new_tensor(T, []int{}, allocator) or_return
	result.data[0] = _fold_view(source, count, operation)
	return result, .None
}

// sum_all, mean_all, prod_all, max_all, and min_all reduce every element to a
// rank-zero tensor. Max and min propagate NaN. mean requires a floating-point
// type. Integer sums and products wrap on overflow.
sum_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(as_view(source), .Sum, allocator)
}
mean_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_all_views(as_view(source), .Mean, allocator)
}
prod_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(as_view(source), .Prod, allocator)
}
max_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_all_views(as_view(source), .Max, allocator)
}
min_all :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_all_views(as_view(source), .Min, allocator)
}

sum_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(source, .Sum, allocator)
}
mean_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _reduce_all_views(source, .Mean, allocator)
}
prod_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _reduce_all_views(source, .Prod, allocator)
}
max_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_all_views(source, .Max, allocator)
}
min_all_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return _reduce_all_views(source, .Min, allocator)
}

// _variance_run returns the variance of count elements starting at offset and
// advancing by step, dividing by count-ddof. It takes the mean first, then sums
// squared deviations, which avoids the cancellation of the one-pass formula.
// Both sums are pairwise. take_sqrt returns the standard deviation instead.
_variance_run :: #force_inline proc(data: []$T, offset, step, count, ddof: int, $take_sqrt: bool) -> T {
	mean := _reduce_run(data, offset, step, count, .Mean)
	squares := _pairwise_sum(data, offset, step, count, mean, true)
	value := squares / T(count-ddof)
	when take_sqrt {
		value = math.sqrt(value)
	}
	return value
}

// _variance_axis_views_into writes the variance (or, with take_sqrt, the
// standard deviation) along axis into an exactly shaped, contiguous output.
// An empty axis, or ddof >= its length, returns Empty_Reduction. Output must
// not overlap source; an overlap returns Aliased_Output.
_variance_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis, ddof: int, $take_sqrt: bool, keep_dims := false) -> Tensor_Error {
	_validate_view(source) or_return
	reduced_axis := _normalize_axis(axis, len(source.shape)) or_return
	if !_reduction_shape_matches(output^, source.shape, reduced_axis, keep_dims) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}
	reduced_count := source.shape[reduced_axis]
	if reduced_count == 0 || reduced_count <= ddof {
		return .Empty_Reduction
	}
	inner := _reduction_inner_count(source.shape, reduced_axis)
	reduced_stride := source.strides[reduced_axis]
	for output_index in 0..<len(output.data) {
		base_offset := _reduction_base_offset(source, reduced_axis, inner, output_index)
		output.data[output_index] = _variance_run(source.data, base_offset, reduced_stride, reduced_count, ddof, take_sqrt)
	}
	return .None
}

// _variance_axis_views returns a new tensor of variances or standard deviations
// along axis.
_variance_axis_views :: proc(source: Tensor_View($T), axis, ddof: int, $take_sqrt: bool, keep_dims := false, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	normalized := _normalize_axis(axis, len(source.shape)) or_return
	if source.shape[normalized] == 0 || source.shape[normalized] <= ddof {
		return result, .Empty_Reduction
	}
	shape_buffer: [MAX_RANK]int
	result = new_tensor(T, _reduced_shape(source.shape, normalized, keep_dims, shape_buffer[:]), allocator) or_return
	if err = _variance_axis_views_into(&result, source, normalized, ddof, take_sqrt, keep_dims); err != .None {
		release(&result)
	}
	return
}

// var_axis and std_axis return the variance and standard deviation along one
// axis, dividing by n-ddof as in NumPy. A negative axis counts from the end,
// and keep_dims keeps the reduced axis with size 1. n == 0 or ddof >= n returns
// Empty_Reduction. Both require a floating-point type.
var_axis :: proc(source: Tensor($T), axis: int, ddof := 0, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_axis_views(as_view(source), axis, ddof, false, keep_dims, allocator)
}
std_axis :: proc(source: Tensor($T), axis: int, ddof := 0, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_axis_views(as_view(source), axis, ddof, true, keep_dims, allocator)
}

var_axis_views :: proc(source: Tensor_View($T), axis: int, ddof := 0, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_axis_views(source, axis, ddof, false, keep_dims, allocator)
}
std_axis_views :: proc(source: Tensor_View($T), axis: int, ddof := 0, keep_dims := false, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_axis_views(source, axis, ddof, true, keep_dims, allocator)
}

var_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, ddof := 0, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _variance_axis_views_into(output, as_view(source), axis, ddof, false, keep_dims)
}
std_axis_into :: proc(output: ^Tensor($T), source: Tensor(T), axis: int, ddof := 0, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _variance_axis_views_into(output, as_view(source), axis, ddof, true, keep_dims)
}

var_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, ddof := 0, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _variance_axis_views_into(output, source, axis, ddof, false, keep_dims)
}
std_axis_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), axis: int, ddof := 0, keep_dims := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _variance_axis_views_into(output, source, axis, ddof, true, keep_dims)
}

// _variance_all_views returns the variance (or, with take_sqrt, the standard
// deviation) of every element as a rank-zero tensor. An empty source, or ddof
// >= the element count, returns Empty_Reduction. It makes two passes: a
// pairwise mean, then a pairwise sum of squared deviations.
_variance_all_views :: proc(source: Tensor_View($T), ddof: int, $take_sqrt: bool, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	count, _ := _element_count(source.shape)
	if count == 0 || count <= ddof {
		return result, .Empty_Reduction
	}
	result = new_tensor(T, []int{}, allocator) or_return
	mean := _fold_view(source, count, .Mean)
	row, row_count, step := _fold_layout(source, count)
	squares := _sum_rows(source, 0, row_count, row, step, mean, true)
	value := squares / T(count-ddof)
	when take_sqrt {
		value = math.sqrt(value)
	}
	result.data[0] = value
	return result, .None
}

// var_all and std_all return the variance and standard deviation of every
// element as a rank-zero tensor, dividing by n-ddof as in NumPy. n == 0 or
// ddof >= n returns Empty_Reduction. Both require a floating-point type.
var_all :: proc(source: Tensor($T), ddof := 0, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_all_views(as_view(source), ddof, false, allocator)
}
std_all :: proc(source: Tensor($T), ddof := 0, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_all_views(as_view(source), ddof, true, allocator)
}

var_all_views :: proc(source: Tensor_View($T), ddof := 0, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_all_views(source, ddof, false, allocator)
}
std_all_views :: proc(source: Tensor_View($T), ddof := 0, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _variance_all_views(source, ddof, true, allocator)
}
