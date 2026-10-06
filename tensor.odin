package tensor

import "core:mem"

Tensor_Error :: enum {
	None,
	Invalid_Shape,
	Shape_Mismatch,
	Out_Of_Bounds,
	Allocation_Failed,
	Insufficient_Buffer,
	Division_By_Zero,
	Integer_Overflow,
	Aliased_Output,
	Empty_Reduction,
	Gradient_Not_Tracked,
	Invalid_Tape,
	Unsupported_Operation,
}

// MAX_RANK is the largest supported number of dimensions, matching NumPy.
// Kernels use fixed buffers of this size so they never allocate metadata.
MAX_RANK :: 64

// Tensor owns its contiguous, row-major data and metadata allocations.
// Treat Tensor values as move-only: copying one aliases its allocations.
Tensor :: struct($T: typeid) {
	data:      []T,
	shape:     []int,
	strides:   []int,
	allocator: mem.Allocator,
}

// Tensor_View borrows its data and metadata and never frees them. The caller
// keeps the source tensor and metadata buffers alive while the view is used.
Tensor_View :: struct($T: typeid) {
	data:    []T,
	shape:   []int,
	strides: []int,
	offset:  int,
}

// _element_count validates a shape and returns the product of its dimensions.
// A rank-zero shape is a scalar with one element. Shapes with more than
// MAX_RANK dimensions are invalid.
_element_count :: proc(shape: []int) -> (count: int, ok: bool) {
	if len(shape) > MAX_RANK {
		return 0, false
	}
	count = 1
	for dim in shape {
		if dim < 0 {
			return 0, false
		}
		if dim == 0 {
			count = 0
			continue
		}
		if count != 0 && count > max(int)/dim {
			return 0, false
		}
		count *= dim
	}
	return count, true
}

// _strides_are_valid ensures row-major strides fit in int, including for empty
// shapes whose dimensions can otherwise hide an overflowing stride.
_strides_are_valid :: proc(shape: []int) -> bool {
	stride := 1
	for i := len(shape)-1; i >= 0; i -= 1 {
		if shape[i] > 0 && stride > max(int)/shape[i] {
			return false
		}
		stride *= shape[i]
	}
	return true
}

_fill_row_major_strides :: proc(shape, strides: []int) {
	stride := 1
	for i := len(shape)-1; i >= 0; i -= 1 {
		strides[i] = stride
		stride *= shape[i]
	}
}

_checked_mul :: proc(a, b: int) -> (value: int, ok: bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > 0 {
		if b > 0 && a > max(int)/b {
			return 0, false
		}
		if b < 0 && b < min(int)/a {
			return 0, false
		}
	} else {
		if b > 0 && a < min(int)/b {
			return 0, false
		}
		if b < 0 && a < max(int)/b {
			return 0, false
		}
	}
	return a*b, true
}

_checked_add :: proc(a, b: int) -> (value: int, ok: bool) {
	if b > 0 && a > max(int)-b {
		return 0, false
	}
	if b < 0 && a < min(int)-b {
		return 0, false
	}
	return a+b, true
}

// new_tensor creates a zero-initialized tensor with the requested shape.
new_tensor :: proc($T: typeid, shape: []int, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	count, valid := _element_count(shape)
	if !valid || !_strides_are_valid(shape) {
		return result, .Invalid_Shape
	}

	owned_shape, alloc_err := make([]int, len(shape), allocator)
	if alloc_err != nil {
		return result, .Allocation_Failed
	}
	for i in 0..<len(shape) {
		owned_shape[i] = shape[i]
	}

	owned_strides: []int
	owned_strides, alloc_err = make([]int, len(shape), allocator)
	if alloc_err != nil {
		delete(owned_shape, allocator)
		return result, .Allocation_Failed
	}
	_fill_row_major_strides(owned_shape, owned_strides)

	values: []T
	values, alloc_err = make([]T, count, allocator)
	if alloc_err != nil {
		delete(owned_shape, allocator)
		delete(owned_strides, allocator)
		return result, .Allocation_Failed
	}
	result = Tensor(T){data = values, shape = owned_shape, strides = owned_strides, allocator = allocator}
	return result, .None
}

// from_slice constructs a tensor by copying values into newly owned storage.
from_slice :: proc($T: typeid, values: []T, shape: []int, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	count, valid := _element_count(shape)
	if !valid || !_strides_are_valid(shape) {
		return result, .Invalid_Shape
	}
	if count != len(values) {
		return result, .Shape_Mismatch
	}

	result, err = new_tensor(T, shape, allocator)
	if err != .None {
		return
	}
	for i in 0..<count {
		result.data[i] = values[i]
	}
	return result, .None
}

// release frees owned storage and resets the tensor. Release each tensor
// exactly once; a shallow copy shares its allocations, so never use or release
// one after the original is released.
release :: proc(t: ^Tensor($T)) {
	delete(t.data, t.allocator)
	delete(t.shape, t.allocator)
	delete(t.strides, t.allocator)
	t^ = {}
}

// reshape changes a tensor's shape without moving its data. As in NumPy, one
// dimension may be -1 and is inferred from the element count. A reshape that
// changes rank replaces the shape and strides allocations, so views created
// earlier with as_view (including borrowed autodiff leaves) become invalid.
reshape :: proc(t: ^Tensor($T), shape: []int) -> Tensor_Error {
	inferred_buffer: [MAX_RANK]int
	new_shape := _reshape_target(shape, len(t.data), inferred_buffer[:]) or_return

	if len(new_shape) == len(t.shape) {
		_fill_row_major_strides(new_shape, t.strides)
		for i in 0..<len(new_shape) {
			t.shape[i] = new_shape[i]
		}
		return .None
	}

	owned_shape, alloc_err := make([]int, len(new_shape), t.allocator)
	if alloc_err != nil {
		return .Allocation_Failed
	}
	owned_strides: []int
	owned_strides, alloc_err = make([]int, len(new_shape), t.allocator)
	if alloc_err != nil {
		delete(owned_shape, t.allocator)
		return .Allocation_Failed
	}
	for i in 0..<len(new_shape) {
		owned_shape[i] = new_shape[i]
	}
	_fill_row_major_strides(owned_shape, owned_strides)
	delete(t.shape, t.allocator)
	delete(t.strides, t.allocator)
	t.shape = owned_shape
	t.strides = owned_strides
	return .None
}

// _reshape_target resolves a reshape request for count elements, inferring
// one -1 dimension into buffer when present.
_reshape_target :: proc(shape: []int, count: int, buffer: []int) -> (new_shape: []int, err: Tensor_Error) {
	new_shape = shape
	inferred_axis := -1
	known := 1
	for dim, axis in shape {
		if dim == -1 && inferred_axis < 0 {
			inferred_axis = axis
		} else if dim < 0 {
			return new_shape, .Invalid_Shape
		} else {
			known *= dim
		}
	}
	if inferred_axis >= 0 {
		if len(shape) > len(buffer) {
			return new_shape, .Invalid_Shape
		}
		if known == 0 || count % known != 0 {
			return new_shape, .Shape_Mismatch
		}
		copy(buffer, shape)
		buffer[inferred_axis] = count / known
		new_shape = buffer[:len(shape)]
	}
	new_count, valid := _element_count(new_shape)
	if !valid || !_strides_are_valid(new_shape) {
		return new_shape, .Invalid_Shape
	}
	if new_count != count {
		return new_shape, .Shape_Mismatch
	}
	return new_shape, .None
}

// as_view creates a non-owning view over a tensor. It allocates no memory;
// the tensor must outlive the returned view.
as_view :: proc(t: Tensor($T)) -> Tensor_View(T) {
	return Tensor_View(T){data = t.data, shape = t.shape, strides = t.strides}
}

// _view_extent returns the inclusive range of data indices a view addresses.
// ok is false for empty views. The view must already be validated.
_view_extent :: proc(view: Tensor_View($T)) -> (first, last: int, ok: bool) {
	first, last = view.offset, view.offset
	for axis in 0..<len(view.shape) {
		if view.shape[axis] == 0 {
			return 0, 0, false
		}
		delta := (view.shape[axis]-1) * view.strides[axis]
		if delta < 0 {
			first += delta
		} else {
			last += delta
		}
	}
	return first, last, true
}

// _output_overlaps_view reports whether any element addressed by a validated
// view shares memory with output's data.
_output_overlaps_view :: proc(output: Tensor($O), view: Tensor_View($T)) -> bool {
	if len(output.data) == 0 {
		return false
	}
	first, last, ok := _view_extent(view)
	if !ok {
		return false
	}
	view_start := uintptr(&view.data[first])
	view_end := uintptr(&view.data[last]) + size_of(T)
	output_start := uintptr(raw_data(output.data))
	output_end := output_start + uintptr(len(output.data)*size_of(O))
	return view_start < output_end && output_start < view_end
}

// fill overwrites every tensor element with value.
fill :: proc(t: ^Tensor($T), value: T) {
	for i in 0..<len(t.data) {
		t.data[i] = value
	}
}

set_linear :: proc(t: ^Tensor($T), index: int, value: T) -> Tensor_Error {
	if index < 0 || index >= len(t.data) {
		return .Out_Of_Bounds
	}
	t.data[index] = value
	return .None
}

get_linear :: proc(t: Tensor($T), index: int) -> (value: T, err: Tensor_Error) {
	if index < 0 || index >= len(t.data) {
		return value, .Out_Of_Bounds
	}
	return t.data[index], .None
}

// _offset_for_indices maps coordinates to a storage offset. As in NumPy, a
// negative index counts from the end of its axis.
_offset_for_indices :: proc(shape, strides: []int, base: int, indices: []int) -> (offset: int, err: Tensor_Error) {
	if len(indices) != len(shape) || len(strides) != len(shape) {
		return 0, .Shape_Mismatch
	}
	offset = base
	for index, axis in indices {
		index := index
		if index < 0 {
			index += shape[axis]
		}
		if index < 0 || index >= shape[axis] {
			return 0, .Out_Of_Bounds
		}
		offset += index * strides[axis]
	}
	return offset, .None
}

// set_at and get_at address a tensor by row-major coordinates. Negative
// indices count from the end; the wrong number of indices is Shape_Mismatch.
set_at :: proc(t: ^Tensor($T), indices: []int, value: T) -> Tensor_Error {
	offset := _offset_for_indices(t.shape, t.strides, 0, indices) or_return
	t.data[offset] = value
	return .None
}

get_at :: proc(t: Tensor($T), indices: []int) -> (value: T, err: Tensor_Error) {
	offset := _offset_for_indices(t.shape, t.strides, 0, indices) or_return
	return t.data[offset], .None
}

// get_at_view and set_at_view use the view's strides and base offset.
get_at_view :: proc(view: Tensor_View($T), indices: []int) -> (value: T, err: Tensor_Error) {
	offset := _offset_for_indices(view.shape, view.strides, view.offset, indices) or_return
	if offset < 0 || offset >= len(view.data) {
		return value, .Out_Of_Bounds
	}
	return view.data[offset], .None
}

set_at_view :: proc(view: Tensor_View($T), indices: []int, value: T) -> Tensor_Error {
	offset := _offset_for_indices(view.shape, view.strides, view.offset, indices) or_return
	if offset < 0 || offset >= len(view.data) {
		return .Out_Of_Bounds
	}
	view.data[offset] = value
	return .None
}

// _normalize_axis maps a NumPy-style axis in [-rank, rank) to [0, rank).
_normalize_axis :: proc(axis, rank: int) -> (int, Tensor_Error) {
	normalized := axis
	if normalized < 0 {
		normalized += rank
	}
	if normalized < 0 || normalized >= rank {
		return 0, .Out_Of_Bounds
	}
	return normalized, .None
}

// _validate_view checks view metadata and that every addressed element lies
// within view.data. Kernels rely on this to index without per-element checks.
_validate_view :: proc(view: Tensor_View($T)) -> Tensor_Error {
	rank := len(view.shape)
	if len(view.strides) != rank {
		return .Invalid_Shape
	}
	if _, valid := _element_count(view.shape); !valid {
		return .Invalid_Shape
	}
	for dim in view.shape {
		if dim == 0 {
			return .None
		}
	}
	min_offset, max_offset := view.offset, view.offset
	for axis in 0..<rank {
		delta, ok := _checked_mul(view.shape[axis]-1, view.strides[axis])
		if !ok {
			return .Out_Of_Bounds
		}
		if delta < 0 {
			min_offset, ok = _checked_add(min_offset, delta)
		} else {
			max_offset, ok = _checked_add(max_offset, delta)
		}
		if !ok {
			return .Out_Of_Bounds
		}
	}
	if min_offset < 0 || max_offset >= len(view.data) {
		return .Out_Of_Bounds
	}
	return .None
}

// _validate_output checks that an owned output tensor is contiguous row-major
// with exactly as many elements as its shape.
_validate_output :: proc(output: Tensor($T)) -> Tensor_Error {
	count, valid := _element_count(output.shape)
	if !valid || !_strides_are_valid(output.shape) || count != len(output.data) || len(output.strides) != len(output.shape) {
		return .Invalid_Shape
	}
	if !_is_contiguous_view(as_view(output)) {
		return .Invalid_Shape
	}
	return .None
}

_shapes_match :: proc(a, b: []int) -> bool {
	if len(a) != len(b) {
		return false
	}
	for i in 0..<len(a) {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

_is_contiguous_view :: proc(view: Tensor_View($T)) -> bool {
	if len(view.strides) != len(view.shape) {
		return false
	}
	stride := 1
	for axis := len(view.shape)-1; axis >= 0; axis -= 1 {
		if view.shape[axis] != 1 && view.strides[axis] != stride {
			return false
		}
		stride *= view.shape[axis]
	}
	return true
}

// _broadcast_offset maps a row-major linear index of result_shape to a storage
// offset in a validated view that broadcasts to result_shape.
_broadcast_offset :: proc(view: Tensor_View($T), result_shape: []int, index: int) -> int {
	offset := view.offset
	remaining := index
	leading := len(result_shape)-len(view.shape)
	for axis := len(result_shape)-1; axis >= leading; axis -= 1 {
		coordinate := remaining % result_shape[axis]
		remaining /= result_shape[axis]
		view_axis := axis-leading
		if view.shape[view_axis] != 1 {
			offset += coordinate * view.strides[view_axis]
		}
	}
	return offset
}

// _inner_step is the storage step between consecutive elements along the last
// axis of result_shape, or zero when that axis is broadcast.
_inner_step :: proc(view: Tensor_View($T), result_shape: []int) -> int {
	rank := len(view.shape)
	if rank == 0 || len(result_shape) == 0 || view.shape[rank-1] == 1 {
		return 0
	}
	return view.strides[rank-1]
}

// _row_length is the length of the last axis, the unit kernels iterate over
// with a fixed step. Rank-zero shapes are a single row of one element.
_row_length :: proc(shape: []int) -> int {
	if len(shape) == 0 {
		return 1
	}
	return shape[len(shape)-1]
}

_copy_into :: proc(output: []$T, source: Tensor_View(T)) {
	count := len(output)
	if count == 0 {
		return
	}
	if _is_contiguous_view(source) {
		copy(output, source.data[source.offset:][:count])
		return
	}
	row := _row_length(source.shape)
	step := _inner_step(source, source.shape)
	for start := 0; start < count; start += row {
		offset := _broadcast_offset(source, source.shape, start)
		for i in start..<start+row {
			output[i] = source.data[offset]
			offset += step
		}
	}
}
