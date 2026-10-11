package tensor

// Borrowed view construction and owned copies of tensor views.

// transpose_view reverses the axis order, the usual transpose for a matrix.
// The caller's buffers hold the view metadata. Each needs at least rank
// elements, must not alias source.shape or source.strides, and must stay alive
// and unchanged while the view is used.
transpose_view :: proc(source: Tensor_View($T), shape_buffer, strides_buffer: []int) -> (view: Tensor_View(T), err: Tensor_Error) {
	rank := len(source.shape)
	if len(source.strides) != rank {
		return view, .Invalid_Shape
	}
	if len(shape_buffer) < rank || len(strides_buffer) < rank {
		return view, .Insufficient_Buffer
	}
	if !_strides_are_valid(source.shape) {
		return view, .Invalid_Shape
	}
	for i in 0..<rank {
		shape_buffer[i] = source.shape[rank-1-i]
		strides_buffer[i] = source.strides[rank-1-i]
	}
	view = Tensor_View(T){
		data    = source.data,
		shape   = shape_buffer[:rank],
		strides = strides_buffer[:rank],
		offset  = source.offset,
	}
	return view, .None
}

// permute_view reorders axes without copying, like np.transpose(a, axes):
// result axis i is source axis axes[i]. Negative axes count from the end.
// Buffer rules match transpose_view.
permute_view :: proc(source: Tensor_View($T), axes, shape_buffer, strides_buffer: []int) -> (view: Tensor_View(T), err: Tensor_Error) {
	rank := len(source.shape)
	if len(source.strides) != rank || len(axes) != rank {
		return view, .Invalid_Shape
	}
	if len(shape_buffer) < rank || len(strides_buffer) < rank {
		return view, .Insufficient_Buffer
	}
	seen: [MAX_RANK]bool
	if rank > MAX_RANK {
		return view, .Invalid_Shape
	}
	for axis, i in axes {
		source_axis := _normalize_axis(axis, rank) or_return
		if seen[source_axis] {
			return view, .Invalid_Shape
		}
		seen[source_axis] = true
		shape_buffer[i] = source.shape[source_axis]
		strides_buffer[i] = source.strides[source_axis]
	}
	view = Tensor_View(T){
		data    = source.data,
		shape   = shape_buffer[:rank],
		strides = strides_buffer[:rank],
		offset  = source.offset,
	}
	return view, .None
}

// copy_view makes an owned, contiguous, row-major copy of any valid view, like
// np.ascontiguousarray.
copy_view :: proc(source: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	result = _new_tensor(T, source.shape, false, allocator) or_return
	_copy_into(result.data, source)
	return result, .None
}

// transpose_copy creates an owned copy with the axis order reversed. Unlike
// transpose_view, its metadata and data do not borrow the source.
transpose_copy :: proc(source: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	shape_buffer, strides_buffer: [MAX_RANK]int
	transposed := transpose_view(source, shape_buffer[:], strides_buffer[:]) or_return
	return copy_view(transposed, allocator)
}

transpose_copy_tensor :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) {
	return transpose_copy(as_view(source), allocator)
}

// slice_view selects count elements along one axis, starting at start and
// advancing by step. NumPy's a[start:stop:step] takes a stop; a count gives
// reverse and empty slices unambiguous bounds. Negative axis, start, and step
// count from the end. The metadata buffers must not overlap and must stay alive
// and unchanged while the view is used.
slice_view :: proc(source: Tensor_View($T), axis, start, count, step: int, shape_buffer, strides_buffer: []int) -> (view: Tensor_View(T), err: Tensor_Error) {
	rank := len(source.shape)
	if len(source.strides) != rank {
		return view, .Invalid_Shape
	}
	if len(shape_buffer) < rank || len(strides_buffer) < rank {
		return view, .Insufficient_Buffer
	}
	slice_axis := _normalize_axis(axis, rank) or_return
	first := start
	if first < 0 {
		first += source.shape[slice_axis]
	}
	if step == 0 || count < 0 {
		return view, .Invalid_Shape
	}

	new_offset := source.offset
	if count == 0 {
		if first < 0 || first > source.shape[slice_axis] {
			return view, .Out_Of_Bounds
		}
	} else {
		delta, delta_ok := _checked_mul(count-1, step)
		if !delta_ok {
			return view, .Out_Of_Bounds
		}
		last, last_ok := _checked_add(first, delta)
		if !last_ok || first < 0 || first >= source.shape[slice_axis] || last < 0 || last >= source.shape[slice_axis] {
			return view, .Out_Of_Bounds
		}
		storage_delta, storage_ok := _checked_mul(first, source.strides[slice_axis])
		if !storage_ok {
			return view, .Out_Of_Bounds
		}
		new_offset, storage_ok = _checked_add(source.offset, storage_delta)
		if !storage_ok || new_offset < 0 || new_offset >= len(source.data) {
			return view, .Out_Of_Bounds
		}
	}
	new_stride, stride_ok := _checked_mul(source.strides[slice_axis], step)
	if !stride_ok {
		return view, .Invalid_Shape
	}
	for i in 0..<rank {
		shape_buffer[i] = source.shape[i]
		strides_buffer[i] = source.strides[i]
	}
	shape_buffer[slice_axis] = count
	strides_buffer[slice_axis] = new_stride
	view = Tensor_View(T){
		data    = source.data,
		shape   = shape_buffer[:rank],
		strides = strides_buffer[:rank],
		offset  = new_offset,
	}
	return view, .None
}

// broadcast_view expands source to target_shape without copying. Dimensions
// align from the right, and a source dimension of one gets a zero stride.
// shape_buffer and strides_buffer each need len(target_shape) elements, must
// not overlap each other or source's metadata, and must stay alive and
// unchanged while the view is used.
broadcast_view :: proc(source: Tensor_View($T), target_shape, shape_buffer, strides_buffer: []int) -> (view: Tensor_View(T), err: Tensor_Error) {
	source_rank := len(source.shape)
	target_rank := len(target_shape)
	if len(source.strides) != source_rank {
		return view, .Invalid_Shape
	}
	if len(shape_buffer) < target_rank || len(strides_buffer) < target_rank {
		return view, .Insufficient_Buffer
	}
	if source_rank > target_rank {
		return view, .Shape_Mismatch
	}
	if _, valid := _element_count(source.shape); !valid {
		return view, .Invalid_Shape
	}
	if _, valid := _element_count(target_shape); !valid || !_strides_are_valid(target_shape) {
		return view, .Invalid_Shape
	}

	leading := target_rank-source_rank
	for axis in 0..<target_rank {
		target_dim := target_shape[axis]
		shape_buffer[axis] = target_dim
		if axis < leading {
			strides_buffer[axis] = 0
			continue
		}
		source_axis := axis-leading
		source_dim := source.shape[source_axis]
		if source_dim != target_dim && source_dim != 1 {
			return view, .Shape_Mismatch
		}
		if source_dim == 1 && target_dim != 1 {
			strides_buffer[axis] = 0
		} else {
			strides_buffer[axis] = source.strides[source_axis]
		}
	}
	view = Tensor_View(T){
		data    = source.data,
		shape   = shape_buffer[:target_rank],
		strides = strides_buffer[:target_rank],
		offset  = source.offset,
	}
	return view, .None
}
