package tensor

// Window pooling for convolutional networks.

import "base:intrinsics"

// max_pool2d takes the maximum of each non-overlapping pool_height ×
// pool_width window of a channels-last image, separately per channel. source
// is [..., height, width, channels], with any leading batch axes, and the
// result is [..., height/pool_height, width/pool_width, channels], rounding
// down, so rows and columns that do not fill a window are ignored. As in
// max_axis, a NaN in a window wins. A source of rank below 3 or a
// non-positive pool size returns Invalid_Shape; a window larger than the image
// returns Shape_Mismatch.
max_pool2d :: proc(source: Tensor($T), pool_height, pool_width: int, allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	return max_pool2d_views(as_view(source), pool_height, pool_width, allocator)
}

max_pool2d_views :: proc(source: Tensor_View($T), pool_height, pool_width: int, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) where intrinsics.type_is_ordered_numeric(T) {
	_validate_view(source) or_return
	shape_buffer: [MAX_RANK]int
	rank := _pool_shape(source.shape, pool_height, pool_width, shape_buffer[:]) or_return
	result = _new_tensor(T, shape_buffer[:rank], false, allocator) or_return
	_max_pool2d_kernel(result.data, source, pool_height, pool_width, shape_buffer[:rank])
	return result, .None
}

// max_pool2d_into and max_pool2d_views_into write into an existing output
// with exactly max_pool2d's result shape, which must not overlap source.
max_pool2d_into :: proc(output: ^Tensor($T), source: Tensor(T), pool_height, pool_width: int) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	return max_pool2d_views_into(output, as_view(source), pool_height, pool_width)
}

max_pool2d_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), pool_height, pool_width: int) -> Tensor_Error where intrinsics.type_is_ordered_numeric(T) {
	_validate_view(source) or_return
	_validate_output(output^) or_return
	shape_buffer: [MAX_RANK]int
	rank := _pool_shape(source.shape, pool_height, pool_width, shape_buffer[:]) or_return
	if !_shapes_match(output.shape, shape_buffer[:rank]) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}
	_max_pool2d_kernel(output.data, source, pool_height, pool_width, shape_buffer[:rank])
	return .None
}

// _pool_shape writes max_pool2d's result shape for source_shape into buffer
// and returns its rank, which equals the source's.
_pool_shape :: proc(source_shape: []int, pool_height, pool_width: int, buffer: []int) -> (rank: int, err: Tensor_Error) {
	rank = len(source_shape)
	if rank < 3 || pool_height <= 0 || pool_width <= 0 {
		return 0, .Invalid_Shape
	}
	if len(buffer) < rank {
		return 0, .Insufficient_Buffer
	}
	if source_shape[rank-3] < pool_height || source_shape[rank-2] < pool_width {
		return 0, .Shape_Mismatch
	}
	copy(buffer, source_shape[:rank])
	buffer[rank-3] = source_shape[rank-3] / pool_height
	buffer[rank-2] = source_shape[rank-2] / pool_width
	return rank, .None
}

// _Pool_Geometry is what the pooling kernels need to walk a validated source:
// a view of its batch axes and the strides of its image axes.
_Pool_Geometry :: struct($T: typeid) {
	batch:          Tensor_View(T),
	batch_count:    int,
	out_height:     int,
	out_width:      int,
	channels:       int,
	row_stride:     int,
	pixel_stride:   int,
	channel_stride: int,
}

_pool_geometry :: proc(source: Tensor_View($T), result_shape: []int) -> (geometry: _Pool_Geometry(T)) {
	rank := len(source.shape)
	geometry.batch = Tensor_View(T){data = source.data, shape = source.shape[:rank-3], strides = source.strides[:rank-3], offset = source.offset}
	geometry.batch_count, _ = _element_count(geometry.batch.shape)
	geometry.out_height, geometry.out_width = result_shape[rank-3], result_shape[rank-2]
	geometry.channels = source.shape[rank-1]
	geometry.row_stride, geometry.pixel_stride, geometry.channel_stride = source.strides[rank-3], source.strides[rank-2], source.strides[rank-1]
	return geometry
}

// _max_pool2d_kernel fills the contiguous output from a validated source,
// splitting the (image, output row) pairs between threads. For each window it
// starts every channel at the window's first pixel and folds the remaining
// pixels in row-major order with max_axis's rule, a loop over channels that
// the compiler can vectorize when channels are contiguous.
_max_pool2d_kernel :: proc(output: []$T, source: Tensor_View(T), pool_height, pool_width: int, result_shape: []int) {
	if len(output) == 0 {
		return
	}
	job := _Pool_Job(T){output = output, source = source, pool_height = pool_height, pool_width = pool_width, g = _pool_geometry(source, result_shape)}
	_parallel_ranges(job.g.batch_count*job.g.out_height, 1, proc(data: rawptr, first, last: int) {
		_max_pool2d_rows((^_Pool_Job(T))(data), first, last)
	}, &job, job.g.out_width*job.g.channels*pool_height*pool_width)
}

_Pool_Job :: struct($T: typeid) {
	output, gradient:         []T,
	upstream:                 []T,
	source:                   Tensor_View(T),
	pool_height, pool_width:  int,
	g:                        _Pool_Geometry(T),
	width:                    int,
	store:                    bool,
}

_max_pool2d_rows :: proc(job: ^_Pool_Job($T), first, last: int) #no_bounds_check {
	g := job.g
	data := job.source.data
	pool_height, pool_width := job.pool_height, job.pool_width
	next := first * g.out_width * g.channels
	for pair in first..<last {
		batch, oy := pair / g.out_height, pair % g.out_height
		base := _broadcast_offset(g.batch, g.batch.shape, batch)
		for ox in 0..<g.out_width {
			window := base + oy*pool_height*g.row_stride + ox*pool_width*g.pixel_stride
			out := job.output[next:][:g.channels]
			for &value, c in out {
				value = data[window + c*g.channel_stride]
			}
			for dy in 0..<pool_height {
				for dx in 0..<pool_width {
					if dy == 0 && dx == 0 {
						continue
					}
					pixel := window + dy*g.row_stride + dx*g.pixel_stride
					if g.channel_stride == 1 {
						row := data[pixel:][:g.channels]
						for &value, c in out {
							value = _combine(value, row[c], _Reduction_Operation.Max)
						}
						continue
					}
					for &value, c in out {
						value = _combine(value, data[pixel + c*g.channel_stride], _Reduction_Operation.Max)
					}
				}
			}
			next += g.channels
		}
	}
}

// _backward_max_pool2d adds each upstream value to the one input element of
// its window that argmax would pick, scanning in row-major order: the first
// NaN, or else the first maximum. gradient is contiguous in input's logical
// shape. With store, gradient holds no values yet: each (image, output row)
// pair first zeroes the rows of it that its windows cover, and the last pair
// of an image the rows below that no window covers.
_backward_max_pool2d :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), pool_height, pool_width: int, store := false) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_view(input) or_return
	shape_buffer: [MAX_RANK]int
	rank := _pool_shape(input.shape, pool_height, pool_width, shape_buffer[:]) or_return
	if !_shapes_match(output_grad.shape, shape_buffer[:rank]) || !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	if len(output_grad.data) == 0 {
		if store {
			_fill_zero(gradient.data)
		}
		return .None
	}
	// Windows don't overlap, so each (image, output row) pair writes its own
	// rows of the gradient.
	job := _Pool_Job(T){gradient = gradient.data, upstream = output_grad.data, source = input, pool_height = pool_height, pool_width = pool_width, g = _pool_geometry(input, shape_buffer[:rank]), width = input.shape[rank-2], store = store}
	_parallel_ranges(job.g.batch_count*job.g.out_height, 1, proc(data: rawptr, first, last: int) {
		_backward_max_pool2d_rows((^_Pool_Job(T))(data), first, last)
	}, &job, job.g.out_width*job.g.channels*pool_height*pool_width)
	return .None
}

_backward_max_pool2d_rows :: proc(job: ^_Pool_Job($T), first, last: int) #no_bounds_check {
	g := job.g
	data := job.source.data
	pool_height, pool_width := job.pool_height, job.pool_width
	rank := len(job.source.shape)
	height := job.source.shape[rank-3]
	row_size := job.width * g.channels
	image_size := height * row_size
	next := first * g.out_width * g.channels
	for pair in first..<last {
		batch, oy := pair / g.out_height, pair % g.out_height
		base := _broadcast_offset(g.batch, g.batch.shape, batch)
		gradient_base := batch * image_size
		if job.store {
			last_row := (oy+1)*pool_height if oy < g.out_height-1 else height
			_fill_zero(job.gradient[gradient_base + oy*pool_height*row_size:gradient_base + last_row*row_size])
		}
		for ox in 0..<g.out_width {
			window := base + oy*pool_height*g.row_stride + ox*pool_width*g.pixel_stride
			if g.channel_stride == 1 {
				gradient_window := gradient_base + (oy*pool_height*job.width + ox*pool_width)*g.channels
				for first_channel := 0; first_channel < g.channels; first_channel += _POOL_CHANNEL_BLOCK {
					count := min(_POOL_CHANNEL_BLOCK, g.channels - first_channel)
					_backward_max_pool2d_channels(job, window + first_channel, gradient_window + first_channel, job.upstream[next:][:count])
					next += count
				}
				continue
			}
			for c in 0..<g.channels {
				channel := window + c*g.channel_stride
				best := data[channel]
				best_dy, best_dx := 0, 0
				scan: for dy in 0..<pool_height {
					for dx in 0..<pool_width {
						if best != best {
							break scan
						}
						element := data[channel + dy*g.row_stride + dx*g.pixel_stride]
						if _arg_is_better(element, best, _Reduction_Operation.Max) {
							best, best_dy, best_dx = element, dy, dx
						}
					}
				}
				y, x := oy*pool_height + best_dy, ox*pool_width + best_dx
				job.gradient[gradient_base + (y*job.width + x)*g.channels + c] += job.upstream[next]
				next += 1
			}
		}
	}
}

// _POOL_CHANNEL_BLOCK channels of a window are handled together by
// _backward_max_pool2d_channels.
_POOL_CHANNEL_BLOCK :: 64

// _backward_max_pool2d_channels adds upstream, one value per channel, to the
// element of each channel's window that argmax picks, for len(upstream)
// contiguous channels starting at source offset window and gradient offset
// gradient_window. It scans every channel at once with selects rather than
// branches, keeping the rule of _backward_max_pool2d_rows: a position takes
// over unless the best so far is NaN, when it is larger or NaN.
_backward_max_pool2d_channels :: proc(job: ^_Pool_Job($T), window, gradient_window: int, upstream: []T) #no_bounds_check {
	g := job.g
	data := job.source.data
	count := len(upstream)
	best: [_POOL_CHANNEL_BLOCK]T
	position: [_POOL_CHANNEL_BLOCK]T
	copy(best[:count], data[window:][:count])
	k: T = 0
	for dy in 0..<job.pool_height {
		for dx in 0..<job.pool_width {
			row := data[window + dy*g.row_stride + dx*g.pixel_stride:][:count]
			for c in 0..<count {
				element := row[c]
				// & and | rather than && and ||, which would branch.
				take := (best[c] == best[c]) & ((element > best[c]) | (element != element))
				best[c] = element if take else best[c]
				position[c] = k if take else position[c]
			}
			k += 1
		}
	}
	k = 0
	for dy in 0..<job.pool_height {
		for dx in 0..<job.pool_width {
			gradient := job.gradient[gradient_window + (dy*job.width + dx)*g.channels:][:count]
			for c in 0..<count {
				gradient[c] = gradient[c] + upstream[c] if position[c] == k else gradient[c]
			}
			k += 1
		}
	}
}
