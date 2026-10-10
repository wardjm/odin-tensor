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
	result = new_tensor(T, shape_buffer[:rank], allocator) or_return
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

// _max_pool2d_kernel fills the contiguous output from a validated source. For
// each window it starts every channel at the window's first pixel and folds
// the remaining pixels in row-major order with max_axis's rule, a loop over
// channels that the compiler can vectorize when channels are contiguous.
_max_pool2d_kernel :: proc(output: []$T, source: Tensor_View(T), pool_height, pool_width: int, result_shape: []int) #no_bounds_check {
	if len(output) == 0 {
		return
	}
	g := _pool_geometry(source, result_shape)
	data := source.data
	next := 0
	for batch in 0..<g.batch_count {
		base := _broadcast_offset(g.batch, g.batch.shape, batch)
		for oy in 0..<g.out_height {
			for ox in 0..<g.out_width {
				window := base + oy*pool_height*g.row_stride + ox*pool_width*g.pixel_stride
				out := output[next:][:g.channels]
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
}

// _backward_max_pool2d adds each upstream value to the one input element of
// its window that argmax would pick, scanning in row-major order: the first
// NaN, or else the first maximum. gradient is contiguous in input's logical
// shape.
_backward_max_pool2d :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), pool_height, pool_width: int) -> Tensor_Error where intrinsics.type_is_float(T) #no_bounds_check {
	_validate_view(input) or_return
	shape_buffer: [MAX_RANK]int
	rank := _pool_shape(input.shape, pool_height, pool_width, shape_buffer[:]) or_return
	if !_shapes_match(output_grad.shape, shape_buffer[:rank]) || !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	if len(output_grad.data) == 0 {
		return .None
	}
	g := _pool_geometry(input, shape_buffer[:rank])
	width := input.shape[rank-2]
	image_size := input.shape[rank-3] * width * g.channels
	data := input.data
	next := 0
	for batch in 0..<g.batch_count {
		base := _broadcast_offset(g.batch, g.batch.shape, batch)
		gradient_base := batch * image_size
		for oy in 0..<g.out_height {
			for ox in 0..<g.out_width {
				window := base + oy*pool_height*g.row_stride + ox*pool_width*g.pixel_stride
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
					gradient.data[gradient_base + (y*width + x)*g.channels + c] += output_grad.data[next]
					next += 1
				}
			}
		}
	}
	return .None
}
