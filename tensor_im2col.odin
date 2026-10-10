package tensor

// Sliding-window gathers for convolution.

import "base:intrinsics"

// im2col gathers every kernel_height × kernel_width window of a channels-last
// image into one row, the usual way to turn a convolution into a single matrix
// product. source is [..., height, width, channels], with any leading batch
// axes, and the result is [..., out_height, out_width,
// kernel_height*kernel_width*channels], where out_height is
// (height-kernel_height)/stride + 1, rounded down, and likewise for width.
// Element (dy*kernel_width + dx)*channels + c of the row at (oy, ox) is
// source[..., oy*stride+dy, ox*stride+dx, c], so multiplying the result by a
// [kernel_height, kernel_width, channels, filters] kernel reshaped to
// [kernel_height*kernel_width*channels, filters] gives a valid convolution.
// A source of rank below 3 or a non-positive kernel size or stride returns
// Invalid_Shape; a kernel larger than the image returns Shape_Mismatch.
im2col :: proc(source: Tensor($T), kernel_height, kernel_width: int, stride := 1, allocator := context.allocator) -> (Tensor(T), Tensor_Error) {
	return im2col_views(as_view(source), kernel_height, kernel_width, stride, allocator)
}

im2col_views :: proc(source: Tensor_View($T), kernel_height, kernel_width: int, stride := 1, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(source) or_return
	shape_buffer: [MAX_RANK]int
	rank := _im2col_shape(source.shape, kernel_height, kernel_width, stride, shape_buffer[:]) or_return
	result = new_tensor(T, shape_buffer[:rank], allocator) or_return
	_im2col_kernel(result.data, source, kernel_height, kernel_width, stride, shape_buffer[:rank])
	return result, .None
}

// im2col_into and im2col_views_into write into an existing output with
// exactly im2col's result shape, which must not overlap source.
im2col_into :: proc(output: ^Tensor($T), source: Tensor(T), kernel_height, kernel_width: int, stride := 1) -> Tensor_Error {
	return im2col_views_into(output, as_view(source), kernel_height, kernel_width, stride)
}

im2col_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), kernel_height, kernel_width: int, stride := 1) -> Tensor_Error {
	_validate_view(source) or_return
	_validate_output(output^) or_return
	shape_buffer: [MAX_RANK]int
	rank := _im2col_shape(source.shape, kernel_height, kernel_width, stride, shape_buffer[:]) or_return
	if !_shapes_match(output.shape, shape_buffer[:rank]) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) {
		return .Aliased_Output
	}
	_im2col_kernel(output.data, source, kernel_height, kernel_width, stride, shape_buffer[:rank])
	return .None
}

// _im2col_shape writes im2col's result shape for source_shape into buffer and
// returns its rank, which equals the source's.
_im2col_shape :: proc(source_shape: []int, kernel_height, kernel_width, stride: int, buffer: []int) -> (rank: int, err: Tensor_Error) {
	rank = len(source_shape)
	if rank < 3 || kernel_height <= 0 || kernel_width <= 0 || stride <= 0 {
		return 0, .Invalid_Shape
	}
	if len(buffer) < rank {
		return 0, .Insufficient_Buffer
	}
	height, width, channels := source_shape[rank-3], source_shape[rank-2], source_shape[rank-1]
	if height < kernel_height || width < kernel_width {
		return 0, .Shape_Mismatch
	}
	window, ok := _checked_mul(kernel_height, kernel_width)
	columns: int
	if ok {
		columns, ok = _checked_mul(window, channels)
	}
	if !ok {
		return 0, .Invalid_Shape
	}
	copy(buffer, source_shape[:rank-3])
	buffer[rank-3] = (height-kernel_height)/stride + 1
	buffer[rank-2] = (width-kernel_width)/stride + 1
	buffer[rank-1] = columns
	if _, valid := _element_count(buffer[:rank]); !valid || !_strides_are_valid(buffer[:rank]) {
		return 0, .Invalid_Shape
	}
	return rank, .None
}

// _im2col_kernel fills the contiguous output, whose shape result_shape comes
// from _im2col_shape, from a validated source, splitting the (image, output
// row) pairs between threads. When pixels and channels are contiguous, each
// window row (oy, ox, dy) is one copy of kernel_width*channels elements.
_im2col_kernel :: proc(output: []$T, source: Tensor_View(T), kernel_height, kernel_width, stride: int, result_shape: []int) {
	if len(output) == 0 {
		return
	}
	rank := len(source.shape)
	job := _Im2col_Job(T){output = output, source = source, kernel_height = kernel_height, kernel_width = kernel_width, stride = stride}
	job.batch = Tensor_View(T){data = source.data, shape = source.shape[:rank-3], strides = source.strides[:rank-3], offset = source.offset}
	batch_count, _ := _element_count(job.batch.shape)
	job.out_height, job.out_width, job.columns = result_shape[rank-3], result_shape[rank-2], result_shape[rank-1]
	_parallel_ranges(batch_count*job.out_height, 1, proc(data: rawptr, first, last: int) {
		_im2col_rows((^_Im2col_Job(T))(data), first, last)
	}, &job, job.out_width*job.columns)
}

_Im2col_Job :: struct($T: typeid) {
	output:                            []T,
	source:                            Tensor_View(T),
	batch:                             Tensor_View(T),
	kernel_height, kernel_width:       int,
	stride:                            int,
	out_height, out_width, columns:    int,
}

// _im2col_rows fills the output rows for (image, output row) pairs first..<last.
_im2col_rows :: proc(job: ^_Im2col_Job($T), first, last: int) #no_bounds_check {
	source, output := job.source, job.output
	rank := len(source.shape)
	channels := source.shape[rank-1]
	row_stride, pixel_stride, channel_stride := source.strides[rank-3], source.strides[rank-2], source.strides[rank-1]
	contiguous_rows := channel_stride == 1 && pixel_stride == channels
	row_length := job.kernel_width * channels
	next := first * job.out_width * job.columns
	for pair in first..<last {
		batch, oy := pair / job.out_height, pair % job.out_height
		base := _broadcast_offset(job.batch, job.batch.shape, batch)
		for ox in 0..<job.out_width {
			for dy in 0..<job.kernel_height {
				start := base + (oy*job.stride+dy)*row_stride + ox*job.stride*pixel_stride
				if contiguous_rows {
					copy(output[next:][:row_length], source.data[start:][:row_length])
					next += row_length
					continue
				}
				for dx in 0..<job.kernel_width {
					pixel := start + dx*pixel_stride
					for c in 0..<channels {
						output[next] = source.data[pixel + c*channel_stride]
						next += 1
					}
				}
			}
		}
	}
}

// _backward_im2col adds each upstream value back to the input element im2col
// copied it from, so elements shared by overlapping windows sum their
// gradients. gradient is contiguous in input's logical shape.
_backward_im2col :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), kernel_height, kernel_width, stride: int) -> Tensor_Error where intrinsics.type_is_float(T) {
	shape_buffer: [MAX_RANK]int
	rank := _im2col_shape(input.shape, kernel_height, kernel_width, stride, shape_buffer[:]) or_return
	if !_shapes_match(output_grad.shape, shape_buffer[:rank]) || !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	if len(output_grad.data) == 0 {
		return .None
	}
	// Overlapping windows of one image add into the same elements, so the
	// images, whose gradients don't overlap, are what threads split.
	job := _Im2col_Backward_Job(T){gradient = gradient.data, upstream = output_grad.data, kernel_height = kernel_height, kernel_width = kernel_width, stride = stride}
	job.height, job.width, job.channels = input.shape[rank-3], input.shape[rank-2], input.shape[rank-1]
	job.out_height, job.out_width = shape_buffer[rank-3], shape_buffer[rank-2]
	image_size := job.height * job.width * job.channels
	_parallel_ranges(len(gradient.data)/image_size, 1, proc(data: rawptr, first, last: int) {
		_backward_im2col_images((^_Im2col_Backward_Job(T))(data), first, last)
	}, &job, len(output_grad.data)/(len(gradient.data)/image_size))
	return .None
}

_Im2col_Backward_Job :: struct($T: typeid) {
	gradient, upstream:              []T,
	kernel_height, kernel_width:     int,
	stride:                          int,
	height, width, channels:         int,
	out_height, out_width:           int,
}

_backward_im2col_images :: proc(job: ^_Im2col_Backward_Job($T), first, last: int) #no_bounds_check {
	image_size := job.height * job.width * job.channels
	row_length := job.kernel_width * job.channels
	window_size := job.kernel_height * row_length
	next := first * job.out_height * job.out_width * window_size
	for batch in first..<last {
		base := batch * image_size
		for oy in 0..<job.out_height {
			for ox in 0..<job.out_width {
				for dy in 0..<job.kernel_height {
					target := job.gradient[base + ((oy*job.stride+dy)*job.width + ox*job.stride)*job.channels:][:row_length]
					for &value, i in target {
						value += job.upstream[next+i]
					}
					next += row_length
				}
			}
		}
	}
}
