package tensor

// High-level autodiff operations: each one evaluates the forward result from
// its parents' tape values and records it, so the recorded operation always
// matches the computation.

import "base:intrinsics"

// _validate_autodiff_view checks metadata and ensures every logical element is
// within its backing storage. Empty views may point at the one-past-end offset.
_validate_autodiff_view :: proc(value: Tensor_View($T)) -> Tensor_Error {
	if count, valid := _element_count(value.shape); valid && count == 0 && len(value.strides) == len(value.shape) {
		if value.offset < 0 || value.offset > len(value.data) {
			return .Out_Of_Bounds
		}
		return .None
	}
	return _validate_view(value)
}

// tape_add, tape_sub, tape_mul, and tape_div broadcast like add/sub/mul/div.
tape_add :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Add)
}

tape_sub :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Sub)
}

tape_mul :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Mul)
}

tape_div :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Div)
}

// tape_dot follows dot's rank-one/rank-two rules; batched (rank > 2) operands
// return Invalid_Shape. tape_matmul follows matmul, including broadcast batch
// axes, whose gradients sum over the batches each operand was broadcast to.
tape_dot :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Dot)
}

tape_matmul :: proc(tape: ^Tape($T), a, b: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_binary(tape, a, b, .Matmul)
}

tape_relu :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Relu)
}

tape_sigmoid :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Sigmoid)
}

tape_tanh :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Tanh)
}

tape_exp :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Exp)
}

// tape_log of zero records -Inf and gives an infinite gradient (NaN where the
// upstream gradient is zero). Negative inputs give NaN, as in log.
tape_log :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Log)
}

// tape_sqrt of zero gives an infinite gradient (NaN where the upstream gradient
// is zero). Negative inputs give NaN, as in sqrt.
tape_sqrt :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_unary(tape, x, .Sqrt)
}

tape_sum_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Sum_All, -1, false)
}

tape_mean_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Mean_All, -1, false)
}

// tape_sum_axis and tape_mean_axis accept a negative axis.
tape_sum_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Sum_Axis, axis, keep_dims)
}

tape_mean_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Mean_Axis, axis, keep_dims)
}

// tape_prod_all and tape_prod_axis give each element the product of the others
// as its derivative, built from prefix and suffix products rather than by
// dividing the result, so zeros get exact gradients. An empty product is 1.
tape_prod_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Prod_All, -1, false)
}

tape_prod_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Prod_Axis, axis, keep_dims)
}

// tape_max_all, tape_min_all, tape_max_axis, and tape_min_axis send each
// upstream value to the single element argmax or argmin picks: the first on a
// tie, and the first NaN if there is one. An empty reduction returns
// Empty_Reduction.
tape_max_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Max_All, -1, false)
}

tape_min_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Min_All, -1, false)
}

tape_max_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Max_Axis, axis, keep_dims)
}

tape_min_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Min_Axis, axis, keep_dims)
}

// tape_transpose, tape_slice, and tape_reshape record views of x. Their
// values borrow x's storage instead of copying it, except that tape_reshape
// copies an x that is not contiguous. The gradient of each is the upstream
// gradient placed back through the same view of x's gradient.

// tape_transpose reverses the axis order, like transpose_view.
tape_transpose :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_view(tape, x, .Transpose, 0, 0, 0, 0, nil)
}

// tape_slice selects count elements along axis from start by step, with
// slice_view's rules: a negative axis, start, or step counts from the end.
tape_slice :: proc(tape: ^Tape($T), x: Tape_Value(T), axis, start, count: int, step := 1) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_view(tape, x, .Slice, axis, start, count, step, nil)
}

// tape_reshape gives x a new shape with the same row-major elements, inferring
// one -1 dimension, like reshape.
tape_reshape :: proc(tape: ^Tape($T), x: Tape_Value(T), shape: []int) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_view(tape, x, .Reshape, 0, 0, 0, 0, shape)
}

// tape_im2col records im2col of x, which must be [..., height, width,
// channels]. Its gradient adds each upstream value back to the element of x it
// was copied from, so elements in overlapping windows sum their gradients;
// values that came from padding have no element and are dropped.
tape_im2col :: proc(tape: ^Tape($T), x: Tape_Value(T), kernel_height, kernel_width: int, stride := 1, padding := 0) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	parent_node := _tape_node(tape, x)
	if parent_node == nil {
		return result, .Invalid_Tape
	}
	output := im2col_views(parent_node.value, kernel_height, kernel_width, stride, padding, tape.allocator) or_return
	defer release(&output)
	return _tape_take_output(tape, &output, _Tape_Node(T){
		requires_grad = parent_node.requires_grad,
		operation     = .Im2col,
		parent_a      = x.id,
		parent_b      = -1,
		parent_count  = 1,
		step          = stride,
		padding       = padding,
		kernel_height = kernel_height,
		kernel_width  = kernel_width,
	})
}

// tape_batch_norm records batch normalization of x over every axis but the
// last, multiplied by scale: element (..., f) becomes
// (x - mean_f) / sqrt(variance_f + epsilon) * scale_f, where mean_f and the
// biased variance_f are taken over every other axis, and scale is a
// [features] value. mean and variance, when given, receive each feature's
// batch statistics, for callers that keep running estimates. The gradient
// flows through the statistics, as for PyTorch's batch_norm in training mode;
// add a shift with tape_add. One fused node replaces the dozen element-wise
// operations and reductions this would otherwise take, and its backward pass
// is two passes over the gradient. An x of rank zero, a scale that is not
// [features], or a mean or variance of another length returns Shape_Mismatch,
// and an x with no elements returns Empty_Reduction.
tape_batch_norm :: proc(tape: ^Tape($T), x, scale: Tape_Value(T), epsilon: T, mean: []T = nil, variance: []T = nil) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	x_node, scale_node := _tape_node(tape, x), _tape_node(tape, scale)
	if x_node == nil || scale_node == nil {
		return result, .Invalid_Tape
	}
	x_value, scale_value := x_node.value, scale_node.value
	rank := len(x_value.shape)
	if rank == 0 || len(scale_value.shape) != 1 || scale_value.shape[0] != x_value.shape[rank-1] {
		return result, .Shape_Mismatch
	}
	features := x_value.shape[rank-1]
	if (mean != nil && len(mean) != features) || (variance != nil && len(variance) != features) {
		return result, .Shape_Mismatch
	}
	count, _ := _element_count(x_value.shape)
	if count == 0 {
		return result, .Empty_Reduction
	}
	rows := count / features

	contiguous: Tensor(T)
	defer release(&contiguous)
	source := x_value
	if !_is_contiguous_view(source) {
		contiguous = copy_view(source, tape.allocator) or_return
		source = as_view(contiguous)
	}
	// stats holds each feature's mean, then its standard deviation, for the
	// backward pass, then its scale and variance, which are dropped after.
	stats := new_tensor(T, []int{4, features}, tape.allocator) or_return
	defer release(&stats)
	means, deviations := stats.data[:features], stats.data[features:2*features]
	scales, variances := stats.data[2*features:3*features], stats.data[3*features:]
	_column_sums(means, source.data, source.offset, rows, features, nil, nil, nil, .Value)
	for &value in means {
		value /= T(rows)
	}
	_column_sums(variances, source.data, source.offset, rows, features, means, nil, nil, .Squared_Deviation)
	for f in 0..<features {
		variances[f] /= T(rows)
		deviations[f] = intrinsics.sqrt(variances[f] + epsilon)
		scales[f] = scale_value.data[scale_value.offset + f*scale_value.strides[0]]
	}
	if mean != nil {
		copy(mean, means)
	}
	if variance != nil {
		copy(variance, variances)
	}

	output := _new_tensor(T, x_value.shape, false, tape.allocator) or_return
	defer release(&output)
	_batch_norm_rows(output.data, source.data[source.offset:], stats.data, rows, features)
	saved := new_tensor(T, []int{2, features}, tape.allocator) or_return
	copy(saved.data, stats.data[:2*features])
	result, err = _tape_take_output(tape, &output, _Tape_Node(T){
		requires_grad = x_node.requires_grad || scale_node.requires_grad,
		operation     = .Batch_Norm,
		parent_a      = x.id,
		parent_b      = scale.id,
		parent_count  = 2,
		saved         = saved,
	})
	if err != .None {
		release(&saved)
	}
	return result, err
}

// _batch_norm_rows writes (x - mean) / deviation * scale for each of rows
// contiguous rows of features, split between threads. stats holds the means,
// deviations and scales, features each.
_batch_norm_rows :: proc(output, x, stats: []$T, rows, features: int) {
	Job :: struct {
		output, x, stats: []T,
		features:         int,
	}
	job := Job{output, x, stats, features}
	_parallel_ranges(rows, 1, proc(data: rawptr, first, last: int) {
		job := (^Job)(data)
		features := job.features
		means, deviations, scales := job.stats[:features], job.stats[features:2*features], job.stats[2*features:3*features]
		for row in first..<last {
			x := job.x[row*features:][:features]
			output := job.output[row*features:][:features]
			for f in 0..<features {
				output[f] = (x[f] - means[f]) / deviations[f] * scales[f]
			}
		}
	}, &job, features)
}

// tape_max_pool2d records max_pool2d of x. Each upstream value goes to the one
// element of its window that argmax picks: the first NaN, or else the first
// maximum in row-major order. PyTorch also sends a tie's gradient to a single
// element.
tape_max_pool2d :: proc(tape: ^Tape($T), x: Tape_Value(T), pool_height, pool_width: int) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	parent_node := _tape_node(tape, x)
	if parent_node == nil {
		return result, .Invalid_Tape
	}
	output := max_pool2d_views(parent_node.value, pool_height, pool_width, tape.allocator) or_return
	defer release(&output)
	return _tape_take_output(tape, &output, _Tape_Node(T){
		requires_grad = parent_node.requires_grad,
		operation     = .Max_Pool2D,
		parent_a      = x.id,
		parent_b      = -1,
		parent_count  = 1,
		kernel_height = pool_height,
		kernel_width  = pool_width,
	})
}

_tape_binary :: proc(tape: ^Tape($T), a, b: Tape_Value(T), operation: Tape_Operation) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	a_view := tape_value(tape, a) or_return
	b_view := tape_value(tape, b) or_return
	output: Tensor(T)
	#partial switch operation {
	case .Add:
		output = add_views(a_view, b_view, tape.allocator) or_return
	case .Sub:
		output = sub_views(a_view, b_view, tape.allocator) or_return
	case .Mul:
		output = mul_views(a_view, b_view, tape.allocator) or_return
	case .Div:
		output = div_views(a_view, b_view, tape.allocator) or_return
	case .Dot:
		output = dot_views(a_view, b_view, tape.allocator) or_return
	case .Matmul:
		output = matmul_views(a_view, b_view, tape.allocator) or_return
	case:
		return result, .Unsupported_Operation
	}
	result, err = tape_record_binary(tape, &output, operation, a, b)
	release(&output)
	return
}

_tape_unary :: proc(tape: ^Tape($T), x: Tape_Value(T), operation: Tape_Operation) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	view := tape_value(tape, x) or_return
	output: Tensor(T)
	#partial switch operation {
	case .Relu:
		output = relu_views(view, tape.allocator) or_return
	case .Sigmoid:
		output = sigmoid_views(view, tape.allocator) or_return
	case .Tanh:
		output = tanh_views(view, tape.allocator) or_return
	case .Exp:
		output = exp_views(view, tape.allocator) or_return
	case .Log:
		output = log_views(view, tape.allocator) or_return
	case .Sqrt:
		output = sqrt_views(view, tape.allocator) or_return
	case:
		return result, .Unsupported_Operation
	}
	result, err = tape_record_unary(tape, &output, operation, x)
	release(&output)
	return
}

_tape_reduce :: proc(tape: ^Tape($T), x: Tape_Value(T), operation: Tape_Operation, axis: int, keep_dims: bool) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	view := tape_value(tape, x) or_return
	output: Tensor(T)
	#partial switch operation {
	case .Sum_All:
		output = sum_all_views(view, tape.allocator) or_return
	case .Mean_All:
		output = mean_all_views(view, tape.allocator) or_return
	case .Sum_Axis:
		output = sum_axis_views(view, axis, keep_dims, tape.allocator) or_return
	case .Mean_Axis:
		output = mean_axis_views(view, axis, keep_dims, tape.allocator) or_return
	case .Prod_All:
		output = prod_all_views(view, tape.allocator) or_return
	case .Prod_Axis:
		output = prod_axis_views(view, axis, keep_dims, tape.allocator) or_return
	case .Max_All:
		output = max_all_views(view, tape.allocator) or_return
	case .Min_All:
		output = min_all_views(view, tape.allocator) or_return
	case .Max_Axis:
		output = max_axis_views(view, axis, keep_dims, tape.allocator) or_return
	case .Min_Axis:
		output = min_axis_views(view, axis, keep_dims, tape.allocator) or_return
	case:
		return result, .Unsupported_Operation
	}
	result, err = _tape_record_reduction(tape, &output, operation, x, axis, keep_dims)
	release(&output)
	return
}

// _tape_view records a view of x. storage owns the view's metadata, plus its
// data when a reshape has to copy.
_tape_view :: proc(tape: ^Tape($T), x: Tape_Value(T), operation: Tape_Operation, axis, start, count, step: int, shape: []int) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	parent_node := _tape_node(tape, x)
	if parent_node == nil {
		return result, .Invalid_Tape
	}
	parent := parent_node.value
	template := _Tape_Node(T){
		requires_grad = parent_node.requires_grad,
		operation     = operation,
		parent_a      = x.id,
		parent_b      = -1,
		parent_count  = 1,
	}
	value: Tensor_View(T)
	storage: Tensor(T)
	defer release(&storage)
	#partial switch operation {
	case .Transpose:
		storage = _new_view_storage(T, len(parent.shape), tape.allocator) or_return
		value = transpose_view(parent, storage.shape, storage.strides) or_return
	case .Slice:
		template.axis = _normalize_axis(axis, len(parent.shape)) or_return
		template.start = start
		template.step = step
		storage = _new_view_storage(T, len(parent.shape), tape.allocator) or_return
		value = slice_view(parent, template.axis, template.start, count, step, storage.shape, storage.strides) or_return
	case .Reshape:
		parent_count, _ := _element_count(parent.shape)
		shape_buffer: [MAX_RANK]int
		new_shape := _reshape_target(shape, parent_count, shape_buffer[:]) or_return
		if _is_contiguous_view(parent) {
			storage = _new_view_storage(T, len(new_shape), tape.allocator) or_return
			copy(storage.shape, new_shape)
			_fill_row_major_strides(storage.shape, storage.strides)
			value = Tensor_View(T){data = parent.data, shape = storage.shape, strides = storage.strides, offset = parent.offset}
		} else {
			storage = new_tensor(T, new_shape, tape.allocator) or_return
			_copy_into(storage.data, parent)
			value = as_view(storage)
		}
	case:
		return result, .Unsupported_Operation
	}
	return _tape_take_storage(tape, &storage, value, template)
}

// _new_view_storage allocates shape and strides of the given rank and no data.
_new_view_storage :: proc($T: typeid, rank: int, allocator := context.allocator) -> (storage: Tensor(T), err: Tensor_Error) {
	shape, alloc_err := make([]int, rank, allocator)
	if alloc_err != nil {
		return storage, .Allocation_Failed
	}
	strides: []int
	strides, alloc_err = make([]int, rank, allocator)
	if alloc_err != nil {
		delete(shape, allocator)
		return storage, .Allocation_Failed
	}
	return Tensor(T){shape = shape, strides = strides, allocator = allocator}, .None
}
