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
