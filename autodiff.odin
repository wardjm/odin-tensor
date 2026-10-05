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

// tape_dot and tape_matmul follow dot's rank-one/rank-two rules; batched
// (rank > 2) operands return Invalid_Shape.
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

tape_sum_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Sum_All, -1, false)
}

tape_mean_all :: proc(tape: ^Tape($T), x: Tape_Value(T)) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Mean_All, -1, false)
}

// tape_sum_axis and tape_mean_axis accept a negative axis, counted from the end.
tape_sum_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Sum_Axis, axis, keep_dims)
}

tape_mean_axis :: proc(tape: ^Tape($T), x: Tape_Value(T), axis: int, keep_dims := false) -> (Tape_Value(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_reduce(tape, x, .Mean_Axis, axis, keep_dims)
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
	case:
		return result, .Unsupported_Operation
	}
	result, err = _tape_record_reduction(tape, &output, operation, x, axis, keep_dims)
	release(&output)
	return
}
