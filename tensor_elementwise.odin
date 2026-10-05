package tensor

// Broadcasted binary arithmetic.

import "base:intrinsics"

_Binary_Operation :: enum {
	Add,
	Subtract,
	Multiply,
	Divide,
}

// _apply_binary_operation is specialized per operation at compile time, so
// kernels contain no per-element branch on the operation.
_apply_binary_operation :: #force_inline proc(a, b: $T, $operation: _Binary_Operation) -> T {
	when operation == .Add {
		return a + b
	} else when operation == .Subtract {
		return a - b
	} else when operation == .Multiply {
		return a * b
	} else {
		return a / b
	}
}

// broadcast_shape writes the NumPy-style broadcasted shape of a and b into
// shape_buffer. Dimensions are aligned from the right and must either match or
// have one operand equal to one. It returns the required rank.
broadcast_shape :: proc(a, b, shape_buffer: []int) -> (rank: int, err: Tensor_Error) {
	if _, valid := _element_count(a); !valid {
		return 0, .Invalid_Shape
	}
	if _, valid := _element_count(b); !valid {
		return 0, .Invalid_Shape
	}
	rank = max(len(a), len(b))
	if len(shape_buffer) < rank {
		return 0, .Insufficient_Buffer
	}
	for axis in 0..<rank {
		a_axis := axis-(rank-len(a))
		b_axis := axis-(rank-len(b))
		a_dim := a[a_axis] if a_axis >= 0 else 1
		b_dim := b[b_axis] if b_axis >= 0 else 1
		if a_dim == b_dim || b_dim == 1 {
			shape_buffer[axis] = a_dim
		} else if a_dim == 1 {
			shape_buffer[axis] = b_dim
		} else {
			return 0, .Shape_Mismatch
		}
	}
	if !_strides_are_valid(shape_buffer[:rank]) {
		return 0, .Invalid_Shape
	}
	return rank, .None
}

// _validate_division runs before any output is written. Floats follow IEEE
// rules (x/0 is ±Inf or NaN), as in NumPy. Integers reject zero divisors and,
// for signed types, min(T) / -1, which would overflow and trap.
_validate_division :: proc(dividends, divisors: Tensor_View($T), result_shape: []int) -> Tensor_Error {
	when intrinsics.type_is_integer(T) {
		count, _ := _element_count(result_shape)
		for i in 0..<count {
			divisor := divisors.data[_broadcast_offset(divisors, result_shape, i)]
			if divisor == 0 {
				return .Division_By_Zero
			}
			when !intrinsics.type_is_unsigned(T) {
				if divisor == -1 && dividends.data[_broadcast_offset(dividends, result_shape, i)] == min(T) {
					return .Integer_Overflow
				}
			}
		}
	}
	return .None
}

// _binary_kernel writes the broadcast operation for every element of
// result_shape into the contiguous output. Both views must be validated and
// broadcast-compatible with result_shape. Each output element is written only
// after its inputs are read, so output may be the storage of an input that has
// the full result shape.
_binary_kernel :: proc(output: []$T, a, b: Tensor_View(T), result_shape: []int, $operation: _Binary_Operation) {
	count := len(output)
	if count == 0 {
		return
	}
	if _shapes_match(a.shape, result_shape) && _shapes_match(b.shape, result_shape) && _is_contiguous_view(a) && _is_contiguous_view(b) {
		a_data := a.data[a.offset:][:count]
		b_data := b.data[b.offset:][:count]
		for i in 0..<count {
			output[i] = _apply_binary_operation(a_data[i], b_data[i], operation)
		}
		return
	}
	row := _row_length(result_shape)
	a_step := _inner_step(a, result_shape)
	b_step := _inner_step(b, result_shape)
	for start := 0; start < count; start += row {
		a_offset := _broadcast_offset(a, result_shape, start)
		b_offset := _broadcast_offset(b, result_shape, start)
		for i in start..<start+row {
			output[i] = _apply_binary_operation(a.data[a_offset], b.data[b_offset], operation)
			a_offset += a_step
			b_offset += b_step
		}
	}
}

// _binary_views_into validates both views and an exactly shaped output, then
// writes the broadcast result. On any error, output is left unchanged.
_binary_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T), $operation: _Binary_Operation) -> Tensor_Error {
	_validate_view(a) or_return
	_validate_view(b) or_return
	_validate_output(output^) or_return
	shape_buffer: [MAX_RANK]int
	rank := broadcast_shape(a.shape, b.shape, shape_buffer[:]) or_return
	result_shape := shape_buffer[:rank]
	if !_shapes_match(output.shape, result_shape) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, a) && !_is_same_storage(output^, a) {
		return .Aliased_Output
	}
	if _output_overlaps_view(output^, b) && !_is_same_storage(output^, b) {
		return .Aliased_Output
	}
	when operation == .Divide {
		_validate_division(a, b, result_shape) or_return
	}
	_binary_kernel(output.data, a, b, result_shape, operation)
	return .None
}

// _is_same_storage reports whether view addresses exactly output's elements in
// the same order, the one overlap element-wise kernels handle safely.
_is_same_storage :: proc(output: Tensor($T), view: Tensor_View(T)) -> bool {
	return len(output.data) > 0 && _shapes_match(output.shape, view.shape) && _is_contiguous_view(view) &&
		&view.data[view.offset] == &output.data[0]
}

// _binary_views applies operation to two views using NumPy-style
// trailing-dimension broadcasting. The result is a new owned contiguous tensor.
_binary_views :: proc(a, b: Tensor_View($T), $operation: _Binary_Operation, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(a) or_return
	_validate_view(b) or_return
	shape_buffer: [MAX_RANK]int
	rank := broadcast_shape(a.shape, b.shape, shape_buffer[:]) or_return
	result_shape := shape_buffer[:rank]
	when operation == .Divide {
		_validate_division(a, b, result_shape) or_return
	}
	result = new_tensor(T, result_shape, allocator) or_return
	_binary_kernel(result.data, a, b, result_shape, operation)
	return result, .None
}

// _binary_in_place applies operation with b into a without allocating. b may be
// broadcast into a, but a's shape cannot be expanded.
_binary_in_place :: proc(a: ^Tensor($T), b: Tensor_View(T), $operation: _Binary_Operation) -> Tensor_Error {
	_validate_view(b) or_return
	if len(b.shape) > len(a.shape) {
		return .Shape_Mismatch
	}
	leading := len(a.shape)-len(b.shape)
	for b_dim, b_axis in b.shape {
		if b_dim != 1 && b_dim != a.shape[leading+b_axis] {
			return .Shape_Mismatch
		}
	}
	if _output_overlaps_view(a^, b) && !_is_same_storage(a^, b) {
		return .Aliased_Output
	}
	a_view := as_view(a^)
	when operation == .Divide {
		_validate_division(a_view, b, a.shape) or_return
	}
	_binary_kernel(a.data, a_view, b, a.shape, operation)
	return .None
}

// add, sub, mul, and div return a new tensor containing the element-wise
// broadcasted result. The result uses allocator (context.allocator by default).
add :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(as_view(a), as_view(b), .Add, allocator)
}

sub :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(as_view(a), as_view(b), .Subtract, allocator)
}

mul :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(as_view(a), as_view(b), .Multiply, allocator)
}

div :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(as_view(a), as_view(b), .Divide, allocator)
}

// add_views, sub_views, mul_views, and div_views broadcast two strided views
// and return a new owned, contiguous tensor. Neither input is copied.
add_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(a, b, .Add, allocator)
}

sub_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(a, b, .Subtract, allocator)
}

mul_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(a, b, .Multiply, allocator)
}

div_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _binary_views(a, b, .Divide, allocator)
}

// add_into, sub_into, mul_into, and div_into write the broadcast result into an
// existing output with exactly the broadcast shape, without allocating. output
// may be the same tensor as a or b.
add_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, as_view(a), as_view(b), .Add)
}

sub_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, as_view(a), as_view(b), .Subtract)
}

mul_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, as_view(a), as_view(b), .Multiply)
}

div_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, as_view(a), as_view(b), .Divide)
}

// add_views_into, sub_views_into, mul_views_into, and div_views_into are the
// strided-view forms of the _into procedures. An input that overlaps output
// other than as output's own storage returns Aliased_Output.
add_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, a, b, .Add)
}

sub_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, a, b, .Subtract)
}

mul_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, a, b, .Multiply)
}

div_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_views_into(output, a, b, .Divide)
}

// add_in_place, sub_in_place, mul_in_place, and div_in_place update a in place.
// b may broadcast to a's shape, but a's shape is never expanded.
add_in_place :: proc(a: ^Tensor($T), b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_in_place(a, as_view(b), .Add)
}

sub_in_place :: proc(a: ^Tensor($T), b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_in_place(a, as_view(b), .Subtract)
}

mul_in_place :: proc(a: ^Tensor($T), b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_in_place(a, as_view(b), .Multiply)
}

div_in_place :: proc(a: ^Tensor($T), b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _binary_in_place(a, as_view(b), .Divide)
}
