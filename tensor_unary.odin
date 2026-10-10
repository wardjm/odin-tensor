package tensor

// Element-wise unary math and activation operations.

import "base:intrinsics"
import "core:math"

_Unary_Operation :: enum {
	Exp,
	Log,
	Sqrt,
	ReLU,
	Sigmoid,
	Tanh,
}

_apply_unary_operation :: #force_inline proc(value: $T, $operation: _Unary_Operation) -> T where intrinsics.type_is_float(T) {
	when operation == .Exp {
		return math.exp(value)
	} else when operation == .Log {
		return math.ln(value)
	} else when operation == .Sqrt {
		return math.sqrt(value)
	} else when operation == .ReLU {
		return value if value > 0 else 0
	} else when operation == .Sigmoid {
		// Select the expression by sign to avoid overflowing exp for large inputs.
		if value >= 0 {
			return 1 / (1 + math.exp(-value))
		}
		exp_value := math.exp(value)
		return exp_value / (1 + exp_value)
	} else {
		return math.tanh(value)
	}
}

// _unary_kernel applies operation to every element of a validated source,
// writing output in row-major order.
_unary_kernel :: proc(output: []$T, source: Tensor_View(T), $operation: _Unary_Operation) {
	if len(output) == 0 {
		return
	}
	job := _Unary_Job(T){output = output, source = source, contiguous = _is_contiguous_view(source)}
	_parallel_ranges(len(output), 1 if job.contiguous else _row_length(source.shape), proc(data: rawptr, first, last: int) {
		job := (^_Unary_Job(T))(data)
		_unary_range(job, first, last, operation)
	}, &job)
}

_Unary_Job :: struct($T: typeid) {
	output:     []T,
	source:     Tensor_View(T),
	contiguous: bool,
}

// _unary_range fills output[first:last]; for strided views first is a
// multiple of the row length.
_unary_range :: proc(job: ^_Unary_Job($T), first, last: int, $operation: _Unary_Operation) {
	output, source := job.output, job.source
	if job.contiguous {
		source_data := source.data[source.offset:]
		for i in first..<last {
			output[i] = _apply_unary_operation(source_data[i], operation)
		}
		return
	}
	row := _row_length(source.shape)
	step := _inner_step(source, source.shape)
	for start := first; start < last; start += row {
		offset := _broadcast_offset(source, source.shape, start)
		for i in start..<start+row {
			output[i] = _apply_unary_operation(source.data[offset], operation)
			offset += step
		}
	}
}

// _unary_views_into writes an element-wise unary operation into an exactly
// shaped output tensor. Source may be any valid strided view. It returns
// Aliased_Output if source overlaps output other than as output's own storage.
_unary_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T), $operation: _Unary_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_output(output^) or_return
	_validate_view(source) or_return
	if !_shapes_match(output.shape, source.shape) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, source) && !_is_same_storage(output^, source) {
		return .Aliased_Output
	}
	_unary_kernel(output.data, source, operation)
	return .None
}

// _unary_views applies an element-wise unary operation to a strided view and
// returns a new contiguous tensor.
_unary_views :: proc(source: Tensor_View($T), $operation: _Unary_Operation, allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	_validate_view(source) or_return
	result = new_tensor(T, source.shape, allocator) or_return
	_unary_kernel(result.data, source, operation)
	return result, .None
}

// exp, log, sqrt, relu, sigmoid, and tanh return a new tensor. The result uses
// allocator (context.allocator by default).
exp :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .Exp, allocator)
}
log :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .Log, allocator)
}
sqrt :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .Sqrt, allocator)
}
relu :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .ReLU, allocator)
}
sigmoid :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .Sigmoid, allocator)
}
tanh :: proc(source: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(as_view(source), .Tanh, allocator)
}

// The _views forms accept any valid strided view and return a new tensor.
exp_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .Exp, allocator)
}
log_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .Log, allocator)
}
sqrt_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .Sqrt, allocator)
}
relu_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .ReLU, allocator)
}
sigmoid_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .Sigmoid, allocator)
}
tanh_views :: proc(source: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_float(T) {
	return _unary_views(source, .Tanh, allocator)
}

// The _views_into forms write into an exactly shaped output without allocating.
exp_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .Exp)
}
log_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .Log)
}
sqrt_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .Sqrt)
}
relu_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .ReLU)
}
sigmoid_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .Sigmoid)
}
tanh_views_into :: proc(output: ^Tensor($T), source: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, source, .Tanh)
}

// The _into forms take an owned source; output may be the same tensor.
exp_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .Exp)
}
log_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .Log)
}
sqrt_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .Sqrt)
}
relu_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .ReLU)
}
sigmoid_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .Sigmoid)
}
tanh_into :: proc(output: ^Tensor($T), source: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(output, as_view(source), .Tanh)
}

// The _in_place forms overwrite every element of value.
exp_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .Exp)
}
log_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .Log)
}
sqrt_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .Sqrt)
}
relu_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .ReLU)
}
sigmoid_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .Sigmoid)
}
tanh_in_place :: proc(value: ^Tensor($T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _unary_views_into(value, as_view(value^), .Tanh)
}
