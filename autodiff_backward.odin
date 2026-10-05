package tensor

// Reverse traversal and gradient kernels.

import "base:intrinsics"

// tape_backward_with_seed accumulates a same-shaped seed at a graph value and
// propagates its contribution in reverse topological order. Repeated calls
// accumulate gradients without replaying earlier seeds.
tape_backward_with_seed :: proc(tape: ^Tape($T), value: Tape_Value(T), seed: Tensor(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	node := _tape_node(tape, value)
	if node == nil {
		return .Invalid_Tape
	}
	if !node.requires_grad {
		return .Gradient_Not_Tracked
	}
	if !_shapes_match(node.value.shape, seed.shape) {
		return .Shape_Mismatch
	}
	return _tape_backward_seeded(tape, value.id, seed.data)
}

// tape_backward seeds a rank-zero value with one. Repeated calls accumulate;
// call tape_zero_grad to reset gradients. Non-scalar values require an explicit
// seed tensor via tape_backward_with_seed.
tape_backward :: proc(tape: ^Tape($T), value: Tape_Value(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	node := _tape_node(tape, value)
	if node == nil {
		return .Invalid_Tape
	}
	if len(node.value.shape) != 0 {
		return .Shape_Mismatch
	}
	if !node.requires_grad {
		return .Gradient_Not_Tracked
	}
	seed := [1]T{1}
	return _tape_backward_seeded(tape, value.id, seed[:])
}

// _tape_backward_seeded performs a transactional reverse pass in temporary
// gradient buffers, committing only after every active node is supported.
// Only nodes reachable from root are visited, so unrelated branches cannot
// contribute (even NaN from 0*inf), and nodes recorded after root are ignored.
_tape_backward_seeded :: proc(tape: ^Tape($T), root: int, seed: []T) -> Tensor_Error where intrinsics.type_is_float(T) {
	node_count := root+1
	scratch, scratch_alloc_err := make([]Tensor(T), node_count, tape.allocator)
	if scratch_alloc_err != nil {
		return .Allocation_Failed
	}
	defer {
		for i in 0..<node_count {
			release(&scratch[i])
		}
		delete(scratch, tape.allocator)
	}
	reached, reached_alloc_err := make([]bool, node_count, tape.allocator)
	if reached_alloc_err != nil {
		return .Allocation_Failed
	}
	defer delete(reached, tape.allocator)

	scratch[root] = new_tensor(T, tape.nodes[root].value.shape, tape.allocator) or_return
	reached[root] = true
	if len(seed) != len(scratch[root].data) {
		return .Shape_Mismatch
	}
	copy(scratch[root].data, seed)

	for node_id := root; node_id >= 0; node_id -= 1 {
		node := &tape.nodes[node_id]
		if !reached[node_id] || !node.requires_grad || node.operation == .Leaf {
			continue
		}
		parents := [2]int{node.parent_a, node.parent_b}
		for parent in parents[:node.parent_count] {
			if parent < 0 || parent >= node_id {
				return .Invalid_Tape
			}
			if tape.nodes[parent].requires_grad && !reached[parent] {
				scratch[parent] = new_tensor(T, tape.nodes[parent].value.shape, tape.allocator) or_return
				reached[parent] = true
			}
		}
		a, b := node.parent_a, node.parent_b
		err: Tensor_Error
		switch node.operation {
		case .Add, .Sub, .Mul, .Div:
			err = _backward_binary_elementwise(&scratch[a], &scratch[b], scratch[node_id], tape.nodes[a].value, tape.nodes[b].value, node.operation)
		case .Dot, .Matmul:
			err = _backward_dot(&scratch[a], &scratch[b], scratch[node_id], tape.nodes[a].value, tape.nodes[b].value, node.operation)
		case .Relu, .Sigmoid, .Tanh:
			err = _backward_activation(&scratch[a], scratch[node_id], tape.nodes[a].value, node.value, node.operation)
		case .Sum_All:
			err = _backward_sum_all(&scratch[a], scratch[node_id], tape.nodes[a].value)
		case .Mean_All:
			err = _backward_mean_all(&scratch[a], scratch[node_id], tape.nodes[a].value)
		case .Sum_Axis:
			err = _backward_sum_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims)
		case .Mean_Axis:
			err = _backward_mean_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims)
		case .Leaf:
		}
		if err != .None {
			return err
		}
	}

	for i in 0..<node_count {
		if reached[i] && tape.nodes[i].requires_grad {
			for j in 0..<len(tape.nodes[i].grad.data) {
				tape.nodes[i].grad.data[j] += scratch[i].data[j]
			}
		}
	}
	return .None
}

// _backward_sum_all broadcasts the scalar upstream gradient over the input's
// logical elements. The destination gradient is contiguous in logical order,
// independent of the input view's offset or strides.
_backward_sum_all :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	if len(output_grad.shape) != 0 || len(output_grad.data) != 1 {
		return .Shape_Mismatch
	}
	if !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	count, valid := _element_count(input.shape)
	if !valid || count != len(gradient.data) {
		return .Invalid_Shape
	}
	for linear in 0..<count {
		gradient.data[linear] += output_grad.data[0]
	}
	return .None
}

// _backward_sum_axis expands upstream values across the reduced dimension.
_backward_sum_axis :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), axis: int, keep_dims: bool) -> Tensor_Error where intrinsics.type_is_float(T) {
	return _backward_axis_reduction(gradient, output_grad, input, axis, keep_dims, 1)
}

// _backward_mean_axis expands upstream values like sum_axis, scaled by the
// length of the reduced dimension.
_backward_mean_axis :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), axis: int, keep_dims: bool) -> Tensor_Error where intrinsics.type_is_float(T) {
	if err := _validate_view(input); err != .None {
		return err
	}
	if axis < 0 || axis >= len(input.shape) {
		return .Out_Of_Bounds
	}
	if input.shape[axis] == 0 {
		return .Empty_Reduction
	}
	return _backward_axis_reduction(gradient, output_grad, input, axis, keep_dims, input.shape[axis])
}

_backward_axis_reduction :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), axis: int, keep_dims: bool, divisor: int) -> Tensor_Error where intrinsics.type_is_float(T) {
	if err := _validate_view(input); err != .None {
		return err
	}
	rank := len(input.shape)
	if axis < 0 || axis >= rank {
		return .Out_Of_Bounds
	}
	if !_shapes_match(gradient.shape, input.shape) || !_reduction_shape_matches(output_grad, input.shape, axis, keep_dims) {
		return .Shape_Mismatch
	}
	input_count, valid := _element_count(input.shape)
	if !valid || input_count != len(gradient.data) {
		return .Invalid_Shape
	}

	for input_linear in 0..<input_count {
		remaining := input_linear
		output_linear := 0
		for source_axis := rank-1; source_axis >= 0; source_axis -= 1 {
			dimension := input.shape[source_axis]
			coordinate := remaining % dimension
			remaining /= dimension
			if source_axis == axis {
				continue
			}
			output_axis := source_axis
			if !keep_dims && source_axis > axis {
				output_axis -= 1
			}
			term, term_ok := _checked_mul(coordinate, output_grad.strides[output_axis])
			if !term_ok {
				return .Invalid_Shape
			}
			output_linear, valid = _checked_add(output_linear, term)
			if !valid {
				return .Invalid_Shape
			}
		}
		if output_linear < 0 || output_linear >= len(output_grad.data) {
			return .Invalid_Shape
		}
		gradient.data[input_linear] += output_grad.data[output_linear] / T(divisor)
	}
	return .None
}

// _backward_mean_all broadcasts the scalar upstream gradient over the input's
// logical elements, scaled by the reciprocal element count.
_backward_mean_all :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	if len(output_grad.shape) != 0 || len(output_grad.data) != 1 {
		return .Shape_Mismatch
	}
	if !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	count, valid := _element_count(input.shape)
	if !valid || count != len(gradient.data) {
		return .Invalid_Shape
	}
	if count == 0 {
		return .Empty_Reduction
	}
	contribution := output_grad.data[0] / T(count)
	for linear in 0..<count {
		gradient.data[linear] += contribution
	}
	return .None
}

// _backward_activation applies the elementwise VJP for relu, sigmoid, or tanh.
// Forward values stay on the tape: ReLU uses its input and sigmoid/tanh use
// their outputs, avoiding a second transcendental evaluation.
_backward_activation :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input, output: Tensor_View(T), operation: Tape_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	if operation != .Relu && operation != .Sigmoid && operation != .Tanh {
		return .Unsupported_Operation
	}
	if !_shapes_match(input.shape, output.shape) || !_shapes_match(output.shape, output_grad.shape) {
		return .Shape_Mismatch
	}
	_validate_view(input) or_return
	_validate_view(output) or_return
	if len(gradient.data) != len(output_grad.data) {
		return .Shape_Mismatch
	}
	for linear in 0..<len(output_grad.data) {
		input_offset := _broadcast_offset(input, input.shape, linear)
		output_offset := _broadcast_offset(output, output.shape, linear)
		derivative: T = 0
		#partial switch operation {
		case .Relu:
			if input.data[input_offset] > 0 {
				derivative = 1
			}
		case .Sigmoid:
			value := output.data[output_offset]
			derivative = value * (1-value)
		case .Tanh:
			value := output.data[output_offset]
			derivative = 1 - value*value
		}
		gradient.data[linear] += output_grad.data[linear] * derivative
	}
	return .None
}

// _backward_binary_elementwise applies one binary VJP and reduces broadcasted
// contributions into each parent's original shape.
_backward_binary_elementwise :: proc(grad_a, grad_b: ^Tensor($T), output_grad: Tensor(T), a, b: Tensor_View(T), operation: Tape_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_view(a) or_return
	_validate_view(b) or_return
	if !_broadcast_shapes_match(a.shape, b.shape, output_grad.shape) {
		return .Shape_Mismatch
	}
	for linear in 0..<len(output_grad.data) {
		a_offset := _broadcast_offset(a, output_grad.shape, linear)
		b_offset := _broadcast_offset(b, output_grad.shape, linear)
		a_grad_index, a_grad_ok := _broadcast_gradient_index(a.shape, output_grad.shape, linear)
		b_grad_index, b_grad_ok := _broadcast_gradient_index(b.shape, output_grad.shape, linear)
		if !a_grad_ok || !b_grad_ok {
			return .Invalid_Shape
		}
		upstream := output_grad.data[linear]
		a_value := a.data[a_offset]
		b_value := b.data[b_offset]
		contribution_a, contribution_b := T(0), T(0)
		#partial switch operation {
		case .Add:
			contribution_a, contribution_b = upstream, upstream
		case .Sub:
			contribution_a, contribution_b = upstream, -upstream
		case .Mul:
			contribution_a, contribution_b = upstream*b_value, upstream*a_value
		case .Div:
			contribution_a = upstream / b_value
			contribution_b = -(upstream * a_value) / (b_value * b_value)
		case:
			return .Unsupported_Operation
		}
		if grad_a.data != nil {
			grad_a.data[a_grad_index] += contribution_a
		}
		if grad_b.data != nil {
			grad_b.data[b_grad_index] += contribution_b
		}
	}
	return .None
}

// _backward_dot applies vector/vector, matrix/vector, vector/matrix, or
// matrix/matrix product reverse rules, shared by dot and matmul.
_backward_dot :: proc(grad_a, grad_b: ^Tensor($T), output_grad: Tensor(T), a, b: Tensor_View(T), operation: Tape_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	if operation != .Dot && operation != .Matmul {
		return .Unsupported_Operation
	}
	if err := _validate_dot_view(a); err != .None {
		return err
	}
	if err := _validate_dot_view(b); err != .None {
		return err
	}
	shape_buffer := [2]int{}
	result_rank, shape_err := _dot_result_shape(a.shape, b.shape, shape_buffer[:])
	if shape_err != .None {
		return shape_err
	}
	if !_shapes_match(output_grad.shape, shape_buffer[:result_rank]) {
		return .Shape_Mismatch
	}

	a_rank, b_rank := len(a.shape), len(b.shape)
	inner_count := a.shape[a_rank-1]
	m, n := 1, 1
	if a_rank == 2 {
		m = a.shape[0]
	}
	if b_rank == 2 {
		n = b.shape[1]
	}
	for i in 0..<m {
		for j in 0..<n {
			if output_grad.data == nil {
				return .Invalid_Shape
			}
			upstream_index := 0
			if a_rank == 2 && b_rank == 2 {
				upstream_index = i*n+j
			} else if a_rank == 2 {
				upstream_index = i
			} else if b_rank == 2 {
				upstream_index = j
			}
			if upstream_index < 0 || upstream_index >= len(output_grad.data) {
				return .Invalid_Shape
			}
			upstream := output_grad.data[upstream_index]
			for inner in 0..<inner_count {
				a_offset := a.offset + inner*a.strides[a_rank-1]
				if a_rank == 2 {
					a_offset += i*a.strides[0]
				}
				b_offset := b.offset + inner*b.strides[0]
				if b_rank == 2 {
					b_offset += j*b.strides[1]
				}
				a_grad_index, b_grad_index := inner, inner
				if a_rank == 2 {
					a_grad_index = i*inner_count + inner
				}
				if b_rank == 2 {
					b_grad_index = inner*n + j
				}
				if a_offset < 0 || a_offset >= len(a.data) || b_offset < 0 || b_offset >= len(b.data) {
					return .Out_Of_Bounds
				}
				if grad_a.data != nil {
					if a_grad_index < 0 || a_grad_index >= len(grad_a.data) {
						return .Invalid_Shape
					}
					grad_a.data[a_grad_index] += upstream*b.data[b_offset]
				}
				if grad_b.data != nil {
					if b_grad_index < 0 || b_grad_index >= len(grad_b.data) {
						return .Invalid_Shape
					}
					grad_b.data[b_grad_index] += upstream*a.data[a_offset]
				}
			}
		}
	}
	return .None
}

_broadcast_shapes_match :: proc(a, b, result: []int) -> bool {
	if len(a) > len(result) || len(b) > len(result) {
		return false
	}
	for axis in 0..<len(result) {
		a_axis := axis - (len(result)-len(a))
		b_axis := axis - (len(result)-len(b))
		a_dim, b_dim := 1, 1
		if a_axis >= 0 {
			a_dim = a[a_axis]
		}
		if b_axis >= 0 {
			b_dim = b[b_axis]
		}
		if a_dim < 0 || b_dim < 0 || (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
			return false
		}
		output_dim := a_dim
		if a_dim == 1 {
			output_dim = b_dim
		}
		if result[axis] != output_dim {
			return false
		}
	}
	return true
}

_broadcast_gradient_index :: proc(source_shape, result_shape: []int, linear: int) -> (index: int, ok: bool) {
	count, valid := _element_count(result_shape)
	if !valid || linear < 0 || linear >= count || len(source_shape) > len(result_shape) {
		return 0, false
	}
	index = 0
	stride := 1
	for axis := len(source_shape)-1; axis >= 0; axis -= 1 {
		source_dim := source_shape[axis]
		result_axis := axis + len(result_shape)-len(source_shape)
		if source_dim < 0 || (source_dim != result_shape[result_axis] && source_dim != 1) {
			return 0, false
		}
		coordinate := (linear / stride) % result_shape[result_axis]
		if source_dim != 1 {
			index += coordinate * _source_contiguous_stride(source_shape, axis)
		}
		stride *= result_shape[result_axis]
	}
	return index, true
}

_source_contiguous_stride :: proc(shape: []int, axis: int) -> int {
	stride := 1
	for i := axis+1; i < len(shape); i += 1 {
		stride *= shape[i]
	}
	return stride
}

