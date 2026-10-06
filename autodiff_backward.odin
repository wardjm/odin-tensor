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
		case .Dot:
			err = _backward_dot(&scratch[a], &scratch[b], scratch[node_id], tape.nodes[a].value, tape.nodes[b].value)
		case .Matmul:
			err = _backward_matmul(&scratch[a], &scratch[b], scratch[node_id], tape.nodes[a].value, tape.nodes[b].value)
		case .Relu, .Sigmoid, .Tanh, .Exp, .Log, .Sqrt:
			err = _backward_activation(&scratch[a], scratch[node_id], tape.nodes[a].value, node.value, node.operation)
		case .Sum_All:
			err = _backward_sum_all(&scratch[a], scratch[node_id], tape.nodes[a].value)
		case .Mean_All:
			err = _backward_mean_all(&scratch[a], scratch[node_id], tape.nodes[a].value)
		case .Sum_Axis:
			err = _backward_sum_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims)
		case .Mean_Axis:
			err = _backward_mean_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims)
		case .Max_All:
			err = _backward_extreme_all(&scratch[a], scratch[node_id], tape.nodes[a].value, .Max)
		case .Min_All:
			err = _backward_extreme_all(&scratch[a], scratch[node_id], tape.nodes[a].value, .Min)
		case .Max_Axis:
			err = _backward_extreme_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims, .Max)
		case .Min_Axis:
			err = _backward_extreme_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims, .Min)
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

// _backward_extreme_all adds the scalar upstream gradient to the one element
// argmax_all (.Max) or argmin_all (.Min) picks, found again from the input.
_backward_extreme_all :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), $operation: _Reduction_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	if len(output_grad.shape) != 0 || len(output_grad.data) != 1 {
		return .Shape_Mismatch
	}
	if !_shapes_match(gradient.shape, input.shape) {
		return .Shape_Mismatch
	}
	_validate_view(input) or_return
	count, valid := _element_count(input.shape)
	if !valid || count != len(gradient.data) {
		return .Invalid_Shape
	}
	if count == 0 {
		return .Empty_Reduction
	}
	gradient.data[_arg_all_index(input, count, operation)] += output_grad.data[0]
	return .None
}

// _backward_extreme_axis adds each upstream value to the element along axis
// that argmax_axis (.Max) or argmin_axis (.Min) picks, found again from the
// input.
_backward_extreme_axis :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), axis: int, keep_dims: bool, $operation: _Reduction_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_view(input) or_return
	if axis < 0 || axis >= len(input.shape) {
		return .Out_Of_Bounds
	}
	if !_shapes_match(gradient.shape, input.shape) || !_reduction_shape_matches(output_grad, input.shape, axis, keep_dims) {
		return .Shape_Mismatch
	}
	reduced_count := input.shape[axis]
	if reduced_count == 0 {
		return .Empty_Reduction
	}
	inner := _reduction_inner_count(input.shape, axis)
	for output_index in 0..<len(output_grad.data) {
		offset := _reduction_base_offset(input, axis, inner, output_index)
		best := _arg_run(input.data, offset, input.strides[axis], reduced_count, operation)
		// The gradient is contiguous, so the reduced axis advances by inner.
		outer_index := output_index / inner
		inner_index := output_index - outer_index*inner
		gradient.data[(outer_index*reduced_count + best)*inner + inner_index] += output_grad.data[output_index]
	}
	return .None
}

// _backward_activation applies the elementwise VJP for relu, sigmoid, tanh,
// exp, log, or sqrt. Forward values stay on the tape: ReLU and log use their
// input and the others use their outputs, avoiding a second transcendental
// evaluation.
_backward_activation :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input, output: Tensor_View(T), operation: Tape_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	#partial switch operation {
	case .Relu, .Sigmoid, .Tanh, .Exp, .Log, .Sqrt:
	case:
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
		case .Exp:
			derivative = output.data[output_offset]
		case .Log:
			derivative = 1 / input.data[input_offset]
		case .Sqrt:
			derivative = 1 / (2*output.data[output_offset])
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
// matrix/matrix product reverse rules.
_backward_dot :: proc(grad_a, grad_b: ^Tensor($T), output_grad: Tensor(T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
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
	return _backward_matrix_product(grad_a.data, grad_b.data, output_grad.data, a, b)
}

// _backward_matmul applies the matrix product rule once per batch index. A
// batch axis an operand was broadcast along maps several batches onto one
// slice of its gradient, so their contributions sum there.
_backward_matmul :: proc(grad_a, grad_b: ^Tensor($T), output_grad: Tensor(T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_float(T) {
	if err := _validate_autodiff_view(a); err != .None {
		return err
	}
	if err := _validate_autodiff_view(b); err != .None {
		return err
	}
	shape_buffer: [MAX_RANK]int
	result_rank, shape_err := _matmul_result_shape(a.shape, b.shape, shape_buffer[:])
	if shape_err != .None {
		return shape_err
	}
	if !_shapes_match(output_grad.shape, shape_buffer[:result_rank]) {
		return .Shape_Mismatch
	}
	a_batch_rank := max(len(a.shape)-2, 0)
	b_batch_rank := max(len(b.shape)-2, 0)
	batch_shape := shape_buffer[:max(a_batch_rank, b_batch_rank)]
	batch_count, _ := _element_count(batch_shape)
	if batch_count == 0 {
		return .None
	}
	a_batch := Tensor_View(T){data = a.data, shape = a.shape[:a_batch_rank], strides = a.strides[:a_batch_rank], offset = a.offset}
	b_batch := Tensor_View(T){data = b.data, shape = b.shape[:b_batch_rank], strides = b.strides[:b_batch_rank], offset = b.offset}
	a_matrix := Tensor_View(T){data = a.data, shape = a.shape[a_batch_rank:], strides = a.strides[a_batch_rank:]}
	b_matrix := Tensor_View(T){data = b.data, shape = b.shape[b_batch_rank:], strides = b.strides[b_batch_rank:]}
	a_matrix_size, _ := _element_count(a_matrix.shape)
	b_matrix_size, _ := _element_count(b_matrix.shape)
	output_matrix_size := len(output_grad.data) / batch_count
	for batch in 0..<batch_count {
		a_matrix.offset = _broadcast_offset(a_batch, batch_shape, batch)
		b_matrix.offset = _broadcast_offset(b_batch, batch_shape, batch)
		a_grad, b_grad: []T
		if grad_a.data != nil {
			a_index, ok := _broadcast_gradient_index(a_batch.shape, batch_shape, batch)
			if !ok {
				return .Invalid_Shape
			}
			a_grad = grad_a.data[a_index*a_matrix_size:][:a_matrix_size]
		}
		if grad_b.data != nil {
			b_index, ok := _broadcast_gradient_index(b_batch.shape, batch_shape, batch)
			if !ok {
				return .Invalid_Shape
			}
			b_grad = grad_b.data[b_index*b_matrix_size:][:b_matrix_size]
		}
		upstream := output_grad.data[batch*output_matrix_size:][:output_matrix_size]
		_backward_matrix_product(a_grad, b_grad, upstream, a_matrix, b_matrix) or_return
	}
	return .None
}

// _backward_matrix_product accumulates one rank-one/rank-two product's
// gradients into contiguous grad_a and grad_b (nil when not tracked). A vector
// a is a single row and a vector b a single column, as in the forward kernel.
_backward_matrix_product :: proc(grad_a, grad_b, upstream: []$T, a, b: Tensor_View(T)) -> Tensor_Error {
	a_rank, b_rank := len(a.shape), len(b.shape)
	inner_count := a.shape[a_rank-1]
	m := a.shape[0] if a_rank == 2 else 1
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	for i in 0..<m {
		for j in 0..<n {
			upstream_index := i*n + j
			if upstream_index >= len(upstream) {
				return .Invalid_Shape
			}
			gradient := upstream[upstream_index]
			for inner in 0..<inner_count {
				a_offset := a.offset + i*a_row_stride + inner*a.strides[a_rank-1]
				b_offset := b.offset + inner*b.strides[0] + j*b_column_stride
				if a_offset < 0 || a_offset >= len(a.data) || b_offset < 0 || b_offset >= len(b.data) {
					return .Out_Of_Bounds
				}
				if grad_a != nil {
					a_grad_index := i*inner_count + inner
					if a_grad_index >= len(grad_a) {
						return .Invalid_Shape
					}
					grad_a[a_grad_index] += gradient*b.data[b_offset]
				}
				if grad_b != nil {
					b_grad_index := inner*n + j
					if b_grad_index >= len(grad_b) {
						return .Invalid_Shape
					}
					grad_b[b_grad_index] += gradient*a.data[a_offset]
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

