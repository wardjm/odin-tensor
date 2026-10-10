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

// _tape_backward_seeded runs the reverse pass in temporary gradient buffers and
// commits only if every active node is supported. It visits only nodes
// reachable from root, so unrelated branches contribute nothing, not even NaN
// from 0*inf, and nodes recorded after root are ignored.
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
		case .Prod_All:
			err = _backward_prod_all(&scratch[a], scratch[node_id], tape.nodes[a].value, tape.allocator)
		case .Prod_Axis:
			err = _backward_prod_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims, tape.allocator)
		case .Max_All:
			err = _backward_extreme_all(&scratch[a], scratch[node_id], tape.nodes[a].value, .Max)
		case .Min_All:
			err = _backward_extreme_all(&scratch[a], scratch[node_id], tape.nodes[a].value, .Min)
		case .Max_Axis:
			err = _backward_extreme_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims, .Max)
		case .Min_Axis:
			err = _backward_extreme_axis(&scratch[a], scratch[node_id], tape.nodes[a].value, node.axis, node.keep_dims, .Min)
		case .Transpose, .Slice, .Reshape:
			err = _backward_view(&scratch[a], scratch[node_id], node.operation, node.axis, node.start, node.step)
		case .Im2col:
			err = _backward_im2col(&scratch[a], scratch[node_id], tape.nodes[a].value, node.kernel_height, node.kernel_width, node.step)
		case .Max_Pool2D:
			err = _backward_max_pool2d(&scratch[a], scratch[node_id], tape.nodes[a].value, node.kernel_height, node.kernel_width)
		case .Leaf:
		}
		if err != .None {
			return err
		}
	}

	// A node's first gradient takes over its scratch buffer; later backward
	// passes add into it.
	for i in 0..<node_count {
		node := &tape.nodes[i]
		if !reached[i] || !node.requires_grad {
			continue
		}
		if !node.has_grad {
			node.grad, scratch[i] = scratch[i], {}
			node.has_grad = true
			continue
		}
		_add_into(node.grad.data, scratch[i].data)
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

// _backward_prod_all adds the scalar upstream gradient times the product of
// every other element. A strided input is first copied into logical order.
_backward_prod_all :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), allocator := context.allocator) -> Tensor_Error where intrinsics.type_is_float(T) {
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
		return .None
	}
	contiguous := _is_contiguous_view(input)
	stack: [_PROD_STACK_COUNT]T
	scratch, owned := _prod_scratch(stack[:], count if contiguous else 2*count, allocator) or_return
	defer if owned {
		delete(scratch, allocator)
	}
	data, offset := input.data, input.offset
	if !contiguous {
		data, offset = scratch[count:], 0
		_copy_into(data, input)
	}
	_backward_prod_run(gradient.data, 0, 1, data, offset, 1, count, output_grad.data[0], scratch[:count])
	return .None
}

// _backward_prod_axis adds each upstream value times, for every element along
// axis, the product of the other elements in its run.
_backward_prod_axis :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), input: Tensor_View(T), axis: int, keep_dims: bool, allocator := context.allocator) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_view(input) or_return
	if axis < 0 || axis >= len(input.shape) {
		return .Out_Of_Bounds
	}
	if !_shapes_match(gradient.shape, input.shape) || !_reduction_shape_matches(output_grad, input.shape, axis, keep_dims) {
		return .Shape_Mismatch
	}
	input_count, valid := _element_count(input.shape)
	if !valid || input_count != len(gradient.data) {
		return .Invalid_Shape
	}
	if input_count == 0 {
		return .None
	}
	reduced_count := input.shape[axis]
	stack: [_PROD_STACK_COUNT]T
	prefix, owned := _prod_scratch(stack[:], reduced_count, allocator) or_return
	defer if owned {
		delete(prefix, allocator)
	}
	inner := _reduction_inner_count(input.shape, axis)
	for output_index in 0..<len(output_grad.data) {
		offset := _reduction_base_offset(input, axis, inner, output_index)
		// The gradient is contiguous, so the reduced axis advances by inner.
		outer_index := output_index / inner
		inner_index := output_index - outer_index*inner
		_backward_prod_run(gradient.data, outer_index*reduced_count*inner + inner_index, inner, input.data, offset, input.strides[axis], reduced_count, output_grad.data[output_index], prefix)
	}
	return .None
}

// Product backward passes keep up to _PROD_STACK_COUNT scratch elements on the
// stack, so short products need no allocation.
_PROD_STACK_COUNT :: 64

// _prod_scratch returns the first count elements of stack, or allocates count
// elements when stack is too short. owned reports whether to delete them.
_prod_scratch :: proc(stack: []$T, count: int, allocator := context.allocator) -> (scratch: []T, owned: bool, err: Tensor_Error) {
	if count <= len(stack) {
		return stack[:count], false, .None
	}
	buffer, alloc_err := make([]T, count, allocator)
	if alloc_err != nil {
		return nil, false, .Allocation_Failed
	}
	return buffer, true, .None
}

// _backward_prod_run adds upstream times the product of the other count-1
// elements to each element's gradient. A forward pass stores the product of
// the elements before each one in prefix, and a backward pass multiplies in
// the product of those after it, so no division is needed.
_backward_prod_run :: proc(gradient: []$T, gradient_start, gradient_step: int, data: []T, offset, step, count: int, upstream: T, prefix: []T) {
	running: T = 1
	for i in 0..<count {
		prefix[i] = running
		running *= data[offset + i*step]
	}
	running = 1
	for i := count-1; i >= 0; i -= 1 {
		gradient[gradient_start + i*gradient_step] += upstream * (prefix[i]*running)
		running *= data[offset + i*step]
	}
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
	input_count, valid := _element_count(input.shape)
	if !valid || input_count != len(gradient.data) {
		return .Invalid_Shape
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

// _backward_view applies the forward view to the input's contiguous gradient
// and adds the upstream gradient through it, so each upstream value lands on
// the input element it was read from.
_backward_view :: proc(gradient: ^Tensor($T), output_grad: Tensor(T), operation: Tape_Operation, axis, start, step: int) -> Tensor_Error where intrinsics.type_is_float(T) {
	if len(output_grad.data) == 0 {
		return .None
	}
	shape_buffer, strides_buffer: [MAX_RANK]int
	target := as_view(gradient^)
	#partial switch operation {
	case .Transpose:
		target = transpose_view(target, shape_buffer[:], strides_buffer[:]) or_return
	case .Slice:
		if axis < 0 || axis >= len(output_grad.shape) {
			return .Out_Of_Bounds
		}
		target = slice_view(target, axis, start, output_grad.shape[axis], step, shape_buffer[:], strides_buffer[:]) or_return
	case .Reshape:
		// Both gradients are contiguous in the same row-major order.
		if len(gradient.data) != len(output_grad.data) {
			return .Shape_Mismatch
		}
		_add_into(gradient.data, output_grad.data)
		return .None
	case:
		return .Unsupported_Operation
	}
	if !_shapes_match(target.shape, output_grad.shape) {
		return .Shape_Mismatch
	}
	_validate_view(target) or_return
	row := _row_length(target.shape)
	inner_step := _inner_step(target, target.shape)
	for row_start := 0; row_start < len(output_grad.data); row_start += row {
		offset := _broadcast_offset(target, target.shape, row_start)
		for i in row_start..<row_start+row {
			target.data[offset] += output_grad.data[i]
			offset += inner_step
		}
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
	#partial switch operation {
	case .Relu:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Relu)
	case .Sigmoid:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Sigmoid)
	case .Tanh:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Tanh)
	case .Exp:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Exp)
	case .Log:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Log)
	case .Sqrt:
		_backward_activation_kernel(gradient.data, output_grad.data, input, output, .Sqrt)
	}
	return .None
}

// _backward_activation_kernel is specialized per operation at compile time.
// Contiguous views take one flat loop; otherwise it finds storage offsets once
// per row, stepping along it. Both views must be validated and share the shape
// of the contiguous gradient and upstream.
_backward_activation_kernel :: proc(gradient, upstream: []$T, input, output: Tensor_View(T), $operation: Tape_Operation) {
	job := _Activation_Job(T){gradient = gradient, upstream = upstream, input = input, output = output}
	job.contiguous = _is_contiguous_view(input) && _is_contiguous_view(output)
	_parallel_ranges(len(upstream), 1 if job.contiguous else _row_length(input.shape), proc(data: rawptr, first, last: int) {
		_backward_activation_range((^_Activation_Job(T))(data), first, last, operation)
	}, &job)
}

_Activation_Job :: struct($T: typeid) {
	gradient, upstream: []T,
	input, output:      Tensor_View(T),
	contiguous:         bool,
}

// _backward_activation_range adds the gradient for elements first..<last;
// for strided views first is a multiple of the row length. Contiguous views
// take one flat loop the compiler can vectorize.
_backward_activation_range :: proc(job: ^_Activation_Job($T), first, last: int, $operation: Tape_Operation) #no_bounds_check {
	gradient, upstream, input, output := job.gradient, job.upstream, job.input, job.output
	if job.contiguous {
		inputs := input.data[input.offset:]
		outputs := output.data[output.offset:]
		for i in first..<last {
			gradient[i] += upstream[i] * _activation_derivative(inputs[i], outputs[i], operation)
		}
		return
	}
	row := _row_length(input.shape)
	input_step := _inner_step(input, input.shape)
	output_step := _inner_step(output, output.shape)
	for start := first; start < last; start += row {
		input_offset := _broadcast_offset(input, input.shape, start)
		output_offset := _broadcast_offset(output, output.shape, start)
		for i in start..<start+row {
			gradient[i] += upstream[i] * _activation_derivative(input.data[input_offset], output.data[output_offset], operation)
			input_offset += input_step
			output_offset += output_step
		}
	}
}

// _backward_binary_elementwise applies one binary VJP and reduces broadcasted
// contributions into each parent's original shape.
// _add_into adds source into target element by element, split between
// threads when large.
_add_into :: proc(target, source: []$T) {
	Job :: struct {
		target, source: []T,
	}
	job := Job{target, source}
	_parallel_ranges(len(target), 1, proc(data: rawptr, first, last: int) {
		job := (^Job)(data)
		target, source := job.target[first:last], job.source[first:last]
		for &value, i in target {
			value += source[i]
		}
	}, &job)
}

// _activation_derivative is d output / d input for one element, from whichever
// of the forward input and output the rule needs.
_activation_derivative :: #force_inline proc(input, output: $T, $operation: Tape_Operation) -> T {
	when operation == .Relu {
		return 1 if input > 0 else 0
	} else when operation == .Sigmoid {
		return output * (1-output)
	} else when operation == .Tanh {
		return 1 - output*output
	} else when operation == .Exp {
		return output
	} else when operation == .Log {
		return 1 / input
	} else when operation == .Sqrt {
		return 1 / (2*output)
	}
}

_backward_binary_elementwise :: proc(grad_a, grad_b: ^Tensor($T), output_grad: Tensor(T), a, b: Tensor_View(T), operation: Tape_Operation) -> Tensor_Error where intrinsics.type_is_float(T) {
	_validate_view(a) or_return
	_validate_view(b) or_return
	if !_broadcast_shapes_match(a.shape, b.shape, output_grad.shape) {
		return .Shape_Mismatch
	}
	#partial switch operation {
	case .Add:
		return _backward_binary_kernel(grad_a.data, grad_b.data, output_grad, a, b, .Add)
	case .Sub:
		return _backward_binary_kernel(grad_a.data, grad_b.data, output_grad, a, b, .Sub)
	case .Mul:
		return _backward_binary_kernel(grad_a.data, grad_b.data, output_grad, a, b, .Mul)
	case .Div:
		return _backward_binary_kernel(grad_a.data, grad_b.data, output_grad, a, b, .Div)
	}
	return .Unsupported_Operation
}

// _binary_vjp returns the contributions of one upstream value to each operand,
// specialized per operation at compile time.
_binary_vjp :: #force_inline proc(upstream, a, b: $T, $operation: Tape_Operation) -> (contribution_a, contribution_b: T) {
	when operation == .Add {
		return upstream, upstream
	} else when operation == .Sub {
		return upstream, -upstream
	} else when operation == .Mul {
		return upstream*b, upstream*a
	} else {
		return upstream / b, -(upstream * a) / (b * b)
	}
}

// _backward_binary_kernel is specialized per operation at compile time. It
// finds storage offsets and gradient indices once per row of the output and
// steps along it; a parent broadcast along the last axis steps by zero, so its
// contributions sum in place. grad_a or grad_b is nil when not tracked.
_backward_binary_kernel :: proc(grad_a, grad_b: []$T, output_grad: Tensor(T), a, b: Tensor_View(T), $operation: Tape_Operation) -> Tensor_Error {
	if a_period, b_period, ok := _repeating_operands(a, b, output_grad.shape); ok {
		_backward_binary_repeating(grad_a, grad_b, output_grad.data, a, b, a_period, b_period, operation)
		return .None
	}
	shape := output_grad.shape
	row := _row_length(shape)
	a_step := _inner_step(a, shape)
	b_step := _inner_step(b, shape)
	a_grad_step := _gradient_inner_step(a.shape)
	b_grad_step := _gradient_inner_step(b.shape)
	for start := 0; start < len(output_grad.data); start += row {
		a_offset := _broadcast_offset(a, shape, start)
		b_offset := _broadcast_offset(b, shape, start)
		a_grad_index, a_grad_ok := _broadcast_gradient_index(a.shape, shape, start)
		b_grad_index, b_grad_ok := _broadcast_gradient_index(b.shape, shape, start)
		if !a_grad_ok || !b_grad_ok {
			return .Invalid_Shape
		}
		for i in start..<start+row {
			contribution_a, contribution_b := _binary_vjp(output_grad.data[i], a.data[a_offset], b.data[b_offset], operation)
			if grad_a != nil {
				grad_a[a_grad_index] += contribution_a
			}
			if grad_b != nil {
				grad_b[b_grad_index] += contribution_b
			}
			a_offset += a_step
			b_offset += b_step
			a_grad_index += a_grad_step
			b_grad_index += b_grad_step
		}
	}
	return .None
}

// _gradient_inner_step is the step between consecutive output elements along
// the last axis in a parent's contiguous gradient, or zero when the parent is
// broadcast along that axis.
// _repeating_operands reports whether a and b are contiguous and each either
// fills result_shape or broadcasts only along leading axes, like a bias added
// to a batch, so in row-major order it repeats every period elements. At least
// one must fill the result.
_repeating_operands :: proc(a, b: Tensor_View($T), result_shape: []int) -> (a_period, b_period: int, ok: bool) {
	if !_is_contiguous_view(a) || !_is_contiguous_view(b) {
		return 0, 0, false
	}
	total, valid := _element_count(result_shape)
	if !valid {
		return 0, 0, false
	}
	a_period = _suffix_period(a.shape, result_shape) or_return
	b_period = _suffix_period(b.shape, result_shape) or_return
	return a_period, b_period, (a_period == total || b_period == total) && a_period > 0 && b_period > 0
}

// _suffix_period is the element count of shape when, without its leading
// ones, it equals the trailing axes of result_shape.
_suffix_period :: proc(shape, result_shape: []int) -> (period: int, ok: bool) {
	trimmed := shape
	for len(trimmed) > 0 && trimmed[0] == 1 {
		trimmed = trimmed[1:]
	}
	if len(trimmed) > len(result_shape) || !_shapes_match(trimmed, result_shape[len(result_shape)-len(trimmed):]) {
		return 0, false
	}
	period = 1
	for dim in trimmed {
		period *= dim
	}
	return period, true
}

// _backward_binary_repeating is _backward_binary_kernel for operands found by
// _repeating_operands. Gradients of operands that fill the output are split
// between threads by blocks of the shorter period; the repeating operand's
// gradient sums over every block, so threads split its elements instead,
// each adding its blocks in order. Every loop is flat and can vectorize.
_backward_binary_repeating :: proc(grad_a, grad_b, upstream: []$T, a, b: Tensor_View(T), a_period, b_period: int, $operation: Tape_Operation) {
	total := len(upstream)
	job := _Repeating_Job(T){grad_a = grad_a, grad_b = grad_b, upstream = upstream, a = a.data[a.offset:][:a_period], b = b.data[b.offset:][:b_period], period = min(a_period, b_period), total = total}
	job.a_full, job.b_full = a_period == total, b_period == total
	// Blocks are one period long, or a fixed size when both operands fill the
	// output and the period is the whole tensor.
	job.block = job.period
	if job.a_full && job.b_full {
		job.block = min(total, 1 << 12)
	}
	if (grad_a != nil && job.a_full) || (grad_b != nil && job.b_full) {
		_parallel_ranges((total + job.block - 1)/job.block, 1, proc(data: rawptr, first, last: int) {
			_repeating_full_blocks((^_Repeating_Job(T))(data), first, last, operation)
		}, &job, job.block)
	}
	if (grad_a != nil && !job.a_full) || (grad_b != nil && !job.b_full) {
		// Each thread reads every block, so only a period long enough for
		// whole cache lines per thread is worth splitting.
		_parallel_ranges(job.period, 64, proc(data: rawptr, first, last: int) {
			_repeating_sums((^_Repeating_Job(T))(data), first, last, operation)
		}, &job, total/job.period)
	}
}

_Repeating_Job :: struct($T: typeid) {
	grad_a, grad_b, upstream: []T,
	a, b:                     []T,
	period, total:            int,
	block:                    int,
	a_full, b_full:           bool,
}

// _repeating_full_blocks adds the full operands' gradients for blocks
// first..<last of job.block elements; the last block may be shorter.
_repeating_full_blocks :: proc(job: ^_Repeating_Job($T), first, last: int, $operation: Tape_Operation) #no_bounds_check {
	for start := first*job.block; start < min(last*job.block, job.total); start += job.block {
		length := min(job.block, job.total-start)
		g := job.upstream[start:][:length]
		// A repeating operand's block is its whole period.
		a_block := job.a[start:][:length] if job.a_full else job.a
		b_block := job.b[start:][:length] if job.b_full else job.b
		if job.grad_a != nil && job.a_full {
			a_grad := job.grad_a[start:][:length]
			for j in 0..<length {
				contribution, _ := _binary_vjp(g[j], a_block[j], b_block[j], operation)
				a_grad[j] += contribution
			}
		}
		if job.grad_b != nil && job.b_full {
			b_grad := job.grad_b[start:][:length]
			for j in 0..<length {
				_, contribution := _binary_vjp(g[j], a_block[j], b_block[j], operation)
				b_grad[j] += contribution
			}
		}
	}
}

// _repeating_sums adds the repeating operand's gradient for its elements
// first..<last, block by block in order.
_repeating_sums :: proc(job: ^_Repeating_Job($T), first, last: int, $operation: Tape_Operation) #no_bounds_check {
	period := job.period
	for block := 0; block < job.total; block += period {
		g := job.upstream[block:][:period]
		a_block := job.a[block:][:period] if job.a_full else job.a
		b_block := job.b[block:][:period] if job.b_full else job.b
		if job.grad_a != nil && !job.a_full {
			for j in first..<last {
				contribution, _ := _binary_vjp(g[j], a_block[j], b_block[j], operation)
				job.grad_a[j] += contribution
			}
		}
		if job.grad_b != nil && !job.b_full {
			for j in first..<last {
				_, contribution := _binary_vjp(g[j], a_block[j], b_block[j], operation)
				job.grad_b[j] += contribution
			}
		}
	}
}

_gradient_inner_step :: proc(shape: []int) -> int {
	if len(shape) == 0 || shape[len(shape)-1] == 1 {
		return 0
	}
	return 1
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
// gradients into contiguous grad_a and grad_b (nil when not tracked), treating
// a vector a as a row and a vector b as a column. With a as [m,k] and b as
// [k,n], grad_a += upstream @ bᵀ and grad_b += aᵀ @ upstream, both through the
// forward kernel on transposed views. Each element adds its terms in naive-loop
// order and matches that loop exactly, except for a @ a: grad_a and grad_b then
// share a buffer and grad_a's terms land first, so the last bits can differ.
_backward_matrix_product :: proc(grad_a, grad_b, upstream: []$T, a, b: Tensor_View(T)) -> Tensor_Error {
	a_rank, b_rank := len(a.shape), len(b.shape)
	k := a.shape[a_rank-1]
	m := a.shape[0] if a_rank == 2 else 1
	n := b.shape[1] if b_rank == 2 else 1
	a_shape := [2]int{m, k}
	a_strides := [2]int{a.strides[0] if a_rank == 2 else 0, a.strides[a_rank-1]}
	b_shape := [2]int{k, n}
	b_strides := [2]int{b.strides[0], b.strides[1] if b_rank == 2 else 0}
	a_matrix := Tensor_View(T){data = a.data, shape = a_shape[:], strides = a_strides[:], offset = a.offset}
	b_matrix := Tensor_View(T){data = b.data, shape = b_shape[:], strides = b_strides[:], offset = b.offset}
	_validate_autodiff_view(a_matrix) or_return
	_validate_autodiff_view(b_matrix) or_return
	if len(upstream) < m*n || (grad_a != nil && len(grad_a) < m*k) || (grad_b != nil && len(grad_b) < k*n) {
		return .Invalid_Shape
	}
	upstream_shape := [2]int{m, n}
	upstream_strides := [2]int{n, 1}
	upstream_matrix := Tensor_View(T){data = upstream, shape = upstream_shape[:], strides = upstream_strides[:]}
	if grad_a != nil {
		b_transposed_shape := [2]int{n, k}
		b_transposed_strides := [2]int{b_strides[1], b_strides[0]}
		b_transposed := Tensor_View(T){data = b.data, shape = b_transposed_shape[:], strides = b_transposed_strides[:], offset = b.offset}
		_product_kernel(grad_a[:m*k], upstream_matrix, b_transposed, accumulate = true)
	}
	if grad_b != nil {
		a_transposed_shape := [2]int{k, m}
		a_transposed_strides := [2]int{a_strides[1], a_strides[0]}
		a_transposed := Tensor_View(T){data = a.data, shape = a_transposed_shape[:], strides = a_transposed_strides[:], offset = a.offset}
		_product_kernel(grad_b[:k*n], a_transposed, upstream_matrix, accumulate = true)
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

