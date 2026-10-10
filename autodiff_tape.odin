package tensor

// Tape storage, graph recording, and value access.

import "base:intrinsics"
import "core:mem"

// Operation tags identify the forward operation represented by each tape node.
Tape_Operation :: enum {
	Leaf,
	Add,
	Sub,
	Mul,
	Div,
	Dot,
	Matmul,
	Relu,
	Sigmoid,
	Tanh,
	Exp,
	Log,
	Sqrt,
	Sum_All,
	Mean_All,
	Sum_Axis,
	Mean_Axis,
	Prod_All,
	Prod_Axis,
	Max_All,
	Min_All,
	Max_Axis,
	Min_Axis,
	Transpose,
	Slice,
	Reshape,
	Im2col,
	Max_Pool2D,
}

_Tape_Node :: struct($T: typeid) {
	value:         Tensor_View(T),
	owned_value:   Tensor(T),
	owns_value:    bool,
	requires_grad: bool,
	grad:          Tensor(T),
	operation:     Tape_Operation,
	parent_a:      int,
	parent_b:      int,
	parent_count:  int,
	axis:          int,
	keep_dims:     bool,
	start:         int,
	// step is a Slice's step or an Im2col's stride.
	step:          int,
	// kernel_height and kernel_width are an Im2col's kernel or a
	// Max_Pool2D's window.
	kernel_height: int,
	kernel_width:  int,
}

// Tape owns all recorded operation outputs and gradient buffers. Leaf values
// registered with tape_add_leaf_borrow or tape_add_leaf_view remain borrowed.
// The node list grows as needed. A Tape may be moved or copied by value between
// calls, but only one copy may be used and released.
Tape :: struct($T: typeid) {
	nodes:     [dynamic]_Tape_Node(T),
	identity:  u64,
	allocator: mem.Allocator,
}

// Tape_Value identifies one node of one tape. It holds no pointer, so it stays
// valid when the Tape value moves. Handles from another tape, or from before
// tape_reset or tape_release, are rejected with Invalid_Tape.
Tape_Value :: struct($T: typeid) {
	id:       int,
	identity: u64,
}

// _next_tape_identity hands out a distinct, nonzero identity per tape lifetime,
// so a zero Tape_Value is never valid.
@(private)
_next_tape_identity: u64

_new_tape_identity :: proc() -> u64 {
	return intrinsics.atomic_add(&_next_tape_identity, 1) + 1
}

// tape_init creates an empty tape. reserve preallocates room for that many
// nodes; the tape grows beyond it as needed.
tape_init :: proc($T: typeid, reserve := 0, allocator := context.allocator) -> (tape: Tape(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	if reserve < 0 {
		return tape, .Invalid_Shape
	}
	nodes, alloc_err := make([dynamic]_Tape_Node(T), 0, reserve, allocator)
	if alloc_err != nil {
		return tape, .Allocation_Failed
	}
	tape.nodes = nodes
	tape.identity = _new_tape_identity()
	tape.allocator = allocator
	return tape, .None
}

_tape_release_nodes :: proc(tape: ^Tape($T)) {
	for &node in tape.nodes {
		release(&node.grad)
		if node.owns_value {
			release(&node.owned_value)
		}
	}
	clear(&tape.nodes)
}

// tape_release releases every gradient and operation result owned by the tape.
// Borrowed leaf values are not released.
tape_release :: proc(tape: ^Tape($T)) {
	_tape_release_nodes(tape)
	delete(tape.nodes)
	tape^ = {}
}

// tape_reset releases every node but keeps the node storage, so a training
// loop can rebuild its graph each step without reallocating the tape. Handles
// obtained before the reset become invalid.
tape_reset :: proc(tape: ^Tape($T)) {
	_tape_release_nodes(tape)
	tape.identity = _new_tape_identity()
}

// _tape_node returns the node a handle names, or nil if the handle does not
// belong to this tape's current lifetime.
_tape_node :: proc(tape: ^Tape($T), value: Tape_Value(T)) -> ^_Tape_Node(T) {
	if tape == nil || value.identity != tape.identity || value.id < 0 || value.id >= len(tape.nodes) {
		return nil
	}
	return &tape.nodes[value.id]
}

// _tape_append adds a node and returns its handle. On failure nothing changes.
_tape_append :: proc(tape: ^Tape($T), node: _Tape_Node(T)) -> (result: Tape_Value(T), err: Tensor_Error) {
	id := len(tape.nodes)
	if _, alloc_err := append(&tape.nodes, node); alloc_err != nil {
		return result, .Allocation_Failed
	}
	return Tape_Value(T){id = id, identity = tape.identity}, .None
}

// tape_add_leaf_borrow registers a borrowed tensor as a new graph leaf.
tape_add_leaf_borrow :: proc(tape: ^Tape($T), value: ^Tensor(T), requires_grad := true) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return tape_add_leaf_view(tape, as_view(value^), requires_grad)
}

// tape_add_leaf_view registers a borrowed view as a graph leaf. Its data and
// metadata must remain alive and unchanged until the tape is reset or released.
tape_add_leaf_view :: proc(tape: ^Tape($T), value: Tensor_View(T), requires_grad := true) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	if tape == nil || tape.identity == 0 {
		return result, .Invalid_Tape
	}
	if err = _validate_autodiff_view(value); err != .None {
		return result, err
	}
	grad: Tensor(T)
	if requires_grad {
		grad = new_tensor(T, value.shape, tape.allocator) or_return
	}
	result, err = _tape_append(tape, _Tape_Node(T){
		value         = value,
		requires_grad = requires_grad,
		grad          = grad,
		operation     = .Leaf,
		parent_a      = -1,
		parent_b      = -1,
	})
	if err != .None {
		release(&grad)
	}
	return
}

// tape_add_leaf_copy copies a view, in logical order, into tape-owned
// contiguous storage. The source may change or be released afterwards.
tape_add_leaf_copy :: proc(tape: ^Tape($T), value: Tensor_View(T), requires_grad := true) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	if tape == nil || tape.identity == 0 {
		return result, .Invalid_Tape
	}
	if err = _validate_autodiff_view(value); err != .None {
		return result, err
	}
	owned := copy_view(value, tape.allocator) or_return
	result, err = tape_add_leaf_view(tape, as_view(owned), requires_grad)
	if err != .None {
		release(&owned)
		return result, err
	}
	node := &tape.nodes[result.id]
	node.owned_value = owned
	node.owns_value = true
	return result, .None
}

// tape_record_unary moves a ReLU, sigmoid, tanh, exp, log, or sqrt result onto
// the tape. Other operations return Unsupported_Operation.
tape_record_unary :: proc(tape: ^Tape($T), output: ^Tensor(T), operation: Tape_Operation, parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	#partial switch operation {
	case .Relu, .Sigmoid, .Tanh, .Exp, .Log, .Sqrt:
	case:
		return result, .Unsupported_Operation
	}
	return _tape_record_node(tape, output, operation, parent, {}, 1)
}

// tape_record_binary moves an Add, Sub, Mul, Div, Dot, or Matmul result onto
// the tape and records both parent identities. Other operations return
// Unsupported_Operation.
tape_record_binary :: proc(tape: ^Tape($T), output: ^Tensor(T), operation: Tape_Operation, parent_a, parent_b: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	#partial switch operation {
	case .Add, .Sub, .Mul, .Div, .Dot, .Matmul:
	case:
		return result, .Unsupported_Operation
	}
	return _tape_record_node(tape, output, operation, parent_a, parent_b, 2)
}

// tape_record_sum_all and tape_record_mean_all transfer eager global reduction
// results onto the tape. Axis variants retain axis and keep_dims metadata.
tape_record_sum_all :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Sum_All, parent, -1, false)
}

tape_record_mean_all :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Mean_All, parent, -1, false)
}

// tape_record_sum_axis and tape_record_mean_axis accept a negative axis, which
// counts from the end as in NumPy.
tape_record_sum_axis :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T), axis: int, keep_dims := false) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Sum_Axis, parent, axis, keep_dims)
}

tape_record_mean_axis :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T), axis: int, keep_dims := false) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Mean_Axis, parent, axis, keep_dims)
}

// tape_record_prod_all and tape_record_prod_axis transfer eager products.
tape_record_prod_all :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Prod_All, parent, -1, false)
}

tape_record_prod_axis :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T), axis: int, keep_dims := false) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Prod_Axis, parent, axis, keep_dims)
}

// tape_record_max_all, tape_record_min_all, tape_record_max_axis, and
// tape_record_min_axis transfer eager max/min results. Backward sends each
// upstream value to the element argmax or argmin picks.
tape_record_max_all :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Max_All, parent, -1, false)
}

tape_record_min_all :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T)) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Min_All, parent, -1, false)
}

tape_record_max_axis :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T), axis: int, keep_dims := false) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Max_Axis, parent, axis, keep_dims)
}

tape_record_min_axis :: proc(tape: ^Tape($T), output: ^Tensor(T), parent: Tape_Value(T), axis: int, keep_dims := false) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	return _tape_record_reduction(tape, output, .Min_Axis, parent, axis, keep_dims)
}

// _tape_record_reduction transfers a correctly shaped eager result only after
// all parent, operation, and shape checks succeed. Only sums and products
// accept an empty reduction.
_tape_record_reduction :: proc(tape: ^Tape($T), output: ^Tensor(T), operation: Tape_Operation, parent: Tape_Value(T), axis: int, keep_dims: bool) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	parent_node := _tape_node(tape, parent)
	if parent_node == nil {
		return result, .Invalid_Tape
	}
	if err = _validate_autodiff_view(as_view(output^)); err != .None {
		return result, err
	}
	parent_value := parent_node.value
	normalized_axis := axis
	if operation == .Sum_All || operation == .Mean_All || operation == .Prod_All || operation == .Max_All || operation == .Min_All {
		if len(output.shape) != 0 || len(output.data) != 1 {
			return result, .Shape_Mismatch
		}
		if operation != .Sum_All && operation != .Prod_All {
			count, _ := _element_count(parent_value.shape)
			if count == 0 {
				return result, .Empty_Reduction
			}
		}
	} else {
		normalized_axis = _normalize_axis(axis, len(parent_value.shape)) or_return
		if !_reduction_shape_matches(output^, parent_value.shape, normalized_axis, keep_dims) {
			return result, .Shape_Mismatch
		}
		if operation != .Sum_Axis && operation != .Prod_Axis && parent_value.shape[normalized_axis] == 0 {
			return result, .Empty_Reduction
		}
	}
	return _tape_take_output(tape, output, _Tape_Node(T){
		requires_grad = parent_node.requires_grad,
		operation     = operation,
		parent_a      = parent.id,
		parent_b      = -1,
		parent_count  = 1,
		axis          = normalized_axis,
		keep_dims     = keep_dims,
	})
}

// _tape_record_node validates an element-wise, product, or activation result
// against its parents and moves it onto the tape. requires_grad is inherited
// from the parents.
_tape_record_node :: proc(tape: ^Tape($T), output: ^Tensor(T), operation: Tape_Operation, parent_a, parent_b: Tape_Value(T), parent_count: int) -> (result: Tape_Value(T), err: Tensor_Error) where intrinsics.type_is_float(T) {
	node_a := _tape_node(tape, parent_a)
	if node_a == nil {
		return result, .Invalid_Tape
	}
	if err = _validate_autodiff_view(as_view(output^)); err != .None {
		return result, err
	}
	requires_grad := node_a.requires_grad
	parent_b_id := -1
	if parent_count == 2 {
		node_b := _tape_node(tape, parent_b)
		if node_b == nil {
			return result, .Invalid_Tape
		}
		parent_b_id = parent_b.id
		requires_grad = requires_grad || node_b.requires_grad
		a_shape, b_shape := node_a.value.shape, node_b.value.shape
		#partial switch operation {
		case .Add, .Sub, .Mul, .Div:
			if !_broadcast_shapes_match(a_shape, b_shape, output.shape) {
				return result, .Shape_Mismatch
			}
		case .Dot:
			shape_buffer := [2]int{}
			rank := _dot_result_shape(a_shape, b_shape, shape_buffer[:]) or_return
			if !_shapes_match(output.shape, shape_buffer[:rank]) {
				return result, .Shape_Mismatch
			}
		case .Matmul:
			shape_buffer: [MAX_RANK]int
			rank := _matmul_result_shape(a_shape, b_shape, shape_buffer[:]) or_return
			if !_shapes_match(output.shape, shape_buffer[:rank]) {
				return result, .Shape_Mismatch
			}
		}
	} else if !_shapes_match(node_a.value.shape, output.shape) {
		return result, .Shape_Mismatch
	}
	return _tape_take_output(tape, output, _Tape_Node(T){
		requires_grad = requires_grad,
		operation     = operation,
		parent_a      = parent_a.id,
		parent_b      = parent_b_id,
		parent_count  = parent_count,
	})
}

// _tape_take_output allocates the node's gradient, appends it, and only then
// takes ownership of output. On failure output remains owned by the caller.
_tape_take_output :: proc(tape: ^Tape($T), output: ^Tensor(T), template: _Tape_Node(T)) -> (result: Tape_Value(T), err: Tensor_Error) {
	return _tape_take_storage(tape, output, as_view(output^), template)
}

// _tape_take_storage records value and takes ownership of storage, which holds
// value's metadata and, unless value borrows a parent's data, its elements.
// On failure storage remains owned by the caller.
_tape_take_storage :: proc(tape: ^Tape($T), storage: ^Tensor(T), value: Tensor_View(T), template: _Tape_Node(T)) -> (result: Tape_Value(T), err: Tensor_Error) {
	node := template
	if node.requires_grad {
		node.grad = new_tensor(T, value.shape, tape.allocator) or_return
	}
	node.value = value
	node.owned_value = storage^
	node.owns_value = true
	result, err = _tape_append(tape, node)
	if err != .None {
		release(&node.grad)
		return result, err
	}
	storage^ = {}
	return result, .None
}

// tape_value returns the forward value of a node. The view is valid until the
// tape is reset or released.
tape_value :: proc(tape: ^Tape($T), value: Tape_Value(T)) -> (view: Tensor_View(T), err: Tensor_Error) {
	node := _tape_node(tape, value)
	if node == nil {
		return view, .Invalid_Tape
	}
	return node.value, .None
}

// tape_gradient returns a borrowed view of a node's accumulated gradient. The
// node must require gradients.
tape_gradient :: proc(tape: ^Tape($T), value: Tape_Value(T)) -> (view: Tensor_View(T), err: Tensor_Error) {
	node := _tape_node(tape, value)
	if node == nil {
		return view, .Invalid_Tape
	}
	if !node.requires_grad {
		return view, .Gradient_Not_Tracked
	}
	return as_view(node.grad), .None
}

// tape_zero_grad clears every allocated gradient on the tape.
tape_zero_grad :: proc(tape: ^Tape($T)) {
	for &node in tape.nodes {
		if node.requires_grad {
			fill(&node.grad, 0)
		}
	}
}
