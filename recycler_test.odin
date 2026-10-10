package tensor

import "core:mem"
import "core:testing"

// Counting_Allocator forwards to backing and counts the allocations it sees.
Counting_Allocator :: struct {
	backing:     mem.Allocator,
	allocations: int,
}

counting_allocator_proc :: proc(allocator_data: rawptr, mode: mem.Allocator_Mode, size, alignment: int, old_memory: rawptr, old_size: int, location := #caller_location) -> ([]byte, mem.Allocator_Error) {
	state := (^Counting_Allocator)(allocator_data)
	#partial switch mode {
	case .Alloc, .Alloc_Non_Zeroed, .Resize, .Resize_Non_Zeroed:
		state.allocations += 1
	}
	return state.backing.procedure(state.backing.data, mode, size, alignment, old_memory, old_size, location)
}

// training_step records a small dense layer, runs backward, and returns the
// weights' gradient.
training_step :: proc(tape: ^Tape(f64), weights, inputs: ^Tensor(f64)) -> (gradient: Tensor_View(f64), err: Tensor_Error) {
	tape_reset(tape)
	w := tape_add_leaf_borrow(tape, weights) or_return
	x := tape_add_leaf_borrow(tape, inputs, false) or_return
	product := tape_matmul(tape, x, w) or_return
	activated := tape_tanh(tape, product) or_return
	total := tape_sum_all(tape, activated) or_return
	tape_backward(tape, total) or_return
	return tape_gradient(tape, w)
}

@(test)
test_tape_reuses_memory_after_reset :: proc(t: ^testing.T) {
	counter := Counting_Allocator{backing = context.allocator}
	allocator := mem.Allocator{procedure = counting_allocator_proc, data = &counter}
	weights, err := new_tensor(f64, []int{6, 5})
	require_ok(t, err)
	defer release(&weights)
	for &value, i in weights.data do value = 0.1 * f64(i % 7)
	inputs: Tensor(f64)
	inputs, err = new_tensor(f64, []int{4, 6})
	require_ok(t, err)
	defer release(&inputs)
	for &value, i in inputs.data do value = 0.2 * f64(i % 5)

	tape: Tape(f64)
	tape, err = tape_init(f64, 0, allocator)
	require_ok(t, err)
	defer tape_release(&tape)
	first, first_err := training_step(&tape, &weights, &inputs)
	require_ok(t, first_err)
	expected := make([]f64, len(first.data))
	defer delete(expected)
	copy(expected, first.data)

	// The second step's reset files the first step's blocks, creating a free
	// list per size. From then on each step rebuilds the same graph entirely
	// from blocks the previous one freed.
	_, warm_err := training_step(&tape, &weights, &inputs)
	require_ok(t, warm_err)
	before := counter.allocations
	second, second_err := training_step(&tape, &weights, &inputs)
	require_ok(t, second_err)
	testing.expect_value(t, counter.allocations - before, 0)
	for value, i in second.data {
		testing.expect_value(t, value, expected[i])
	}
}

@(test)
test_recycled_blocks_are_zeroed :: proc(t: ^testing.T) {
	tape, err := tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	block: Tensor(f64)
	block, err = new_tensor(f64, []int{16}, tape.allocator)
	require_ok(t, err)
	first := raw_data(block.data)
	fill(&block, 7)
	release(&block)
	block, err = new_tensor(f64, []int{16}, tape.allocator)
	require_ok(t, err)
	defer release(&block)
	testing.expect(t, raw_data(block.data) == first, "the freed block should be reused")
	for value in block.data {
		testing.expect_value(t, value, 0.0)
	}
}

@(test)
test_tape_gradients_allocate_lazily_and_accumulate :: proc(t: ^testing.T) {
	tape, err := tape_init(f64)
	require_ok(t, err)
	defer tape_release(&tape)
	values: Tensor(f64)
	values, err = from_slice(f64, []f64{1, -2, 3}, []int{3})
	require_ok(t, err)
	defer release(&values)
	x := tape_add_leaf_borrow(&tape, &values) or_else panic("leaf")
	doubled := tape_add(&tape, x, x) or_else panic("add")
	total := tape_sum_all(&tape, doubled) or_else panic("sum")

	// Before any backward pass, gradients read as zeros.
	untouched := tape_gradient(&tape, doubled) or_else panic("gradient")
	for value in untouched.data do testing.expect_value(t, value, 0.0)
	require_ok(t, tape_backward(&tape, total))
	require_ok(t, tape_backward(&tape, total))
	gradient := tape_gradient(&tape, x) or_else panic("gradient")
	for value in gradient.data do testing.expect_value(t, value, 4.0)
	intermediate := tape_gradient(&tape, doubled) or_else panic("gradient")
	for value in intermediate.data do testing.expect_value(t, value, 2.0)
	tape_zero_grad(&tape)
	require_ok(t, tape_backward(&tape, total))
	gradient = tape_gradient(&tape, x) or_else panic("gradient")
	for value in gradient.data do testing.expect_value(t, value, 2.0)
}

// binary_gradients returns the gradients of sum(weights * (a op b)) for a and
// b, with b read through view_b.
binary_gradients :: proc(a, b: ^Tensor(f64), view_b: Tensor_View(f64), operation: Tape_Operation) -> (grad_a, grad_b: [dynamic]f64) {
	tape := tape_init(f64) or_else panic("tape")
	defer tape_release(&tape)
	a_value := tape_add_leaf_borrow(&tape, a) or_else panic("a")
	b_value := tape_add_leaf_view(&tape, view_b) or_else panic("b")
	output := apply_tape_helper(&tape, a_value, b_value, operation) or_else panic("op")
	shape := (tape_value(&tape, output) or_else panic("value")).shape
	weights := new_tensor(f64, shape) or_else panic("weights")
	defer release(&weights)
	for &w, i in weights.data do w = 1 + 0.25*f64(i)
	weight_value := tape_add_leaf_copy(&tape, as_view(weights), false) or_else panic("w")
	weighted := tape_mul(&tape, output, weight_value) or_else panic("mul")
	total := tape_sum_all(&tape, weighted) or_else panic("sum")
	if tape_backward(&tape, total) != .None do panic("backward")
	ga := tape_gradient(&tape, a_value) or_else panic("ga")
	gb := tape_gradient(&tape, b_value) or_else panic("gb")
	append(&grad_a, ..ga.data)
	append(&grad_b, ..gb.data)
	return
}

@(test)
test_repeating_operand_backward_matches_general_path :: proc(t: ^testing.T) {
	a, err := from_slice(f64, []f64{0.5, -1.25, 2, 0.75, 1.5, -0.5, 3, 0.25, -2, 1, 0.125, 4}, []int{3, 4})
	require_ok(t, err)
	defer release(&a)
	b: Tensor(f64)
	// Row 1 of a [2, 4] backing tensor: contiguous on its own, or read through
	// a column-reversed view that forces the general path.
	b, err = from_slice(f64, []f64{9, 9, 9, 9, 1.5, -0.75, 2.5, 0.5}, []int{2, 4})
	require_ok(t, err)
	defer release(&b)
	row_shape, row_strides, reversed_shape, reversed_strides: [2]int
	row := slice_view(as_view(b), 0, 1, 1, 1, row_shape[:], row_strides[:]) or_else panic("row")
	reversed := slice_view(row, 1, -1, 4, -1, reversed_shape[:], reversed_strides[:]) or_else panic("reverse")
	reversed_copy := copy_view(reversed) or_else panic("copy")
	defer release(&reversed_copy)
	// The same values as reversed, contiguous and [1, 4], so the leading one
	// is skipped when matching trailing axes.
	contiguous := as_view(reversed_copy)

	for operation in ([]Tape_Operation{.Add, .Sub, .Mul, .Div}) {
		fast_a, fast_b := binary_gradients(&a, &b, contiguous, operation)
		slow_a, slow_b := binary_gradients(&a, &b, reversed, operation)
		defer delete(fast_a)
		defer delete(fast_b)
		defer delete(slow_a)
		defer delete(slow_b)
		for value, i in fast_a do testing.expectf(t, value == slow_a[i], "%v grad a[%d]: %v, want %v", operation, i, value, slow_a[i])
		// slow_b is indexed in the reversed view's logical order, as is fast_b.
		for value, i in fast_b do testing.expectf(t, value == slow_b[i], "%v grad b[%d]: %v, want %v", operation, i, value, slow_b[i])
	}
}
