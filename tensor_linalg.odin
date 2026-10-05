package tensor

// Vector/matrix dot products, matrix multiplication, and scaled updates.

import "base:intrinsics"

// _validate_dot_view accepts rank-one and rank-two views whose addressed
// elements all lie within their backing data.
_validate_dot_view :: proc(view: Tensor_View($T)) -> Tensor_Error {
	if len(view.shape) != 1 && len(view.shape) != 2 {
		return .Invalid_Shape
	}
	return _validate_view(view)
}

// _dot_result_shape applies NumPy's vector/matrix shape rules for ranks 1-2:
// vector·vector is a scalar, matrix·vector and vector·matrix are vectors, and
// matrix·matrix is a matrix.
_dot_result_shape :: proc(a_shape, b_shape, shape_buffer: []int) -> (rank: int, err: Tensor_Error) {
	a_rank, b_rank := len(a_shape), len(b_shape)
	if (a_rank != 1 && a_rank != 2) || (b_rank != 1 && b_rank != 2) {
		return 0, .Invalid_Shape
	}
	if a_shape[a_rank-1] != b_shape[0] {
		return 0, .Shape_Mismatch
	}
	rank = (a_rank-1) + (b_rank-1)
	if len(shape_buffer) < rank {
		return 0, .Insufficient_Buffer
	}
	if a_rank == 2 {
		shape_buffer[0] = a_shape[0]
	}
	if b_rank == 2 {
		shape_buffer[rank-1] = b_shape[1]
	}
	if !_strides_are_valid(shape_buffer[:rank]) {
		return 0, .Invalid_Shape
	}
	return rank, .None
}

// _product_kernel computes output = a @ b with vectors treated as a single row
// (a) or column (b). It loops i-k-j so the innermost loop walks one output row
// and one row of b with fixed steps.
_product_kernel :: proc(output: []$T, a, b: Tensor_View(T)) {
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	k := a.shape[a_rank-1]
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	a_inner_stride := a.strides[a_rank-1]
	b_row_stride := b.strides[0]
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	for &value in output {
		value = 0
	}
	for i in 0..<m {
		output_row := output[i*n:][:n]
		for inner in 0..<k {
			scale := a.data[a.offset + i*a_row_stride + inner*a_inner_stride]
			b_offset := b.offset + inner*b_row_stride
			if b_column_stride == 1 {
				// Contiguous rows of b: a slice loop the compiler can vectorize.
				b_row := b.data[b_offset:][:n]
				for j in 0..<n {
					output_row[j] += scale * b_row[j]
				}
				continue
			}
			for j in 0..<n {
				output_row[j] += scale * b.data[b_offset]
				b_offset += b_column_stride
			}
		}
	}
}

// _product_views_into validates operands and output, then writes a @ b.
_product_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error {
	_validate_dot_view(a) or_return
	_validate_dot_view(b) or_return
	shape_buffer := [2]int{}
	result_rank := _dot_result_shape(a.shape, b.shape, shape_buffer[:]) or_return
	_validate_output(output^) or_return
	if !_shapes_match(output.shape, shape_buffer[:result_rank]) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, a) || _output_overlaps_view(output^, b) {
		return .Aliased_Output
	}
	_product_kernel(output.data, a, b)
	return .None
}

_product_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_dot_view(a) or_return
	_validate_dot_view(b) or_return
	shape_buffer := [2]int{}
	result_rank := _dot_result_shape(a.shape, b.shape, shape_buffer[:]) or_return
	result = new_tensor(T, shape_buffer[:result_rank], allocator) or_return
	_product_kernel(result.data, a, b)
	return result, .None
}

// dot returns the NumPy-style product of rank-one and rank-two tensors:
// vector·vector is a rank-zero tensor, matrix·vector and vector·matrix are
// vectors, and matrix·matrix is a matrix. Inputs may be strided views.
dot :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _product_views(as_view(a), as_view(b), allocator)
}

dot_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _product_views(a, b, allocator)
}

// dot_into and dot_views_into write into an exactly shaped output without
// allocating. Output must not overlap either input; an overlap returns
// Aliased_Output.
dot_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _product_views_into(output, as_view(a), as_view(b))
}

dot_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _product_views_into(output, a, b)
}

// matmul is a @ b. For rank-one and rank-two inputs it matches NumPy's matmul,
// including promotion of a vector operand, and therefore equals dot. Batched
// (rank three and higher) inputs return Invalid_Shape.
matmul :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _product_views(as_view(a), as_view(b), allocator)
}

matmul_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _product_views(a, b, allocator)
}

// matmul_into and matmul_views_into write into an exactly shaped output without
// allocating. Output must not overlap either input; an overlap returns
// Aliased_Output.
matmul_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _product_views_into(output, as_view(a), as_view(b))
}

matmul_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _product_views_into(output, a, b)
}

// scale_in_place multiplies every element by a scalar without allocating.
// Do not use it on storage borrowed by an active autodiff tape.
scale_in_place :: proc(t: ^Tensor($T), factor: T) -> Tensor_Error where intrinsics.type_is_float(T) {
	for i in 0..<len(t.data) {
		t.data[i] *= factor
	}
	return .None
}

// add_scaled_in_place performs t += factor*source without allocating. Shapes
// must match exactly; do not mutate storage borrowed by an active autodiff tape.
add_scaled_in_place :: proc(t: ^Tensor($T), source: Tensor(T), factor: T) -> Tensor_Error where intrinsics.type_is_float(T) {
	if !_shapes_match(t.shape, source.shape) {
		return .Shape_Mismatch
	}
	for i in 0..<len(t.data) {
		t.data[i] += factor*source.data[i]
	}
	return .None
}
