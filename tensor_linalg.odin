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

// Blocking for _product_kernel. Output is computed in register tiles of
// _PRODUCT_TILE_ROWS rows by _PRODUCT_TILE_BYTES of columns, and k is split
// into blocks of _PRODUCT_BLOCK_DEPTH so each tile's slice of b can be packed
// into a contiguous panel that stays in L1 while every row tile reuses it.
// Sizes were tuned on matmul_into 512 in bench. Products with fewer than
// _PRODUCT_TILE_ROWS rows, fewer than _PRODUCT_MIN_BLOCKED_DEPTH inner
// elements, or less than one column tile reuse b too little to pay for packing
// and skip blocking.
_PRODUCT_TILE_ROWS :: 2
_PRODUCT_TILE_BYTES :: 64
_PRODUCT_BLOCK_DEPTH :: 256
_PRODUCT_MIN_BLOCKED_DEPTH :: 8

// _product_kernel computes output = a @ b with vectors treated as a single row
// (a) or column (b). The outer loop walks k blocks; within each, column tiles
// pack their panel of b and then sweep the row tiles, and columns past the
// last full tile take the unpacked i-k-j loop. Each output element still sums
// its k products in order from zero, so blocking does not change the result.
// The inlining directives on the _product_strip calls are measured choices:
// either way round, narrow or tail-heavy products lose 10-15%.
_product_kernel :: proc(output: []$T, a, b: Tensor_View(T)) #no_bounds_check {
	TILE_COLUMNS :: max(1, _PRODUCT_TILE_BYTES/size_of(T))
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
	if m < _PRODUCT_TILE_ROWS || k < _PRODUCT_MIN_BLOCKED_DEPTH || n < TILE_COLUMNS {
		#force_inline _product_strip(output, a, b, 0, k, 0, n)
		return
	}
	full_rows := m - m%_PRODUCT_TILE_ROWS
	full_columns := n - n%TILE_COLUMNS
	panel: [_PRODUCT_BLOCK_DEPTH*TILE_COLUMNS]T = ---
	for k0 := 0; k0 < k; k0 += _PRODUCT_BLOCK_DEPTH {
		depth := min(_PRODUCT_BLOCK_DEPTH, k-k0)
		a_block := a
		a_block.offset += k0*a_inner_stride
		block_panel := panel[:depth*TILE_COLUMNS]
		for j0 := 0; j0 < full_columns; j0 += TILE_COLUMNS {
			for p in 0..<depth {
				b_offset := b.offset + (k0+p)*b_row_stride + j0*b_column_stride
				for j in 0..<TILE_COLUMNS {
					panel[p*TILE_COLUMNS + j] = b.data[b_offset + j*b_column_stride]
				}
			}
			for i0 := 0; i0 < full_rows; i0 += _PRODUCT_TILE_ROWS {
				_product_tile(_PRODUCT_TILE_ROWS, TILE_COLUMNS, output[i0*n + j0:], n, a_block, i0, a_row_stride, a_inner_stride, block_panel)
			}
			for i in full_rows..<m {
				_product_tile(1, TILE_COLUMNS, output[i*n + j0:], n, a_block, i, a_row_stride, a_inner_stride, block_panel)
			}
		}
		if full_columns < n {
			#force_no_inline _product_strip(output, a, b, k0, k0+depth, full_columns, n)
		}
	}
}

// _product_tile adds a's rows i0..<i0+ROWS times the packed panel, whose rows
// are TILE_COLUMNS wide, into the matching output tile, whose first element is
// output[0] and whose rows are row_length apart. The tile is accumulated in
// locals so the compiler can keep it in registers.
_product_tile :: #force_inline proc($ROWS, $TILE_COLUMNS: int, output: []$T, row_length: int, a: Tensor_View(T), i0, a_row_stride, a_inner_stride: int, panel: []T) #no_bounds_check {
	tile: [ROWS][TILE_COLUMNS]T = ---
	for r in 0..<ROWS {
		for j in 0..<TILE_COLUMNS {
			tile[r][j] = output[r*row_length + j]
		}
	}
	a_offset := a.offset + i0*a_row_stride
	for p in 0..<len(panel)/TILE_COLUMNS {
		b_row := panel[p*TILE_COLUMNS:][:TILE_COLUMNS]
		for r in 0..<ROWS {
			scale := a.data[a_offset + r*a_row_stride + p*a_inner_stride]
			for j in 0..<TILE_COLUMNS {
				tile[r][j] += scale * b_row[j]
			}
		}
	}
	for r in 0..<ROWS {
		for j in 0..<TILE_COLUMNS {
			output[r*row_length + j] = tile[r][j]
		}
	}
}

// _product_strip adds a[:, k0..<k1] @ b[k0..<k1, j0..<j1] into the matching
// columns of output for every row. It loops i-k-j so the innermost loop walks
// one output row and one row of b with fixed steps. It keeps bounds checks
// because narrow products measured about 10% slower without them.
_product_strip :: proc(output: []$T, a, b: Tensor_View(T), k0, k1, j0, j1: int) {
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	a_inner_stride := a.strides[a_rank-1]
	b_row_stride := b.strides[0]
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	width := j1 - j0
	for i in 0..<m {
		output_row := output[i*n + j0:][:width]
		for inner in k0..<k1 {
			scale := a.data[a.offset + i*a_row_stride + inner*a_inner_stride]
			b_offset := b.offset + inner*b_row_stride + j0*b_column_stride
			if b_column_stride == 1 {
				// Contiguous rows of b: a slice loop the compiler can vectorize.
				b_row := b.data[b_offset:][:width]
				for j in 0..<width {
					output_row[j] += scale * b_row[j]
				}
				continue
			}
			for j in 0..<width {
				output_row[j] += scale * b.data[b_offset]
				b_offset += b_column_stride
			}
		}
	}
}

// _validate_dot_ranks enforces dot's rank-one/rank-two rule. On those ranks
// matmul's shape rules and kernel agree with dot, so the products below defer
// to the matmul paths after this check.
_validate_dot_ranks :: proc(a, b: Tensor_View($T)) -> Tensor_Error {
	a_rank, b_rank := len(a.shape), len(b.shape)
	if (a_rank != 1 && a_rank != 2) || (b_rank != 1 && b_rank != 2) {
		return .Invalid_Shape
	}
	return .None
}

_product_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error {
	_validate_dot_ranks(a, b) or_return
	return _matmul_views_into(output, a, b)
}

_product_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_dot_ranks(a, b) or_return
	return _matmul_views(a, b, allocator)
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

// _matmul_result_shape applies np.matmul's shape rules. A vector a is a single
// row and a vector b a single column, and that promoted axis is dropped from the
// result. Axes before the last two are batch axes, which broadcast as in add.
_matmul_result_shape :: proc(a_shape, b_shape, shape_buffer: []int) -> (rank: int, err: Tensor_Error) {
	a_rank, b_rank := len(a_shape), len(b_shape)
	if a_rank == 0 || b_rank == 0 {
		return 0, .Invalid_Shape
	}
	b_inner := b_shape[b_rank-2] if b_rank >= 2 else b_shape[0]
	if a_shape[a_rank-1] != b_inner {
		return 0, .Shape_Mismatch
	}
	a_batch := a_shape[:max(a_rank-2, 0)]
	b_batch := b_shape[:max(b_rank-2, 0)]
	rank = max(len(a_batch), len(b_batch))
	if a_rank >= 2 {
		rank += 1
	}
	if b_rank >= 2 {
		rank += 1
	}
	if len(shape_buffer) < rank {
		return 0, .Insufficient_Buffer
	}
	batch_rank := broadcast_shape(a_batch, b_batch, shape_buffer) or_return
	if a_rank >= 2 {
		shape_buffer[batch_rank] = a_shape[a_rank-2]
	}
	if b_rank >= 2 {
		shape_buffer[rank-1] = b_shape[b_rank-1]
	}
	if _, valid := _element_count(shape_buffer[:rank]); !valid || !_strides_are_valid(shape_buffer[:rank]) {
		return 0, .Invalid_Shape
	}
	return rank, .None
}

// _matmul_kernel runs _product_kernel once per batch index, each writing one
// contiguous matrix of output. a and b must be validated and result_shape must
// come from _matmul_result_shape.
_matmul_kernel :: proc(output: []$T, a, b: Tensor_View(T), result_shape: []int) {
	a_batch_rank := max(len(a.shape)-2, 0)
	b_batch_rank := max(len(b.shape)-2, 0)
	a_batch := Tensor_View(T){data = a.data, shape = a.shape[:a_batch_rank], strides = a.strides[:a_batch_rank], offset = a.offset}
	b_batch := Tensor_View(T){data = b.data, shape = b.shape[:b_batch_rank], strides = b.strides[:b_batch_rank], offset = b.offset}
	a_matrix := Tensor_View(T){data = a.data, shape = a.shape[a_batch_rank:], strides = a.strides[a_batch_rank:]}
	b_matrix := Tensor_View(T){data = b.data, shape = b.shape[b_batch_rank:], strides = b.strides[b_batch_rank:]}
	batch_shape := result_shape[:max(a_batch_rank, b_batch_rank)]
	batch_count, _ := _element_count(batch_shape)
	if batch_count == 0 {
		return
	}
	matrix_size := len(output) / batch_count
	for batch in 0..<batch_count {
		a_matrix.offset = _broadcast_offset(a_batch, batch_shape, batch)
		b_matrix.offset = _broadcast_offset(b_batch, batch_shape, batch)
		_product_kernel(output[batch*matrix_size:][:matrix_size], a_matrix, b_matrix)
	}
}

_matmul_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error {
	_validate_view(a) or_return
	_validate_view(b) or_return
	shape_buffer: [MAX_RANK]int
	result_rank := _matmul_result_shape(a.shape, b.shape, shape_buffer[:]) or_return
	_validate_output(output^) or_return
	if !_shapes_match(output.shape, shape_buffer[:result_rank]) {
		return .Shape_Mismatch
	}
	if _output_overlaps_view(output^, a) || _output_overlaps_view(output^, b) {
		return .Aliased_Output
	}
	_matmul_kernel(output.data, a, b, shape_buffer[:result_rank])
	return .None
}

_matmul_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) {
	_validate_view(a) or_return
	_validate_view(b) or_return
	shape_buffer: [MAX_RANK]int
	result_rank := _matmul_result_shape(a.shape, b.shape, shape_buffer[:]) or_return
	result = new_tensor(T, shape_buffer[:result_rank], allocator) or_return
	_matmul_kernel(result.data, a, b, shape_buffer[:result_rank])
	return result, .None
}

// matmul is a @ b with np.matmul's rules. A vector a is promoted to a row and a
// vector b to a column, and the promoted axis is dropped from the result, so
// for rank-one and rank-two inputs matmul equals dot. Higher ranks are stacks
// of matrices: the last two axes multiply and the leading (batch) axes
// broadcast, so [2,1,3,4] @ [5,4,2] is [2,5,3,2]. Rank-zero inputs return
// Invalid_Shape.
matmul :: proc(a, b: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _matmul_views(as_view(a), as_view(b), allocator)
}

matmul_views :: proc(a, b: Tensor_View($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return _matmul_views(a, b, allocator)
}

// matmul_into and matmul_views_into write into an exactly shaped output without
// allocating. Output must not overlap either input; an overlap returns
// Aliased_Output.
matmul_into :: proc(output: ^Tensor($T), a, b: Tensor(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _matmul_views_into(output, as_view(a), as_view(b))
}

matmul_views_into :: proc(output: ^Tensor($T), a, b: Tensor_View(T)) -> Tensor_Error where intrinsics.type_is_numeric(T) {
	return _matmul_views_into(output, a, b)
}

// scale_in_place multiplies every element by a scalar without allocating.
// Do not use it on storage borrowed by an active autodiff tape.
scale_in_place :: proc(t: ^Tensor($T), factor: T) where intrinsics.type_is_float(T) {
	for i in 0..<len(t.data) {
		t.data[i] *= factor
	}
}

// add_scaled_in_place performs t += factor*source without allocating. Shapes
// must match exactly (source does not broadcast); a mismatch returns
// Shape_Mismatch. Do not mutate storage borrowed by an active autodiff tape.
add_scaled_in_place :: proc(t: ^Tensor($T), source: Tensor(T), factor: T) -> Tensor_Error where intrinsics.type_is_float(T) {
	if !_shapes_match(t.shape, source.shape) {
		return .Shape_Mismatch
	}
	for i in 0..<len(t.data) {
		t.data[i] += factor*source.data[i]
	}
	return .None
}
