package tensor

// Vector/matrix dot products, matrix multiplication, and scaled updates.

import "base:intrinsics"
import "core:sys/info"

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

// Blocking for _product_kernel. The packed path follows the usual BLAS
// layout: k is split into blocks of _PRODUCT_BLOCK_DEPTH; for each, a block of
// b up to _PRODUCT_B_BLOCK_BYTES is packed into column panels, and a block of
// a up to _PRODUCT_A_BLOCK_BYTES into row panels, sized for L2. A register
// tile then multiplies one row panel by one column panel, whose
// _PRODUCT_BLOCK_DEPTH rows stay in L1 while every row panel of the a block
// reuses it. Edge panels are padded with zeros, and only the tile's valid
// elements are stored, so every tile takes the same full-width loop. Products
// too small to repay packing (under _PRODUCT_TILE_ROWS rows, or 2 when b's
// rows are strided, _PRODUCT_MIN_BLOCKED_DEPTH inner elements, or
// _PRODUCT_MIN_TILE_COLUMNS columns) skip it.
_PRODUCT_TILE_ROWS :: 6
_PRODUCT_BLOCK_DEPTH :: 256
_PRODUCT_A_BLOCK_BYTES :: 64 * 1024
_PRODUCT_B_BLOCK_BYTES :: 128 * 1024
_PRODUCT_MIN_BLOCKED_DEPTH :: 8
_PRODUCT_MIN_TILE_COLUMNS :: 4
// Products narrower than _PRODUCT_MIN_TILE_COLUMNS, such as a convolution
// down to one channel, sum _PRODUCT_COLUMN_ROWS rows of each column at once.
_PRODUCT_COLUMN_ROWS :: 8

// _product_kernel_serial computes output = a @ b, or output += a @ b when
// accumulate is set, on the calling thread, treating a vector a as a row and a
// vector b as a column. _product_kernel, in tensor_threads.odin, is the entry
// point that may split it across threads. Each output element adds its k
// products in order, one multiply and one add at a time, on every path, so
// neither the path nor the blocking changes the result.
_product_kernel_serial :: proc(output: []$T, a, b: Tensor_View(T), accumulate := false) #no_bounds_check {
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	k := a.shape[a_rank-1]
	n := b.shape[1] if b_rank == 2 else 1
	// The unpacked loop walks b along its rows, so when those are strided,
	// even two rows are worth packing.
	b_contiguous := b_rank == 1 || b.strides[1] == 1
	min_rows := _PRODUCT_TILE_ROWS if b_contiguous else 2
	if m >= min_rows && k >= _PRODUCT_MIN_BLOCKED_DEPTH && n >= _PRODUCT_MIN_TILE_COLUMNS {
		_product_packed(output, a, b, accumulate)
		return
	}
	if !accumulate {
		for &value in output {
			value = 0
		}
	}
	if m >= _PRODUCT_COLUMN_ROWS && k >= _PRODUCT_MIN_BLOCKED_DEPTH && n < _PRODUCT_MIN_TILE_COLUMNS {
		_product_columns(output, a, b)
		return
	}
	#force_inline _product_strip(output, a, b, 0, k, 0, n)
}

// _product_packed computes output = a @ b, or output += a @ b when accumulate
// is set, with the packed path. Without accumulate, the first k block's tiles
// start from zero rather than reading output, which saves a pass over it. On x86 CPUs
// with AVX it runs a copy compiled for AVX, whose tile is 64 bytes of columns
// by _PRODUCT_TILE_ROWS rows in 32-byte vectors; elsewhere the tile's vectors
// are 16 bytes, which SSE and NEON provide. Both multiply and then add, without
// fusing, so they give the same results.
_product_packed :: proc(output: []$T, a, b: Tensor_View(T), accumulate: bool) {
	when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
		if .avx in info.cpu_features() {
			_product_packed_avx(output, a, b, accumulate)
			return
		}
	}
	#force_inline _product_packed_vectors(output, a, b, accumulate, 2, max(1, 16/size_of(T)))
}

when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
	@(enable_target_feature = "avx")
	_product_packed_avx :: proc(output: []$T, a, b: Tensor_View(T), accumulate: bool) {
		#force_inline _product_packed_vectors(output, a, b, accumulate, 2, max(1, 32/size_of(T)))
	}
}

// _product_packed_vectors is _product_packed with a tile VECTORS vectors of
// LANES elements wide. a_pack holds row panel t's column p at
// a_pack[(t*depth + p)*ROWS:][:ROWS], and b_vectors column panel t's row p at
// b_vectors[(t*depth + p)*VECTORS:][:VECTORS].
_product_packed_vectors :: #force_inline proc(output: []$T, a, b: Tensor_View(T), accumulate: bool, $VECTORS, $LANES: int) #no_bounds_check {
	Vector :: #simd[LANES]T
	ROWS :: _PRODUCT_TILE_ROWS
	COLUMNS :: VECTORS * LANES
	DEPTH :: _PRODUCT_BLOCK_DEPTH
	BLOCK_ROWS :: max(ROWS, _PRODUCT_A_BLOCK_BYTES/(DEPTH*size_of(T))/ROWS*ROWS)
	BLOCK_COLUMNS :: max(COLUMNS, _PRODUCT_B_BLOCK_BYTES/(DEPTH*size_of(T))/COLUMNS*COLUMNS)
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	k := a.shape[a_rank-1]
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	a_inner_stride := a.strides[a_rank-1]
	b_row_stride := b.strides[0]
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	a_pack: [BLOCK_ROWS*DEPTH]T = ---
	b_vectors: [DEPTH*BLOCK_COLUMNS/LANES]Vector = ---
	b_pack := ([^]T)(&b_vectors[0])
	for k0 := 0; k0 < k; k0 += DEPTH {
		depth := min(DEPTH, k-k0)
		load := accumulate || k0 > 0
		for j0 := 0; j0 < n; j0 += BLOCK_COLUMNS {
			columns := min(BLOCK_COLUMNS, n-j0)
			column_panels := (columns + COLUMNS - 1) / COLUMNS
			b_block := b
			b_block.offset += k0*b_row_stride + j0*b_column_stride
			_pack_panels(b_pack, b_block, b_row_stride, b_column_stride, depth, columns, COLUMNS, true)
			for i0 := 0; i0 < m; i0 += BLOCK_ROWS {
				rows := min(BLOCK_ROWS, m-i0)
				row_panels := (rows + ROWS - 1) / ROWS
				// a's rows are b's columns, so the same packing fits.
				a_block := a
				a_block.offset += i0*a_row_stride + k0*a_inner_stride
				_pack_panels(([^]T)(&a_pack[0]), a_block, a_inner_stride, a_row_stride, depth, rows, ROWS, false)
				for column_panel in 0..<column_panels {
					width := min(COLUMNS, columns - column_panel*COLUMNS)
					b_panel := ([^]Vector)(&b_vectors[column_panel*depth*VECTORS])
					for row_panel in 0..<row_panels {
						height := min(ROWS, rows - row_panel*ROWS)
						a_panel := ([^]T)(&a_pack[row_panel*depth*ROWS])
						first := (i0 + row_panel*ROWS)*n + j0 + column_panel*COLUMNS
						if height == ROWS && width == COLUMNS {
							_product_tile(ROWS, VECTORS, Vector, ([^]T)(&output[first]), n, a_panel, b_panel, depth, load)
							continue
						}
						edge: [ROWS*COLUMNS]T
						for r in 0..<height {
							for j in 0..<width {
								edge[r*COLUMNS + j] = output[first + r*n + j] if load else 0
							}
						}
						_product_tile(ROWS, VECTORS, Vector, ([^]T)(&edge[0]), COLUMNS, a_panel, b_panel, depth, true)
						for r in 0..<height {
							for j in 0..<width {
								output[first + r*n + j] = edge[r*COLUMNS + j]
							}
						}
					}
				}
			}
		}
	}
}

// _pack_panels copies the depth x count block of source, whose element (p, j)
// is at source.offset + p*depth_stride + j*count_stride, into panels of WIDTH
// columns: panel t's row p goes to pack[(t*depth + p)*WIDTH:][:WIDTH], with
// zeros past count. It reads along whichever of the two axes is contiguous.
_pack_panels :: #force_inline proc(pack: [^]$T, source: Tensor_View(T), depth_stride, count_stride, depth, count: int, $WIDTH: int, $ACROSS: bool) #no_bounds_check {
	panels := (count + WIDTH - 1) / WIDTH
	if ACROSS && (count_stride == 1 || depth_stride != 1) {
		// Row p of the block is read across every panel before row p+1.
		for p in 0..<depth {
			offset := source.offset + p*depth_stride
			for t in 0..<panels {
				width := min(WIDTH, count - t*WIDTH)
				row := pack[(t*depth + p)*WIDTH:][:WIDTH]
				first := offset + t*WIDTH*count_stride
				for j in 0..<width {
					row[j] = source.data[first + j*count_stride]
				}
				for j in width..<WIDTH {
					row[j] = 0
				}
			}
		}
		return
	}
	for t in 0..<panels {
		width := min(WIDTH, count - t*WIDTH)
		panel := pack[t*depth*WIDTH:]
		if count_stride == 1 || depth_stride != 1 {
			for p in 0..<depth {
				row := panel[p*WIDTH:][:WIDTH]
				offset := source.offset + p*depth_stride + t*WIDTH*count_stride
				for j in 0..<width {
					row[j] = source.data[offset + j*count_stride]
				}
				for j in width..<WIDTH {
					row[j] = 0
				}
			}
			continue
		}
		for j in 0..<width {
			offset := source.offset + (t*WIDTH + j)*count_stride
			for p in 0..<depth {
				panel[p*WIDTH + j] = source.data[offset + p]
			}
		}
		for p in 0..<depth {
			for j in width..<WIDTH {
				panel[p*WIDTH + j] = 0
			}
		}
	}
}

// _product_columns adds a @ b into output one column at a time, for products
// too narrow for a column tile. Each column sums _PRODUCT_COLUMN_ROWS rows at
// once, each in its own local, so their additions overlap instead of waiting
// on one another as the i-k-j loop's do, and each output element still adds
// its k products in order.
_product_columns :: proc(output: []$T, a, b: Tensor_View(T)) #no_bounds_check {
	ROWS :: _PRODUCT_COLUMN_ROWS
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	k := a.shape[a_rank-1]
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	a_inner_stride := a.strides[a_rank-1]
	b_row_stride := b.strides[0]
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	full_rows := m - m%ROWS
	for j in 0..<n {
		b_offset := b.offset + j*b_column_stride
		for i0 := 0; i0 < full_rows; i0 += ROWS {
			sums: [ROWS]T = ---
			for r in 0..<ROWS {
				sums[r] = output[(i0+r)*n + j]
			}
			a_offset := a.offset + i0*a_row_stride
			for p in 0..<k {
				scale := b.data[b_offset + p*b_row_stride]
				for r in 0..<ROWS {
					sums[r] += a.data[a_offset + r*a_row_stride + p*a_inner_stride] * scale
				}
			}
			for r in 0..<ROWS {
				output[(i0+r)*n + j] = sums[r]
			}
		}
		for i in full_rows..<m {
			sum := output[i*n + j]
			a_offset := a.offset + i*a_row_stride
			for p in 0..<k {
				sum += a.data[a_offset + p*a_inner_stride] * b.data[b_offset + p*b_row_stride]
			}
			output[i*n + j] = sum
		}
	}
}

// _product_tile adds the ROWS x VECTORS*LANES tile a_panel @ b_panel, depth
// deep, into the matching output tile, whose first element is output[0] and
// whose rows are row_length apart, or stores it there when load is false. The
// tile is accumulated in locals so the compiler can keep it in registers.
_product_tile :: #force_inline proc($ROWS, $VECTORS: int, $Vector: typeid/#simd[$LANES]$T, output: [^]T, row_length: int, a_panel: [^]T, b_panel: [^]Vector, depth: int, load: bool) #no_bounds_check {
	tile: [ROWS][VECTORS]Vector
	if load {
		for r in 0..<ROWS {
			for v in 0..<VECTORS {
				tile[r][v] = intrinsics.unaligned_load((^Vector)(&output[r*row_length + v*LANES]))
			}
		}
	}
	for p in 0..<depth {
		for r in 0..<ROWS {
			scale := Vector(a_panel[p*ROWS + r])
			for v in 0..<VECTORS {
				tile[r][v] += scale * b_panel[p*VECTORS + v]
			}
		}
	}
	for r in 0..<ROWS {
		for v in 0..<VECTORS {
			intrinsics.unaligned_store((^Vector)(&output[r*row_length + v*LANES]), tile[r][v])
		}
	}
}

// _product_strip adds a[:, k0..<k1] @ b[k0..<k1, j0..<j1] into the matching
// columns of output. The i-k-j order makes the inner loop walk one row of
// output and one row of b with fixed steps. Bounds checks stay on: narrow
// products measured about 10% slower without them.
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
// come from _matmul_result_shape. When each product is too small to split but
// there are many, the batches are divided between threads instead; their
// outputs don't overlap, so results are the same.
_matmul_kernel :: proc(output: []$T, a, b: Tensor_View(T), result_shape: []int) {
	a_batch_rank := max(len(a.shape)-2, 0)
	b_batch_rank := max(len(b.shape)-2, 0)
	batch_shape := result_shape[:max(a_batch_rank, b_batch_rank)]
	batch_count, _ := _element_count(batch_shape)
	if batch_count == 0 {
		return
	}
	job := _Matmul_Batches(T){output = output, a = a, b = b, batch_shape = batch_shape, matrix_size = len(output) / batch_count, batch_count = batch_count, per_chunk = batch_count}
	a_matrix, b_matrix := _matmul_matrices(&job)
	m := a_matrix.shape[0] if len(a_matrix.shape) == 2 else 1
	k := a_matrix.shape[len(a_matrix.shape)-1]
	n := b_matrix.shape[1] if len(b_matrix.shape) == 2 else 1
	chunks := 1
	if batch_count > 1 && thread_count() > 1 && m*k*n < _PARALLEL_MIN_WORK && batch_count*m*k*n >= _PARALLEL_MIN_WORK {
		job.per_chunk, chunks = _parallel_chunks(batch_count, 1)
	}
	_parallel_for(chunks, proc(data: rawptr, chunk: int) {
		job := (^_Matmul_Batches(T))(data)
		a_matrix, b_matrix := _matmul_matrices(job)
		a_batch := Tensor_View(T){data = job.a.data, shape = job.a.shape[:max(len(job.a.shape)-2, 0)], strides = job.a.strides[:max(len(job.a.shape)-2, 0)], offset = job.a.offset}
		b_batch := Tensor_View(T){data = job.b.data, shape = job.b.shape[:max(len(job.b.shape)-2, 0)], strides = job.b.strides[:max(len(job.b.shape)-2, 0)], offset = job.b.offset}
		first := chunk * job.per_chunk
		for batch in first..<min(first+job.per_chunk, job.batch_count) {
			a_matrix.offset = _broadcast_offset(a_batch, job.batch_shape, batch)
			b_matrix.offset = _broadcast_offset(b_batch, job.batch_shape, batch)
			_product_kernel(job.output[batch*job.matrix_size:][:job.matrix_size], a_matrix, b_matrix)
		}
	}, &job)
}

_Matmul_Batches :: struct($T: typeid) {
	output:      []T,
	a, b:        Tensor_View(T),
	batch_shape: []int,
	matrix_size: int,
	batch_count: int,
	per_chunk:   int,
}

// _matmul_matrices views the trailing matrix (or vector) axes of a and b.
_matmul_matrices :: proc(job: ^_Matmul_Batches($T)) -> (a_matrix, b_matrix: Tensor_View(T)) {
	a_batch_rank := max(len(job.a.shape)-2, 0)
	b_batch_rank := max(len(job.b.shape)-2, 0)
	a_matrix = Tensor_View(T){data = job.a.data, shape = job.a.shape[a_batch_rank:], strides = job.a.strides[a_batch_rank:]}
	b_matrix = Tensor_View(T){data = job.b.data, shape = job.b.shape[b_batch_rank:], strides = job.b.strides[b_batch_rank:]}
	return
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
