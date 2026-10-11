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
// vector b as a column. Without accumulate, a bias of n values, when given,
// starts each column's sums in place of zero. _product_kernel, in
// tensor_threads.odin, is the entry point that may split it across threads.
// It runs one of several compiled copies of the same paths: on x86, one for
// AVX with FMA, one for AVX alone, and one for SSE; elsewhere one, with
// 16-byte vectors, which NEON fuses. Each output element adds its k products
// in order to its start, on every path, so for a given CPU neither the path
// nor the blocking changes the result. Float products are fused multiply-adds
// where _product_fused reports it, and otherwise a multiply then an add.
_product_kernel_serial :: proc(output: []$T, a, b: Tensor_View(T), accumulate := false, bias: []T = nil) {
	when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
		avx, fma := _product_instructions()
		if fma {
			_product_kernel_fma(output, a, b, accumulate, bias)
			return
		}
		if avx {
			_product_kernel_avx(output, a, b, accumulate, bias)
			return
		}
	}
	#force_inline _product_paths(output, a, b, accumulate, bias, _PRODUCT_FUSED_WITHOUT_AVX, 16)
}

// _PRODUCT_FUSED_WITHOUT_AVX is whether the copy built without x86 features
// fuses: on ARM64 every CPU has fused multiply-adds.
_PRODUCT_FUSED_WITHOUT_AVX :: ODIN_ARCH == .arm64

// _product_fused reports whether float products fuse each multiply-add on this
// CPU.
_product_fused :: proc() -> bool {
	avx, fma := _product_instructions()
	return fma if avx else _PRODUCT_FUSED_WITHOUT_AVX
}

// _product_paths picks the path for _product_kernel_serial, with FUSED
// choosing fused multiply-adds and VECTOR_BYTES the packed tile's vector size.
_product_paths :: #force_inline proc(output: []$T, a, b: Tensor_View(T), accumulate: bool, bias: []T, $FUSED: bool, $VECTOR_BYTES: int) #no_bounds_check {
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	k := a.shape[a_rank-1]
	n := b.shape[1] if b_rank == 2 else 1
	// The unpacked loop walks b along its rows, so when those are strided,
	// even two rows are worth packing.
	b_contiguous := b_rank == 1 || b.strides[1] == 1
	min_rows := _PRODUCT_TILE_ROWS if b_contiguous else 2
	if m >= min_rows && k >= _PRODUCT_MIN_BLOCKED_DEPTH && n >= _PRODUCT_MIN_TILE_COLUMNS {
		#force_inline _product_packed(output, a, b, accumulate, bias, FUSED, 2, max(1, VECTOR_BYTES/size_of(T)))
		return
	}
	if !accumulate {
		_fill_product_start(output, n, bias)
	}
	if m >= _PRODUCT_COLUMN_ROWS && k >= _PRODUCT_MIN_BLOCKED_DEPTH && n < _PRODUCT_MIN_TILE_COLUMNS {
		#force_inline _product_columns(output, a, b, FUSED)
		return
	}
	#force_inline _product_strip(output, a, b, FUSED)
}

// _fill_product_start sets each n-element row of output to bias, or to zero
// without one.
_fill_product_start :: proc(output: []$T, n: int, bias: []T) {
	if bias == nil {
		for &value in output {
			value = 0
		}
		return
	}
	for start := 0; start < len(output); start += n {
		copy(output[start:][:n], bias[:n])
	}
}

// _multiply_add returns sum + x*y, as one fused operation when FUSED is set
// and the elements are floats.
_multiply_add :: #force_inline proc "contextless" (x, y, sum: $V, $FUSED: bool) -> V {
	when FUSED && intrinsics.type_is_float(intrinsics.type_elem_type(V)) {
		return intrinsics.fused_mul_add(x, y, sum)
	} else {
		return sum + x*y
	}
}

// _product_packed computes output = a @ b, or output += a @ b when accumulate
// is set, with the packed path, in a tile VECTORS vectors of LANES elements
// wide. Without accumulate, the first k block's tiles start from zero, or
// from the bias, rather than reading output, which saves a pass over it.
// a_pack holds row panel t's column p at a_pack[(t*depth + p)*ROWS:][:ROWS],
// and b_vectors column panel t's row p at
// b_vectors[(t*depth + p)*VECTORS:][:VECTORS].
_product_packed :: #force_inline proc(output: []$T, a, b: Tensor_View(T), accumulate: bool, bias: []T, $FUSED: bool, $VECTORS, $LANES: int) #no_bounds_check {
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
							start: [^]T = nil if load || bias == nil else ([^]T)(&bias[j0 + column_panel*COLUMNS])
							_product_tile(ROWS, VECTORS, Vector, FUSED, ([^]T)(&output[first]), n, a_panel, b_panel, depth, load, start)
							continue
						}
						edge: [ROWS*COLUMNS]T
						for r in 0..<height {
							for j in 0..<width {
								if load {
									edge[r*COLUMNS + j] = output[first + r*n + j]
								} else if bias != nil {
									edge[r*COLUMNS + j] = bias[j0 + column_panel*COLUMNS + j]
								}
							}
						}
						_product_tile(ROWS, VECTORS, Vector, FUSED, ([^]T)(&edge[0]), COLUMNS, a_panel, b_panel, depth, true, nil)
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
_product_columns :: #force_inline proc(output: []$T, a, b: Tensor_View(T), $FUSED: bool) #no_bounds_check {
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
					sums[r] = _multiply_add(a.data[a_offset + r*a_row_stride + p*a_inner_stride], scale, sums[r], FUSED)
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
				sum = _multiply_add(a.data[a_offset + p*a_inner_stride], b.data[b_offset + p*b_row_stride], sum, FUSED)
			}
			output[i*n + j] = sum
		}
	}
}

// _product_tile adds the ROWS x VECTORS*LANES tile a_panel @ b_panel, depth
// deep, into the matching output tile, whose first element is output[0] and
// whose rows are row_length apart, or stores it there when load is false,
// starting every row from the VECTORS*LANES values at start when given. The
// tile is accumulated in locals so the compiler can keep it in registers.
_product_tile :: #force_inline proc($ROWS, $VECTORS: int, $Vector: typeid/#simd[$LANES]$T, $FUSED: bool, output: [^]T, row_length: int, a_panel: [^]T, b_panel: [^]Vector, depth: int, load: bool, start: [^]T) #no_bounds_check {
	tile: [ROWS][VECTORS]Vector
	if load {
		for r in 0..<ROWS {
			for v in 0..<VECTORS {
				tile[r][v] = intrinsics.unaligned_load((^Vector)(&output[r*row_length + v*LANES]))
			}
		}
	} else if start != nil {
		for v in 0..<VECTORS {
			first_row := intrinsics.unaligned_load((^Vector)(&start[v*LANES]))
			for r in 0..<ROWS {
				tile[r][v] = first_row
			}
		}
	}
	for p in 0..<depth {
		for r in 0..<ROWS {
			scale := Vector(a_panel[p*ROWS + r])
			for v in 0..<VECTORS {
				tile[r][v] = _multiply_add(scale, b_panel[p*VECTORS + v], tile[r][v], FUSED)
			}
		}
	}
	for r in 0..<ROWS {
		for v in 0..<VECTORS {
			intrinsics.unaligned_store((^Vector)(&output[r*row_length + v*LANES]), tile[r][v])
		}
	}
}

// _product_strip adds a @ b into output. The i-k-j order makes the inner loop
// walk one row of output and one row of b with fixed steps.
_product_strip :: #force_inline proc(output: []$T, a, b: Tensor_View(T), $FUSED: bool) #no_bounds_check {
	a_rank, b_rank := len(a.shape), len(b.shape)
	m := a.shape[0] if a_rank == 2 else 1
	n := b.shape[1] if b_rank == 2 else 1
	a_row_stride := a.strides[0] if a_rank == 2 else 0
	a_inner_stride := a.strides[a_rank-1]
	b_row_stride := b.strides[0]
	b_column_stride := b.strides[1] if b_rank == 2 else 0
	k := a.shape[a_rank-1]
	for i in 0..<m {
		output_row := output[i*n:][:n]
		for inner in 0..<k {
			scale := a.data[a.offset + i*a_row_stride + inner*a_inner_stride]
			b_offset := b.offset + inner*b_row_stride
			if b_column_stride == 1 {
				// Contiguous rows of b: a slice loop the compiler can vectorize.
				b_row := b.data[b_offset:][:n]
				for j in 0..<n {
					output_row[j] = _multiply_add(scale, b_row[j], output_row[j], FUSED)
				}
				continue
			}
			for j in 0..<n {
				output_row[j] = _multiply_add(scale, b.data[b_offset], output_row[j], FUSED)
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
	result = _new_tensor(T, shape_buffer[:result_rank], false, allocator) or_return
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

// linear computes x @ weights + bias for an [..., k] x, [k, n] weights and an
// [n] bias, the affine map of a dense layer applied to x's last axis, giving
// [..., n], as PyTorch's Linear does. The bias starts each output element's
// sum, to which its k products are added in order, so it needs no pass of its
// own. x's leading axes are treated as one axis of rows, without a copy when
// they are laid out like a contiguous tensor's. Other ranks or mismatched
// sizes return Shape_Mismatch.
linear :: proc(x, weights, bias: Tensor($T), allocator := context.allocator) -> (Tensor(T), Tensor_Error) where intrinsics.type_is_numeric(T) {
	return linear_views(as_view(x), as_view(weights), as_view(bias), allocator)
}

linear_views :: proc(x, weights, bias: Tensor_View($T), allocator := context.allocator) -> (result: Tensor(T), err: Tensor_Error) where intrinsics.type_is_numeric(T) {
	_validate_view(x) or_return
	_validate_view(weights) or_return
	_validate_view(bias) or_return
	if !_linear_shapes_match(x.shape, weights.shape, bias.shape) {
		return result, .Shape_Mismatch
	}
	starts := bias
	contiguous, contiguous_x: Tensor(T)
	defer release(&contiguous)
	defer release(&contiguous_x)
	if !_is_contiguous_view(bias) {
		contiguous = copy_view(bias, allocator) or_return
		starts = as_view(contiguous)
	}
	rows_shape, rows_strides: [2]int
	rows, collapsed := _linear_rows(x, &rows_shape, &rows_strides)
	if !collapsed {
		contiguous_x = copy_view(x, allocator) or_return
		rows, _ = _linear_rows(as_view(contiguous_x), &rows_shape, &rows_strides)
	}
	shape_buffer: [MAX_RANK]int
	result = _new_tensor(T, _linear_shape(x.shape, weights.shape[1], shape_buffer[:]), false, allocator) or_return
	_product_kernel(result.data, rows, weights, bias = starts.data[starts.offset:][:weights.shape[1]])
	return result, .None
}

// _linear_shapes_match reports whether x, weights and bias are [..., k],
// [k, n] and [n].
_linear_shapes_match :: proc(x, weights, bias: []int) -> bool {
	return len(x) >= 1 && len(weights) == 2 && len(bias) == 1 && x[len(x)-1] == weights[0] && bias[0] == weights[1]
}

// _linear_shape writes linear's result shape, x_shape with its last axis
// replaced by n, into buffer.
_linear_shape :: proc(x_shape: []int, n: int, buffer: []int) -> []int {
	rank := len(x_shape)
	copy(buffer, x_shape[:rank-1])
	buffer[rank-1] = n
	return buffer[:rank]
}

// _linear_rows views x as an [m, k] matrix whose rows are x's leading axes
// taken together, writing its shape and strides into the buffers. That needs
// the leading axes to step through memory as one axis does, as they do in a
// contiguous tensor; collapsed is false when they do not.
_linear_rows :: proc(x: Tensor_View($T), shape, strides: ^[2]int) -> (rows: Tensor_View(T), collapsed: bool) {
	rank := len(x.shape)
	m, row_stride := 1, 0
	// expected is the stride the next outer axis must have, once one is seen.
	expected, seen := 0, false
	for axis := rank-2; axis >= 0; axis -= 1 {
		dim := x.shape[axis]
		if dim == 1 {
			continue
		}
		if seen && x.strides[axis] != expected {
			return rows, false
		}
		if !seen {
			row_stride = x.strides[axis]
		}
		expected, seen = x.strides[axis]*dim, true
		m *= dim
	}
	shape^ = {m, x.shape[rank-1]}
	strides^ = {row_stride, x.strides[rank-1]}
	return Tensor_View(T){data = x.data, shape = shape[:], strides = strides[:], offset = x.offset}, true
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
