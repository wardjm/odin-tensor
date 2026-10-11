# Internals

How odin-tensor computes its results. None of this is needed to use the library; it explains the numerical guarantees in the README and is a map for working on the code.

## Source layout

| File | Contents |
|---|---|
| `tensor.odin` | `Tensor`, `Tensor_View`, errors, creation, indexing, validation helpers |
| `tensor_views.odin` | View construction and copies |
| `tensor_elementwise.odin` | Broadcast arithmetic |
| `tensor_unary.odin` | Unary math and activations |
| `tensor_reductions.odin` | Reductions, pairwise and column sums |
| `tensor_linalg.odin` | `dot`, `matmul`, `linear`, the packed product kernel |
| `tensor_linalg_x86.odin`, `tensor_linalg_other.odin` | Per-CPU product kernel selection |
| `tensor_im2col.odin`, `tensor_pool.odin` | Convolution windows and pooling, with their backward kernels |
| `tensor_threads.odin` | Worker pool, range splitting, the depth-split product |
| `autodiff_tape.odin` | Tape storage, leaves, recording |
| `autodiff.odin` | `tape_*` operations |
| `autodiff_backward.odin` | Reverse pass and gradient kernels |
| `autodiff_recycler.odin` | The tape's recycling allocator |

## Summation

Every float sum is pairwise: runs longer than 128 elements are split in half and the halves summed recursively, and within a run eight interleaved accumulators keep each partial sum short, as NumPy's `pairwise_sum` does. Rounding error grows with the log of the element count. NumPy sums pairwise only along the innermost contiguous axis; here it applies to strided axes, whole-tensor sums over views (row sums are themselves combined pairwise), and both sums inside `var` and `std`. Integer sums add in order.

A float `sum_axis` or `mean_axis` over any axis but the last of a contiguous tensor sums 16 neighbouring outputs at a time along contiguous rows. Each output's splits and accumulators match the single-column pairwise sum, so results are bit-identical to it. Bias gradients and batch normalization's statistics and gradient sums use the same column sums.

## Matrix products

`_product_kernel_serial` follows the usual BLAS layout. The inner dimension is split into blocks; for each, a block of `b` is packed into column panels and a block of `a` into row panels sized for L2. A register tile of SIMD vectors multiplies one row panel by one column panel, whose rows stay in L1 while every row panel of the `a` block reuses it. Edge panels are padded with zeros so every tile takes the same loop. Products too small to repay packing take an unpacked i-k-j loop, and products narrower than a tile sum several rows of each column at once.

Each output element adds its k products in order to its start (zero, or the bias for `linear`), on every path, so neither the path nor the blocking changes a result on a given CPU. On x86, the kernel is compiled three times, for AVX with FMA, AVX alone, and SSE, and picks one at run time. Elsewhere there is one copy with 16-byte vectors; all ARM64 CPUs have FMA. Float products use fused multiply-adds where the CPU supports them, which is why results can differ in the last bits between CPUs with and without FMA.

`linear` starts each output element's sum from its bias, so the bias costs no pass of its own. Its leading axes are treated as one axis of rows, read in place when laid out as in a contiguous tensor and copied otherwise.

## Threads

Work is split into one range per thread once it is large enough: 2¹⁸ multiply-adds for a product, 2¹⁶ elements for element-wise work. Products split by output rows, a batched `matmul` of many small matrices by matrix, `im2col` and pooling by output row, and their backward passes by image or row so no two threads write the same gradient. Each output element is computed on one thread, combining its inputs in the same order, so results are identical for every thread count.

Two cases split differently but keep the same guarantee:

- **Column sums.** When there are fewer 16-column groups than threads, as for 32 channels, each sum is also split along its pairwise tree. The pieces are combined in the tree's shape, giving exactly the sum one thread would compute.
- **Depth split.** A product with an inner dimension of at least 8192 and a small output, such as a convolution's weight gradient `[k·k·c, positions] @ [positions, filters]`, has too few rows to share. Its inner dimension is cut into 2048-long slices whose partial products are added in order, when those partial products fit in 128 KB on the stack, as a first convolution's `[27, 65536] @ [65536, 32]` does in `f32`. The cut depends only on the shape and applies on one thread too. It differs from a single running sum only in rounding: about 4e-5 relative in `f32` over 36,864 terms, slightly closer to the exact sum.

Reductions other than the column sums above, and the backward pass of a general broadcast, stay on the calling thread, since splitting them would change the order of their sums.

Work is split evenly, so a run waits for its slowest thread. On Linux the scheduler sometimes places two workers on the two hardware threads of one core, and that run takes up to twice as long; on an 8-core, 16-thread CPU, 16 threads were slower than 8. Pinning the process to one hardware thread per core, for example `taskset -c 0-7` where CPUs 0 to 7 are distinct cores, keeps times steady. If several threads multiply at once, one gets the workers and the others run on their own threads.

## Autodiff

Each `tape_*` operation computes its forward value from its parents' tape values and records it, so the recorded operation always matches the computation. `tape_backward` visits only nodes reachable from the loss, in reverse order, computing into scratch buffers and committing only if every node succeeded.

- **Buffer reuse.** A node's first backward pass hands it the buffer its gradient was computed in, so nothing is copied. Nodes that write a parent's whole gradient (a reshape, an addition of a same-shaped operand, batch normalization, max pooling, `linear`'s products) get unzeroed buffers.
- **Products.** `tape_prod_*` gives each element the product of the others from prefix and suffix products, so zeros get exact gradients.
- **Views.** `tape_transpose`, `tape_slice` and `tape_reshape` borrow their input's storage. Their backward pass applies the same view to the input's gradient and adds the upstream gradient through it.
- **Convolution windows.** `tape_im2col` adds each upstream value back to the element it was copied from, so elements in overlapping windows sum their gradients; values from padding are dropped.
- **Pooling.** `tape_max_pool2d` sends each window's gradient to the element `argmax` picks in row-major order.
- **`tape_linear`** computes `upstream @ weightsᵀ`, `xᵀ @ upstream`, and each column's pairwise sum of upstream for the bias, so unlike `tape_matmul` followed by `tape_add` neither pass touches a separate copy of the output.
- **`tape_batch_norm`** matches the chain of `tape_mean_axis`, `tape_sub`, `tape_mul`, `tape_sqrt`, `tape_div` and `tape_add` exactly in its forward values. Its backward pass uses the closed form: with `g` the upstream gradient, `x̂ = (x - mean) / deviation` and N rows, `scale`'s gradient is Σ g·x̂ and `x`'s is `scale / deviation · (g - Σg/N - x̂ · Σg·x̂ / N)`. That is two passes over the gradient instead of a dozen. With `relu`, the backward pass recomputes which pre-activation values were positive from `x` rather than reading the output, so the ReLU adds no pass in either direction.

### The recycler

A training loop rebuilds the same graph each step, allocating the same buffers `tape_reset` just freed. Large blocks from the system heap go back to the kernel on free, and taking them again costs a page fault for every page touched. The tape therefore draws all of its memory through a recycling allocator that keeps freed blocks in per-size free lists and hands them out again. Once a graph has been built and reset twice, rebuilding it allocates nothing new. The tape holds its peak memory until `tape_release`.

## Tests

`odin test . -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true` turns leaks and bad frees into failures; plain `odin test .` only reports them. Several tests use a failing allocator to drive every `.Allocation_Failed` path, and the define turns a leak on those paths into a failure. Tests of kernels that write into buffers they don't zero run under an allocator that fills fresh memory with NaN bytes, so an element a kernel misses fails the comparison instead of passing on memory that happened to be zero.

## Benchmarks

`odin run bench -o:speed` prints the median time of each case. Cases with a speed target compare against a baseline measured on the reference machine and fail the run when they fall short. On other hardware, re-measure the baselines first with `odin run bench -o:speed -define:BENCH_BASELINE=true`.
