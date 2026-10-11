# odin-tensor

A small, generic, row-major tensor library for [Odin](https://odin-lang.org), with strided views, NumPy-style broadcasting, reductions, cache-blocked SIMD matrix products, convolution helpers, and reverse-mode autodiff.

- Owned tensors with explicit allocators, and borrowed strided views (transpose, permute, slice, broadcast) that never copy.
- Element-wise `add`/`sub`/`mul`/`div` with broadcasting; `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`.
- `sum`/`mean`/`prod`/`max`/`min`/`var`/`std` and `argmax`/`argmin` over one axis, several axes, or the whole tensor.
- `dot`, batched `matmul`, and `linear`; cache-blocked SIMD kernels, close to a single-threaded BLAS.
- `im2col` and `max_pool2d` for convolutional networks.
- Allocation-free `_into` and `_in_place` forms for hot loops.
- Optional multithreading whose results are identical for every thread count.
- A gradient tape covering most of the above, plus fused batch normalization.

Errors are values: every fallible procedure returns a `Tensor_Error`, and nothing panics on bad shapes.

## Getting started

Copy or clone the repository somewhere your project can import it. The directory name contains a hyphen, so the import needs an explicit name:

```odin
package main

import "core:fmt"
import tensor "path/to/odin-tensor"

main :: proc() {
    shape := []int{2, 2}
    a, err := tensor.from_slice(i32, []i32{1, 2, 3, 4}, shape)
    if err != .None {
        return
    }
    defer tensor.release(&a)

    b: tensor.Tensor(i32)
    b, err = tensor.new_tensor(i32, shape)
    if err != .None {
        return
    }
    defer tensor.release(&b)
    tensor.fill(&b, 10)

    sum: tensor.Tensor(i32)
    sum, err = tensor.add(a, b)
    if err != .None {
        return
    }
    defer tensor.release(&sum)

    value: i32
    value, err = tensor.get_at(sum, []int{1, 0})
    if err == .None {
        fmt.println(value) // 13
    }
}
```

Odin's `:=` cannot redeclare `err`, so later results are declared first and assigned with `=`.

## Coming from NumPy

| NumPy | odin-tensor |
|---|---|
| `np.zeros(shape)` | `new_tensor(T, shape)` |
| `np.array(values).reshape(shape)` | `from_slice(T, values, shape)` |
| `a.reshape(-1, 2)` | `reshape(&a, []int{-1, 2})` (in place) |
| `a[1, -1]` | `get_at(a, []int{1, -1})`, `set_at(&a, ...)` |
| `a.T` | `transpose_view(as_view(a), shape_buf, strides_buf)` |
| `np.transpose(a, (1, 0, 2))` | `permute_view(as_view(a), []int{1, 0, 2}, ...)` |
| `a[start:start+count*step:step]` on one axis | `slice_view(view, axis, start, count, step, ...)` |
| `np.broadcast_to(a, shape)` | `broadcast_view(as_view(a), shape, ...)` |
| `np.ascontiguousarray(view)` | `copy_view(view)` |
| `a + b`, `a / b` | `add(a, b)`, `div(a, b)` |
| `np.add(a, b, out=c)` | `add_into(&c, a, b)` |
| `a += b` | `add_in_place(&a, b)` |
| `a.sum(axis=-1, keepdims=True)` | `sum_axis(a, -1, keep_dims = true)` |
| `a.sum(axis=(0, 2))` | `sum_axes(a, []int{0, 2})` |
| `a.sum(dtype=np.int64)` | `sum_all(a, i64)` |
| `a.std(axis=0, ddof=1)` | `std_axis(a, 0, ddof = 1)` |
| `a.max()` | `max_all(a)` (a rank-zero tensor) |
| `a.argmax(axis=1)` | `argmax_axis(a, 1)` (a `Tensor(int)`) |
| `a @ b`, `np.dot(a, b)` | `matmul(a, b)`, `dot(a, b)` |

Where odin-tensor differs from NumPy:

- **Types never change.** `mean`, `var` and `std` require floats. Integer `sum` and `prod` wrap on overflow where NumPy would widen; `sum_all(a, i64)` opts into a wider accumulator.
- **Integer division is checked.** It truncates, and returns `.Division_By_Zero` for a zero divisor and `.Integer_Overflow` for signed `min(T) / -1`. Float division follows IEEE rules, as in NumPy.
- **Float sums are pairwise.** This holds everywhere, including strided axes and views, so rounding error grows with the log of the element count. Integer sums add in order.
- **Views are explicit.** A view never owns memory, and procedures that build one take caller-provided shape and stride buffers.
- **Slices take a count.** `slice_view` takes a count instead of a stop index, so reverse and empty slices are unambiguous.
- `max`, `min` and `argmax`/`argmin` propagate NaN like NumPy; ties pick the first index.
- Names are snake_case: `keepdims` is `keep_dims`.

## API

Every procedure that returns a new tensor takes `allocator := context.allocator` as its last parameter.

### Tensors and views

A `Tensor(T)` owns contiguous row-major `data` plus its `shape` and `strides`. A `Tensor_View(T)` borrows data with arbitrary strides and an offset, and has no release procedure.

- `new_tensor(T, shape)` creates a zero-filled tensor; `from_slice(T, values, shape)` copies values. `[]int{}` is a rank-zero scalar. Dimensions may be zero; ranks above `MAX_RANK` (64) return `.Invalid_Shape`.
- `release(&t)` frees a tensor. `fill(&t, value)` sets every element.
- `get_at`/`set_at` take one index per axis, negative from the end. `get_linear`/`set_linear` address the flat data.
- `reshape(&t, shape)` reshapes in place, inferring one `-1`.
- `as_view(t)` borrows a tensor. `transpose_view`, `permute_view`, `slice_view` and `broadcast_view` build views over the same data; `get_at_view`/`set_at_view` index them. `copy_view` and `transpose_copy` make owned contiguous copies.

**Ownership.** Release each tensor exactly once. Treat tensors as move-only: copying the struct aliases its allocations. You may write to `data` directly, but don't reassign or resize `data`, `shape` or `strides`. A `reshape` that changes rank reallocates `shape` and `strides`, invalidating earlier views. A view must not outlive its source data or its shape and stride buffers.

### Operation forms

Most operations come in up to five forms. Using `add` as the example:

| Form | Inputs | Output |
|---|---|---|
| `add(a, b)` | tensors | new tensor |
| `add_views(a, b)` | views | new tensor |
| `add_into(&out, a, b)` | tensors | existing `out` of exactly the result shape |
| `add_views_into(&out, a, b)` | views | existing `out` |
| `add_in_place(&a, b)` | tensor | `a`; `b` may broadcast to `a`, but `a` never grows |

`_into` and `_in_place` forms never allocate, and on error leave their output unchanged. Because they can't take a temporary, they reject outputs that overlap an input unsafely with `.Aliased_Output`:

- Element-wise forms may write over an input's own storage, as in `add_into(&a, a, b)`, but no other overlap.
- Products, reductions, `im2col` and pooling reject any overlap. For `x = x @ w`, use the allocating form.

### Element-wise

- `add`, `sub`, `mul`, `div`: all five forms, with NumPy broadcasting. `broadcast_shape(a_shape, b_shape, buffer)` computes a broadcast shape without allocating.
- `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh` (floats only): all five forms; `_in_place` takes only the tensor.
- `scale_in_place(&t, factor)` and `add_scaled_in_place(&t, source, factor)` (`t += factor*source`, same shape only) for optimizer updates.

### Reductions

| Procedures | Arguments | Forms |
|---|---|---|
| `sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, `min_axis` | `(source, axis, keep_dims)` | `_views`, `_into`, `_views_into` |
| `sum_axes`, `mean_axes`, `prod_axes`, `max_axes`, `min_axes` | `(source, axes, keep_dims)` | `_views`, `_into`, `_views_into` |
| `sum_all`, `mean_all`, `prod_all`, `max_all`, `min_all` | `(source)` | `_views` |
| `var_axis`, `std_axis` | `(source, axis, ddof, keep_dims)` | `_views`, `_into`, `_views_into` |
| `var_all`, `std_all` | `(source, ddof)` | `_views` |
| `argmax_axis`, `argmin_axis` | `(source, axis, keep_dims)` | `_views`, `_into`, `_views_into` |
| `argmax_all`, `argmin_all` | `(source)` | `_views` |

- Axes may be negative. In `*_axes`, a repeated axis returns `.Invalid_Shape`, and an empty list copies the source, like NumPy's `axis=()`.
- `*_all` results are rank-zero tensors. `argmax_all` returns the flat index in the view's logical order.
- An empty sum is 0 and an empty product 1; an empty mean, max, min or arg-reduction returns `.Empty_Reduction`, as does `var`/`std` when `ddof >= n`.
- `sum_all(source, i64)` sums in a wider accumulator type, which must hold every value of the element type.
- `var` and `std` divide by `n - ddof` (`ddof` defaults to 0) and subtract the mean before squaring, so a large mean costs no precision.

### Products

- `dot(a, b)` and `matmul(a, b)` follow NumPy for rank-one and rank-two operands: vector·vector is rank zero, matrix·vector and vector·matrix are vectors, matrix·matrix is a matrix.
- `matmul` also takes stacks of matrices with broadcast batch axes, so `[2,1,3,4] @ [5,4,2]` is `[2,5,3,2]`. `dot` returns `.Invalid_Shape` above rank two, since `np.dot` on stacks means something else.
- `linear(x, weights, bias)` computes `x @ weights + bias` for `[..., k]` `x`, `[k, n]` weights and `[n]` bias, like PyTorch's `Linear`.

`dot` and `matmul` have `_views`, `_into` and `_views_into` forms; `linear` has a `_views` form.

### Convolution and pooling

Both take channels-last images, `[..., height, width, channels]`, with any leading batch axes, and have `_views`, `_into` and `_views_into` forms.

- `im2col(source, kernel_height, kernel_width, stride = 1, padding = 0)` copies each window into one row, giving `[..., out_height, out_width, kernel_height*kernel_width*channels]`, where `out_height = (height + 2*padding - kernel_height)/stride + 1`. Rows are ordered by window row, column, then channel, so multiplying by a `[kernel_height, kernel_width, channels, filters]` kernel reshaped to `[kernel_height*kernel_width*channels, filters]` is a convolution.
- `max_pool2d(source, pool_height, pool_width)` takes the maximum of each non-overlapping window per channel, giving `[..., height/pool_height, width/pool_width, channels]`. Leftover rows and columns are ignored.

## Threads

The library is single-threaded by default. `set_thread_count(n)` lets it use up to `n` threads, counting the caller, and `set_thread_count(1)` stops the workers. Matrix products, element-wise operations, `im2col`, pooling, batch normalization, and their backward passes split large work across threads. Results are bit-identical for every thread count, so you can turn threads on without changing any output.

```odin
package main

import "core:fmt"
import "core:sys/info"
import tensor "path/to/odin-tensor"

main :: proc() {
    physical, _, _ := info.cpu_core_count()
    if err := tensor.set_thread_count(max(physical, 1)); err != .None {
        fmt.eprintln("threads:", err)
        return
    }
    defer tensor.set_thread_count(1)

    a, _ := tensor.new_tensor(f32, []int{512, 512})
    defer tensor.release(&a)
    tensor.fill(&a, 0.5)
    product, err := tensor.matmul(a, a) // split across the threads
    if err != .None {
        fmt.eprintln("matmul:", err)
        return
    }
    defer tensor.release(&product)
    fmt.println(tensor.thread_count(), "threads:", product.data[0]) // 128
}
```

Use the number of physical cores: hyperthreads make these kernels slower. For steady timings on Linux, pin the process to one hardware thread per core, for example with `taskset`. `set_thread_count` is safe to call while other threads use the library.

## Autodiff

A `Tape(T)` for floating-point `T` records operations and computes gradients in reverse:

```odin
package main

import "core:fmt"
import tensor "path/to/odin-tensor"

main :: proc() {
    x, _ := tensor.from_slice(f64, []f64{1, 2, 3}, []int{3})
    defer tensor.release(&x)
    w, _ := tensor.from_slice(f64, []f64{0.5, -1, 2}, []int{3})
    defer tensor.release(&w)

    tape, _ := tensor.tape_init(f64)
    defer tensor.tape_release(&tape)

    // Leaves borrow their tensors; requires_grad defaults to true.
    x_value, _ := tensor.tape_add_leaf_borrow(&tape, &x, false)
    w_value, _ := tensor.tape_add_leaf_borrow(&tape, &w)

    // loss = mean(tanh(x * w))
    product, _ := tensor.tape_mul(&tape, x_value, w_value)
    activated, _ := tensor.tape_tanh(&tape, product)
    loss, _ := tensor.tape_mean_all(&tape, activated)

    if tensor.tape_backward(&tape, loss) == .None {
        loss_view, _ := tensor.tape_value(&tape, loss)
        grad, _ := tensor.tape_gradient(&tape, w_value)
        fmt.println(loss_view.data[0], grad.data)
    }
}
```

**Leaves.** `tape_add_leaf_borrow(&tape, &t, requires_grad = true)` and `tape_add_leaf_view` borrow their data, which must stay alive and unchanged until the tape is reset or released. `tape_add_leaf_copy` copies a view into the tape. Pass `requires_grad = false` for constants such as inputs and targets.

**Operations.** Each computes its forward value and records it in one call, returning a `Tape_Value` handle:

| Kind | Procedures |
|---|---|
| Arithmetic (broadcasting) | `tape_add`, `tape_sub`, `tape_mul`, `tape_div` |
| Products | `tape_dot`, `tape_matmul`, `tape_linear(&tape, x, weights, bias)` |
| Unary | `tape_relu`, `tape_sigmoid`, `tape_tanh`, `tape_exp`, `tape_log`, `tape_sqrt` |
| Reductions | `tape_sum_all`, `tape_mean_all`, `tape_prod_all`, `tape_max_all`, `tape_min_all`, and the `_axis` form of each |
| Views | `tape_transpose`, `tape_slice(&tape, x, axis, start, count, step = 1)`, `tape_reshape` |
| Convolution | `tape_im2col`, `tape_max_pool2d` |
| Normalization | `tape_batch_norm(&tape, x, scale, shift, epsilon, mean, variance, relu)` |

- `tape_value(&tape, v)` returns a node's forward value as a view usable with any `_views` procedure.
- View operations share `x`'s storage (`tape_reshape` copies only a non-contiguous `x`), so treat their values as read-only.
- Max and min send each gradient to the one element `argmax`/`argmin` picks; PyTorch instead splits ties evenly.
- Product gradients are computed without division, so inputs containing zeros get exact gradients.
- `tape_log` and `tape_sqrt` of zero give an infinite gradient (NaN where the upstream gradient is zero).
- `tape_batch_norm` normalizes each feature (the last axis of `x`) by its batch mean and biased variance, then applies `[features]` `scale` and `shift`. It optionally writes the batch statistics into `mean` and `variance` and applies ReLU. It is one fused node, and its gradient flows through the statistics, like PyTorch's batch norm in training mode.
- `tape_record_unary`, `tape_record_binary` and the `tape_record_*` reductions move a result you computed yourself onto the tape. On success the tape owns it.

**Gradients.** `tape_backward(&tape, loss)` seeds a rank-zero value with one; `tape_backward_with_seed` takes an explicit seed for other shapes. Only nodes the loss depends on are visited. Gradients accumulate across calls until `tape_zero_grad`. `tape_gradient` returns a view of a node's gradient, or `.Gradient_Not_Tracked` for constants.

**Handles and lifetime.** A `Tape_Value` is an id, not a pointer, so a `Tape` can be moved freely. A handle from another tape, or from before `tape_reset`, returns `.Invalid_Tape`. `tape_reset` drops every node but keeps the memory, so a training loop that rebuilds the same graph each step stops allocating after its second step. `tape_release` frees everything.

Not yet differentiable: `var`/`std`, the `*_axes` reductions, `permute_view`, and `broadcast_view`. Broadcasting inside `tape_add`, `tape_matmul` and the like is differentiable.

## Performance notes

- Kernels are plain loops the compiler vectorizes, plus packed, register-tiled matrix products. x86 builds pick AVX+FMA, AVX or SSE at run time.
- Float products use fused multiply-adds where the CPU has FMA, so the last bits of a result can differ between CPUs with and without it.

See [docs/internals.md](docs/internals.md) for how the kernels, threading and the tape work.

## Development

```sh
odin test . -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true   # tests; the define fails on leaks
scripts/check_readme.sh                                  # compile every odin block in this README
odin run bench -o:speed                                  # benchmarks
```

Each `odin` block in this README must be a complete `package main` program importing `"path/to/odin-tensor"`; `check_readme.sh` points that import at the checkout. Benchmarks print each kernel's median time, and exit with status 1 if a case misses its speed target.

## License

MIT. See [LICENSE](LICENSE).
