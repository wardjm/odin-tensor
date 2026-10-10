# Odin tensor

A small, generic, row-major tensor library for Odin, with strided views, NumPy-style broadcasting, matrix products, reductions, and a reverse-mode autodiff tape.

## Features

- Owned tensors with explicit allocators, plus borrowed strided views (transpose, axis permutation, slicing, broadcasting) that never copy.
- Element-wise `add`/`sub`/`mul`/`div` with NumPy broadcasting; `exp`, `log`, `sqrt`, `relu`, `sigmoid`, and `tanh` for floats.
- `sum`/`mean`/`prod`/`max`/`min` and `var`/`std` (with `ddof`) over one axis, several axes, or the whole tensor, with negative axes and `keep_dims`; `argmax`/`argmin` over one axis or the whole tensor.
- `dot` for vectors and matrices; `matmul` also for stacks of matrices, with broadcast batch axes.
- Allocation-free `_into` and `_in_place` forms for hot loops.
- Reverse-mode autodiff for add/sub/mul/div, dot/matmul, ReLU/sigmoid/tanh, exp/log/sqrt, sum/mean/max/min reductions, and transpose/slice/reshape views.

## Quick start

Replace the import path with the collection path used by your project:

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

The directory name contains a hyphen, so the import needs an explicit name such as `tensor`. Odin's `:=` cannot redeclare `err`, so later results are declared first and assigned with `=`.

## Coming from NumPy

| NumPy | odin-tensor |
|---|---|
| `np.zeros(shape)` | `new_tensor(T, shape)` |
| `np.array(values).reshape(shape)` | `from_slice(T, values, shape)` |
| `a.reshape(-1, 2)` | `reshape(&a, []int{-1, 2})` (in place, one `-1` allowed) |
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
| `a.prod(axis=0)` | `prod_axis(a, 0)` |
| `a.sum(dtype=np.int64)` | `sum_all(a, i64)` |
| `a.sum(axis=(0, 2))` | `sum_axes(a, []int{0, 2})` |
| `a.std(axis=0, ddof=1)` | `std_axis(a, 0, ddof = 1)` |
| `a.max()` | `max_all(a)` (returns a rank-zero tensor) |
| `a.argmax(axis=1)` | `argmax_axis(a, 1)` (returns a `Tensor(int)`) |
| `a.argmax()` | `argmax_all(a)` (returns a rank-zero `Tensor(int)`) |
| `a @ b`, `np.dot(a, b)` | `matmul(a, b)`, `dot(a, b)` |

Differences to keep in mind:

- Errors are values. Every fallible procedure returns a `Tensor_Error`; nothing panics on bad shapes.
- Results never change type. `mean`, `var`, and `std` don't compile for integer tensors, and integer `sum` and `prod` wrap on overflow where NumPy would widen. `sum_all(a, i64)` opts into a wider accumulator, like NumPy's `dtype=`.
- Integer division truncates and returns `.Division_By_Zero` for a zero divisor, and `.Integer_Overflow` for signed `min(T) / -1`. Float division follows IEEE rules like NumPy: `x/0` is ±Inf or NaN.
- `max` and `min` propagate NaN, like `np.max`.
- Every float sum is pairwise, including strided axes, whole-tensor sums over views, and both sums inside `var` and `std`, so rounding error grows with the log of the element count. NumPy sums pairwise only along the innermost contiguous axis. Integer sums add in order.
- Names follow Odin's snake_case, so NumPy's `keepdims` is spelled `keep_dims` everywhere.
- Views are explicit. A view never owns memory, and procedures that build one take caller-provided shape and stride buffers.
- `slice_view` takes a count instead of a stop index, so reverse and empty slices are unambiguous.
- Element types must be numeric (`add` on `Tensor(bool)` is a compile error); unary math requires floats.

## API overview

### Creating and indexing

- `new_tensor(T, shape, allocator)` creates a zero-filled tensor; `from_slice(T, values, shape, allocator)` copies values. A rank-zero shape (`[]int{}`) is a scalar; dimensions may be zero but not negative. Shapes with more than `MAX_RANK` (64) dimensions return `.Invalid_Shape`.
- `release(&t)` frees a tensor. `fill(&t, value)` overwrites every element.
- `get_at`/`set_at` take one index per axis; negative indices count from the end. The wrong number of indices returns `.Shape_Mismatch`, an index out of range returns `.Out_Of_Bounds`. `get_linear`/`set_linear` address the flat row-major data.
- `reshape(&t, shape)` changes shape in place, inferring one `-1` dimension.

### Views

`as_view(t)` borrows a tensor. `transpose_view`, `permute_view`, `slice_view`, and `broadcast_view` build new views over the same data, with negative axes allowed. `get_at_view`/`set_at_view` index a view. `copy_view` and `transpose_copy` make owned contiguous copies.

### Arithmetic

Each of `add`, `sub`, `mul`, and `div` comes in five forms:

| Form | Inputs | Output |
|---|---|---|
| `add(a, b, allocator)` | tensors | new tensor |
| `add_views(a, b, allocator)` | views | new tensor |
| `add_into(&out, a, b)` | tensors | existing `out` with the exact broadcast shape |
| `add_views_into(&out, a, b)` | views | existing `out` |
| `add_in_place(&a, b)` | tensor | `a`; `b` may broadcast to `a`'s shape but `a` never grows |

`exp`, `log`, `sqrt`, `relu`, `sigmoid`, and `tanh` have the same allocating, `_views`, `_into`, `_views_into`, and `_in_place` forms (`_in_place` takes only the tensor).

`broadcast_shape(a_shape, b_shape, buffer)` computes a broadcast shape without allocating. `scale_in_place(&t, factor)` and `add_scaled_in_place(&t, source, factor)` are simple optimizer updates. `scale_in_place` cannot fail and returns nothing. `add_scaled_in_place` returns `.Shape_Mismatch` unless `source` has exactly the shape of `t`. It doesn't broadcast, because an optimizer step always pairs a parameter with a same-shaped gradient.

### Reductions

`sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, and `min_axis` take `(source, axis, keep_dims, allocator)` and come in `_views`, `_into`, and `_views_into` forms. `sum_all`, `mean_all`, `prod_all`, `max_all`, and `min_all` return a rank-zero tensor and come in `_views` forms. Summing an empty tensor gives zero and its product is one; mean/max/min of an empty reduction return `.Empty_Reduction`. Integer sums and products accumulate in the element type and wrap on overflow, so the product of seven `i8` 2s is -128. To sum in a wider type, pass it as the accumulator: `sum_all(source, i64)` and `sum_all_views(view, i64)` convert each element to `i64` and return a `Tensor(i64)`, so two `i8` 100s give 200 instead of -56. The accumulator must hold every value of the element type (as wide with the same signedness, or wider and signed for an unsigned source); `sum_all(source_i32, u8)` is a compile error.

`sum_axes`, `mean_axes`, `prod_axes`, `max_axes`, and `min_axes` reduce several axes at once and take `(source, axes, keep_dims, allocator)`, with the same `_views`, `_into`, and `_views_into` forms. Negative axes count from the end, a repeated axis returns `.Invalid_Shape`, and an out-of-range one `.Out_Of_Bounds`. An empty `axes` list reduces nothing and copies the source, as NumPy's `axis=()` does.

`var_axis` and `std_axis` take `(source, axis, ddof, keep_dims, allocator)` and come in `_views`, `_into`, and `_views_into` forms. `var_all` and `std_all` take `(source, ddof, allocator)` and come in `_views` forms. They divide by `n - ddof` (`ddof` defaults to 0, as in NumPy) and return `.Empty_Reduction` when `n` is 0 or `ddof >= n`. They take the mean first and then sum the squared deviations, so a large mean relative to the spread costs no precision.

`argmax_axis` and `argmin_axis` take the same arguments as `sum_axis`, come in `_views`, `_into`, and `_views_into` forms, and return a `Tensor(int)` of indices along the axis. Ties pick the first index and the first NaN wins, as in NumPy; an empty axis returns `.Empty_Reduction`. `max`, `min`, `argmax`, and `argmin` need an ordered element type, so complex tensors don't compile.

`argmax_all` and `argmin_all` take `(source, allocator)`, come in `_views` forms, and return the flat row-major index as a rank-zero `Tensor(int)`. For a view the index follows the view's logical order, so a transposed view counts along its own rows, not its storage. Ties and NaN behave as for the axis forms; an empty tensor returns `.Empty_Reduction`.

### Products

`dot` and `matmul` accept rank-one and rank-two operands with NumPy's rules: vector·vector is a rank-zero tensor, matrix·vector and vector·matrix are vectors, matrix·matrix is a matrix. For these ranks the two are identical. Both have `_views`, `_into`, and `_views_into` forms.

`matmul` also takes higher ranks, as `np.matmul` does: each operand is a stack of matrices in its last two axes, and the leading batch axes broadcast like `add`'s, so `[2,1,3,4] @ [5,4,2]` is `[2,5,3,2]`. A vector operand is promoted to a row (left) or column (right) of every matrix, and that axis is dropped from the result. Batch axes that cannot broadcast return `.Shape_Mismatch`. `dot` stays limited to ranks one and two and returns `.Invalid_Shape` for anything higher, since `np.dot` on stacks means something different.

### Convolution windows and pooling

`im2col(source, kernel_height, kernel_width, stride = 1, allocator)` copies every window of a channels-last `[..., height, width, channels]` image into one row, giving `[..., out_height, out_width, kernel_height*kernel_width*channels]` with `out_height = (height - kernel_height)/stride + 1`, rounded down, and likewise for width. Row elements are ordered by window row, then column, then channel, so a `[kernel_height, kernel_width, channels, filters]` kernel reshaped to `[kernel_height*kernel_width*channels, filters]` turns a valid convolution into one `matmul`. Leading axes are batch axes. It has `_views`, `_into`, and `_views_into` forms. A source of rank below 3, or a kernel size or stride below 1, returns `.Invalid_Shape`, and a kernel larger than the image returns `.Shape_Mismatch`.

`max_pool2d(source, pool_height, pool_width, allocator)` takes the maximum of each non-overlapping window, per channel, so `[..., height, width, channels]` becomes `[..., height/pool_height, width/pool_width, channels]`, rounding down. A NaN in a window wins, as in `max_axis`. It has `_views`, `_into`, and `_views_into` forms and the same errors as `im2col`.

## Threads

The library is single-threaded by default. `set_thread_count(n)` lets matrix products (`dot`, `matmul` and their tape and backward forms) use up to `n` threads, counting the caller; `thread_count()` reports the setting, and `set_thread_count(1)` joins the workers again. A product with at least 2¹⁸ multiply-adds is split into one range of output rows per thread, and a batched `matmul` of many small matrices is split by matrix. Each output element is still computed on one thread, adding its products in the same order, so results are identical for every thread count.

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

Physical cores are the right ceiling: on an 8-core, 16-thread CPU, 16 threads were slower than 8. Product shapes with few rows and a long inner dimension, such as a convolution's weight gradient, split poorly. `set_thread_count` is safe to call while other threads use the library. If several threads multiply at once, one gets the workers and the others run on their own threads.

## Allocators

Every procedure that returns a new tensor takes `allocator := context.allocator` as its last parameter, so results never silently inherit an operand's allocator. `_into` and `_in_place` forms never allocate.

## Output aliasing

Since `_into` forms can't allocate a temporary, they reject outputs that share memory with an input in ways that would corrupt the result:

- Element-wise `_into`, `_views_into`, and `_in_place` forms may use the output's own storage as an input, for example `add_into(&a, a, b)`. Any other overlap, such as a transposed view of the output or a row of it broadcast back over it, returns `.Aliased_Output`.
- `dot`/`matmul` `_into` forms and the axis-reduction and `argmax`/`argmin` `_into` forms return `.Aliased_Output` if the output overlaps any input. Use the allocating form or a separate output, for example for `x = x @ w`.

On any error, an `_into` or `_in_place` call leaves its output unchanged.

## Ownership and views

A `Tensor(T)` owns its data and metadata. Call `release(&tensor)` exactly once when finished, and treat tensors as move-only: copying the Odin struct aliases its allocations, so never use or release a copy after the original is released. You may write element values in `data` directly, but don't reassign or resize the `data`, `shape`, or `strides` slices. A `reshape` that changes rank reallocates `shape` and `strides`, which invalidates any view taken from the tensor beforehand.

A `Tensor_View(T)` borrows storage and has no release function. Its source data and caller-provided shape/stride buffers must stay alive and unchanged while the view is used.

## Autodiff

A `Tape(T)` (floating-point `T`) records a graph of operations and computes gradients in reverse:

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

- **Leaves.** `tape_add_leaf_borrow` and `tape_add_leaf_view` borrow data, which must stay alive and unchanged until the tape is reset or released. `tape_add_leaf_copy` snapshots a view into tape-owned storage. Pass `false` as `requires_grad` for constants such as targets.
- **Operations.** `tape_add`, `tape_sub`, `tape_mul`, `tape_div`, `tape_dot`, `tape_matmul`, `tape_relu`, `tape_sigmoid`, `tape_tanh`, `tape_exp`, `tape_log`, `tape_sqrt`, `tape_sum_all`, `tape_mean_all`, `tape_sum_axis`, `tape_mean_axis`, `tape_prod_all`, `tape_prod_axis`, `tape_max_all`, `tape_min_all`, `tape_max_axis`, `tape_min_axis`, `tape_transpose`, `tape_slice`, `tape_reshape`, `tape_im2col`, and `tape_max_pool2d` compute the forward value and record it in one call. `tape_dot` keeps `dot`'s rank-one/rank-two rule, while `tape_matmul` accepts batched operands like `matmul`. Their result is tape-owned; read it with `tape_value`, which returns a view usable with any `_views` procedure. `tape_log` and `tape_sqrt` of zero give an infinite gradient rather than an error, or NaN where the upstream gradient is zero, and that NaN stays in the input's gradient until `tape_zero_grad`. Negative inputs give NaN, as in `log` and `sqrt`. A product's gradient for each element is the product of the other elements, computed without division, so inputs containing zeros get exact gradients. An empty product is 1 and sends no gradient back. Max and min send the whole gradient to the one element `argmax`/`argmin` picks: the first on a tie, or the first NaN. PyTorch instead splits a tie's gradient evenly between the tied elements. `tape_im2col` adds each upstream value back to the image element it was copied from, so an element in several overlapping windows gets the sum. `tape_max_pool2d` sends each window's gradient to the element `argmax` would pick in row-major order.
- **Views.** `tape_transpose(&tape, x)`, `tape_slice(&tape, x, axis, start, count, step)` (`step` defaults to 1), and `tape_reshape(&tape, x, shape)` follow `transpose_view`, `slice_view`, and `reshape`, including negative axes, starts, and steps and one inferred `-1`. Their values borrow `x`'s storage instead of copying it; `tape_reshape` copies only when `x` is not contiguous, for example after `tape_transpose`. Because they share that storage, treat view values as read-only: writing into one changes `x` and every other view of it, and corrupts their gradients. Each upstream gradient value goes back to the element of `x` it was read from.
- **Recording eager results.** The lower-level `tape_record_unary`, `tape_record_binary`, and `tape_record_*` reduction procedures move a result you computed yourself onto the tape. They check shapes but trust that the result matches the named operation. On success they take ownership of the output; on failure the caller keeps it.
- **Gradients.** `tape_backward(&tape, loss)` seeds a rank-zero value with one; `tape_backward_with_seed` takes an explicit seed for other shapes. Only nodes the loss depends on are visited. Gradients accumulate across calls until `tape_zero_grad`; `tape_gradient` returns a view of a node's gradient, or `.Gradient_Not_Tracked` for constants. A node's first backward pass hands it the buffer its gradient was computed in, so nothing is copied; asking for the gradient of a node no backward pass has reached allocates zeros.
- **Handles.** A `Tape_Value` is an id, not a pointer, so a `Tape` can be moved freely. Every call takes the tape explicitly. A handle from another tape, or from before `tape_reset`/`tape_release`, returns `.Invalid_Tape`.
- **Lifetime.** The tape grows as needed (`tape_init(T, reserve)` preallocates). `tape_reset` releases every node but keeps the storage, for rebuilding the graph each training step. The tape draws all of its memory through `tape.allocator`, which keeps freed blocks by size and hands them out again, so a training loop that rebuilds the same graph stops allocating after its second step and avoids a page fault per freshly mapped page. The tape holds its peak memory until `tape_release`, which returns everything to the allocator given to `tape_init`.

Not differentiable yet: `var`/`std`, the several-axes `*_axes` reductions, `permute_view`, and `broadcast_view` (broadcasting inside `tape_add`, `tape_matmul` etc. is differentiable).

## Limitations

- Kernels are plain loops that rely on the compiler to vectorize: contiguous inputs take a direct path and strided inputs advance by row. Matrix products are cache-blocked with register tiles, but there is no hand-written SIMD, so large products are well short of a BLAS.
- Only matrix products use more than one thread. Element-wise operations, reductions, `im2col` and pooling stay on the calling thread.

## Tests

```sh
odin test . -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
```

The define makes leaks and bad frees fail the run; plain `odin test .` only reports them. Several tests use a failing allocator to drive every `.Allocation_Failed` path, and the flag turns a leak on those paths into a failure.

```sh
scripts/check_readme.sh
```

Builds every `odin` code block in this README against the library (with `-vet -strict-style`) and exits with status 1 if any fails to compile. Each block must be a complete `package main` program that imports `"path/to/odin-tensor"`; the script points that import at the checkout.

## Benchmarks

```sh
odin run bench -o:speed
```

Prints the median time of each kernel. Cases with a speed target show their baseline and target, and the run exits with status 1 while any target is unmet.
