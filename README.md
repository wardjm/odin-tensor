# Odin tensor

A small, generic, row-major tensor library for Odin, with strided views, NumPy-style broadcasting, matrix products, reductions, and a reverse-mode autodiff tape.

## Features

- Owned tensors with explicit allocators, plus borrowed strided views (transpose, axis permutation, slicing, broadcasting) that never copy.
- Element-wise `add`/`sub`/`mul`/`div` with NumPy broadcasting; `exp`, `log`, `sqrt`, `relu`, `sigmoid`, and `tanh` for floats.
- `sum`/`mean`/`max`/`min` over one axis or the whole tensor, with negative axes and `keep_dims`.
- `dot` and `matmul` for vectors and matrices.
- Allocation-free `_into` and `_in_place` forms for hot loops.
- Reverse-mode autodiff for add/sub/mul/div, dot/matmul, ReLU/sigmoid/tanh, and sum/mean reductions.

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
| `a.sum(axis=-1, keepdims=True)` | `sum_axis(a, -1, true)` |
| `a.prod(axis=0)` | `prod_axis(a, 0)` |
| `a.sum(axis=(0, 2))` | `sum_axes(a, []int{0, 2})` |
| `a.std(axis=0, ddof=1)` | `std_axis(a, 0, ddof = 1)` |
| `a.max()` | `max_all(a)` (returns a rank-zero tensor) |
| `a.argmax(axis=1)` | `argmax_axis(a, 1)` (returns a `Tensor(int)`) |
| `a.argmax()` | `argmax_all(a)` (returns a rank-zero `Tensor(int)`) |
| `a @ b`, `np.dot(a, b)` | `matmul(a, b)`, `dot(a, b)` |

Differences to keep in mind:

- Errors are values. Every fallible procedure returns a `Tensor_Error`; nothing panics on bad shapes.
- Results never change type. `mean`, `var`, and `std` are only defined for floating-point tensors (they are a compile error on integers), and integer `sum` and `prod` accumulate in `T`, so they wrap on overflow where NumPy would widen.
- Integer division truncates and returns `.Division_By_Zero` for a zero divisor, and `.Integer_Overflow` for signed `min(T) / -1`. Float division follows IEEE rules like NumPy: `x/0` is ±Inf or NaN.
- `max` and `min` propagate NaN, like `np.max`.
- Float `sum` and `mean` use pairwise summation, so rounding error grows with the log of the element count. NumPy only sums pairwise along the innermost contiguous axis; here every float sum is pairwise, including strided axes and whole-tensor sums over views. Integer sums add in order. `var` and `std` take a pairwise mean, but sum the squared deviations in order.
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

`broadcast_shape(a_shape, b_shape, buffer)` computes a broadcast shape without allocating. `scale_in_place(&t, factor)` and `add_scaled_in_place(&t, source, factor)` are simple optimizer updates.

### Reductions

`sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, and `min_axis` take `(source, axis, keep_dims, allocator)` and come in `_views`, `_into`, and `_views_into` forms. `sum_all`, `mean_all`, `prod_all`, `max_all`, and `min_all` return a rank-zero tensor and come in `_views` forms. Summing an empty tensor gives zero and its product is one; mean/max/min of an empty reduction return `.Empty_Reduction`. Integer sums and products accumulate in the element type and wrap on overflow, so the product of seven `i8` 2s is -128.

`sum_axes`, `mean_axes`, `prod_axes`, `max_axes`, and `min_axes` reduce several axes at once and take `(source, axes, keep_dims, allocator)`, with the same `_views`, `_into`, and `_views_into` forms. Negative axes count from the end, a repeated axis returns `.Invalid_Shape`, and an out-of-range one `.Out_Of_Bounds`. An empty `axes` list reduces nothing and copies the source, as NumPy's `axis=()` does.

`var_axis` and `std_axis` take `(source, axis, ddof, keep_dims, allocator)` and come in `_views`, `_into`, and `_views_into` forms. `var_all` and `std_all` take `(source, ddof, allocator)` and come in `_views` forms. They divide by `n - ddof` (`ddof` defaults to 0, as in NumPy) and return `.Empty_Reduction` when `n` is 0 or `ddof >= n`. The variance is computed in two passes, the mean first and then the squared deviations, so it does not lose precision when the mean is large compared with the spread.

`argmax_axis` and `argmin_axis` take the same arguments as `sum_axis`, come in `_views`, `_into`, and `_views_into` forms (the output is a `Tensor(int)`), and return a `Tensor(int)` of indices along the axis. Ties pick the first index and the first NaN wins, as in NumPy; an empty axis returns `.Empty_Reduction`. `max`, `min`, `argmax`, and `argmin` need an ordered element type, so complex tensors do not compile.

`argmax_all` and `argmin_all` take `(source, allocator)`, come in `_views` forms, and return the flat row-major index as a rank-zero `Tensor(int)`. For a view the index follows the view's logical order, so a transposed view counts along its own rows, not its storage. Ties and NaN behave as for the axis forms; an empty tensor returns `.Empty_Reduction`.

### Products

`dot` and `matmul` accept rank-one and rank-two operands with NumPy's rules: vector·vector is a rank-zero tensor, matrix·vector and vector·matrix are vectors, matrix·matrix is a matrix. For these ranks the two are identical. Both have `_views`, `_into`, and `_views_into` forms. Batched (rank three and higher) products are not supported and return `.Invalid_Shape`.

## Allocators

Every procedure that returns a new tensor takes `allocator := context.allocator` as its last parameter, so results never silently inherit an operand's allocator. `_into` and `_in_place` forms never allocate.

## Output aliasing

`_into` forms never allocate, so they reject outputs that would be corrupted by reading and writing the same memory:

- Element-wise `_into`, `_views_into`, and `_in_place` forms may use the output's own storage as an input, for example `add_into(&a, a, b)`. Any other overlap, such as a transposed view of the output or a row of it broadcast back over it, returns `.Aliased_Output`.
- `dot`/`matmul` `_into` forms and the axis-reduction and `argmax`/`argmin` `_into` forms return `.Aliased_Output` if the output overlaps any input. Use the allocating form or a separate output, for example for `x = x @ w`.

On any error, an `_into` or `_in_place` call leaves its output unchanged.

## Ownership and views

A `Tensor(T)` owns its data and metadata. Call `release(&tensor)` exactly once when finished, and treat tensors as **move-only**: copying the Odin struct aliases its allocations, so never use or release a copy after the original is released. Element values in `data` may be written directly, but do not reassign or resize the `data`, `shape`, or `strides` slices. A `reshape` that changes rank reallocates `shape` and `strides`, which invalidates any view taken from the tensor beforehand.

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
- **Operations.** `tape_add`, `tape_sub`, `tape_mul`, `tape_div`, `tape_dot`, `tape_matmul`, `tape_relu`, `tape_sigmoid`, `tape_tanh`, `tape_sum_all`, `tape_mean_all`, `tape_sum_axis`, and `tape_mean_axis` compute the forward value and record it in one call. Their result is tape-owned; read it with `tape_value`, which returns a view usable with any `_views` procedure.
- **Recording eager results.** The lower-level `tape_record_unary`, `tape_record_binary`, and `tape_record_*` reduction procedures move a result you computed yourself onto the tape. They check shapes but trust that the result matches the named operation. On success they take ownership of the output; on failure the caller keeps it.
- **Gradients.** `tape_backward(&tape, loss)` seeds a rank-zero value with one; `tape_backward_with_seed` takes an explicit seed for other shapes. Only nodes the loss depends on are visited. Gradients accumulate across calls until `tape_zero_grad`; `tape_gradient` returns a view of a node's gradient, or `.Gradient_Not_Tracked` for constants.
- **Handles.** A `Tape_Value` is an id, not a pointer, so a `Tape` can be moved freely. Every call takes the tape explicitly. A handle from another tape, or from before `tape_reset`/`tape_release`, returns `.Invalid_Tape`.
- **Lifetime.** The tape grows as needed (`tape_init(T, reserve)` preallocates). `tape_reset` releases every node but keeps the storage, for rebuilding the graph each training step.

Not differentiable yet: `exp`, `log`, `sqrt`, `max`/`min`, `prod`, `var`/`std`, and slicing or broadcasting views (broadcasting inside `tape_add` etc. is differentiable).

## Limitations

- No batched (rank > 2) `matmul`.
- Kernels are straightforward loops: contiguous inputs take a direct path and strided inputs advance by row, but there is no SIMD tuning or cache blocking for large matrix products.

## Tests

```sh
odin test . -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
```

The define makes leaks and bad frees fail the run; plain `odin test .` only reports them. Several tests use a failing allocator to drive every `.Allocation_Failed` path, so this flag is what turns a leak on those paths into a test failure.

```sh
scripts/check_readme.sh
```

Builds every `odin` code block in this README against the library (with `-vet -strict-style`) and exits with status 1 if any fails to compile. Each block must be a complete `package main` program that imports `"path/to/odin-tensor"`; the script points that import at the checkout.

## Benchmarks

```sh
odin run bench -o:speed
```

Prints the median time of each kernel. Cases tied to a speed item in `TODO.md` show their baseline and target, and the run exits with status 1 while any target is unmet.
