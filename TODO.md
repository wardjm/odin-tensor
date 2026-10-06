# TODO

Each item is small enough for one change, and each "Done when" is something a test can check. Run tests with:

```sh
odin test . -vet -strict-style -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
```

Speed items are checked with the benchmark instead:

```sh
odin run bench -o:speed
```

It prints the median time for each case and exits with status 1 if any case with a target misses it. Baselines are machine-specific; on new hardware, re-measure with `-define:BENCH_BASELINE=true` and update them in `bench/main.odin`.

Every new public procedure also needs a failing-allocator case (see `expect_clean_allocation_failures`) and a README entry.

## Tooling

- [x] **Check the README examples automatically.** Add a script that extracts each ```` ```odin ```` block from `README.md`, builds it against the library, and fails on a compile error.
  - Done when: the script fails after a deliberate typo in an example, and passes on the current README.
- [x] **Put the repo under git with CI.** Initialise git and add a CI job that runs the test command above plus the README check.
  - Done when: a pushed commit with a failing test turns CI red.

## Reductions

- [x] **`argmax_axis` / `argmin_axis`.** Return a `Tensor(int)` of indices along one axis, with negative axes, `keep_dims` and the `_views` forms. Ties pick the first index and a NaN wins, as in NumPy.
  - Done when tests cover: `[[1,5,5],[7,2,7]]` axis 1 gives `[1,0]`; axis 0 gives `[1,0,1]`; NaN at index 2 gives 2; an empty axis gives `.Empty_Reduction`.
- [x] **`argmax_all` / `argmin_all`.** Return the flat row-major index as a rank-zero `Tensor(int)`.
  - Done when: a transposed view returns the index in the view's logical order, not the storage order.
- [x] **`prod_axis` / `prod_all`.** The product of an empty reduction is 1.
  - Done when tests cover: `[[1,2],[3,4]]` gives `[3,8]` (axis 0) and 24 (all); an empty input gives 1; i8 overflow wraps (documented).
- [x] **`var_axis` / `std_axis` / `var_all` / `std_all`.** Floats only, with a `ddof := 0` parameter.
  - Done when tests cover: `[1,2,3,4]` gives var 1.25 and std √1.25; `ddof=1` gives 5/3; `ddof >= count` gives `.Empty_Reduction`.
- [x] **Reduce over several axes.** Add `sum_axes(source, axes: []int, keep_dims)` and the same for mean, max and min. Repeated or out-of-range axes are an error.
  - Done when tests cover: shape `[2,3,4]` over axes `{0,2}` gives shape `[3]` and matches nested `sum_axis` calls; `{0,0}` gives `.Invalid_Shape`; `{-1}` equals `{2}`.
- [x] **Pairwise summation for floats** in `sum`/`mean`, as NumPy does.
  - Done when tests cover: the f32 sum of 10⁷ copies of `0.1` is within 1e-3 relative of 10⁶ (naive summation is off by about 10%); integer sums are unchanged.
- [x] **Pairwise squared deviations in `var`/`std`.** The mean pass is already pairwise; the sum of squared deviations still adds in order. NumPy sums both pairwise.
  - Done when: the f32 `var_all` of 10⁷ elements alternating `0.9` and `1.1` is within 1e-3 relative of 0.01 (adding in order gives about 0.00957); the README no longer lists this as a difference from NumPy.

## Products

- [x] **Batched `matmul`.** Leading dimensions broadcast, as in `np.matmul`; the last two axes multiply.
  - Done when tests cover: `[2,3,4] @ [4,5]` gives `[2,3,5]` and equals two 2-D matmuls; `[2,1,3,4] @ [5,4,2]` gives `[2,5,3,2]`; mismatched batch dimensions give `.Shape_Mismatch`; the `_into` overlap rules still hold.
- [x] **Batched `matmul` in autodiff.**
  - Done when: `check_tape_helper(.Matmul, …)` passes finite differences with a `[2,2,3] @ [3,2]` case.
- [x] **Cache-block large `matmul`.**
  - Done when: existing product tests pass unchanged, and `odin run bench -o:speed` reports PASS for `matmul_into 512` (2× faster than its 17.6 ms baseline).

## Autodiff coverage

- [x] **`tape_exp`, `tape_log`, `tape_sqrt`.** Add the backward rules (exp: y; log: 1/x; sqrt: 1/(2y)).
  - Done when: `check_tape_helper` passes finite differences for each with positive inputs, and `log` of 0 gives an Inf gradient, not an error.
- [x] **`tape_max_axis` / `tape_max_all` (and min).** The gradient goes to the arg-max element, with ties going to the first index (matches `argmax`; PyTorch splits ties evenly, so record the choice in the README).
  - Done when tests cover: finite differences pass on inputs without ties, and the gradient of `[3,3]` is `[1,0]`.
- [x] **Differentiable views: `tape_transpose`, `tape_slice`, `tape_reshape`.**
  - Done when: finite differences pass through a transpose → matmul chain and a slice → sum chain.
- [x] **Per-row offsets in the backward kernels.** `_backward_activation` and `_backward_binary_elementwise` find offsets once per row and step along it instead of computing a full index per element.
  - Done when: all autodiff tests pass unchanged, and `odin run bench -o:speed` reports PASS for `tape_mul + backward 1000x1000` (5× faster than its 130 ms baseline).

## API cleanups

- [x] **`scale_in_place` returns nothing.** It can only ever return `.None`. Also decide whether `add_scaled_in_place` should accept a broadcast `source` (it currently requires the same shape).
  - Done when: the signature changes and the callers in tests compile.
- [x] **Decide `keep_dims` vs `keepdims`.** Either rename to NumPy's spelling or record the decision in the README.
  - Done when: one name is used everywhere (grep finds no mix).
- [x] **Optional wider accumulator for integer sums.** For example `sum_all(source, Acc = i64)`, returning `Tensor(Acc)`.
  - Done when: an i8 `[100,100]` sum into i64 gives 200, and the default still returns i8 (wrapping to -56).

## Test hygiene

- [ ] **Messages on bare `testing.expect`.** Give each one a message, or switch to `expect_value`, so failures say what differed.
  - Done when: `grep -n 'testing.expect(t, ' *_test.odin` returns no lines.
- [ ] **Stop setup failures from cascading into panics.** Stop the test immediately when setup fails.
  - Done when: a deliberately failing `from_slice` in a test reports one failure instead of an index panic.
- [ ] **Pass the allocator through every call in the tensor allocation scenario.** `tensor_allocation_scenario` calls `add`, `sum_axis`, `matmul`, `dot`, `exp` and `mean_all` without `allocator`, so those calls never see the failing allocator and their partial-failure cleanup is untested.
  - Done when: every allocating call in the scenario passes `allocator`, and the tests still pass.
- [ ] **Remove the remaining reads of `tape.nodes[...]` in tests.** Test recorded operation, axis and parent through behaviour (values and gradients) instead.
  - Done when: `grep -n 'tape.nodes\[' autodiff_test.odin` returns no lines.
