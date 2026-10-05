// Benchmarks for odin-tensor. Run from the repository root:
//
//     odin run bench -o:speed
//
// Each case reports the median of several timed runs. Cases with a target
// compare against a baseline measured on the reference machine and fail the
// run (exit code 1) when the speedup falls short, so TODO speed items have a
// pass/fail check. Baselines are machine-specific: re-measure them with
// `odin run bench -o:speed -define:BENCH_BASELINE=true` (which prints medians
// without verdicts) before relying on the verdicts on different hardware.
// Reference machine baselines were measured on 2026-10-05.
package bench

import "core:fmt"
import "core:os"
import "core:slice"
import "core:time"
import tensor ".."

PRINT_BASELINE :: #config(BENCH_BASELINE, false)

Case :: struct {
	name:        string,
	run:         proc(state: ^State),
	repetitions: int,
	// baseline_ms is the median before the targeted optimization; zero means
	// the case is informational.
	baseline_ms: f64,
	// target_speedup is the required baseline_ms / measured_ms.
	target_speedup: f64,
}

State :: struct {
	a, b, row, output:              tensor.Tensor(f32),
	square_512_a, square_512_b:     tensor.Tensor(f32),
	square_512_out:                 tensor.Tensor(f32),
	square_256_a, square_256_b:     tensor.Tensor(f32),
	square_256_out:                 tensor.Tensor(f32),
	grad_left, grad_right:          tensor.Tensor(f64),
}

N :: 1000

CASES := []Case{
	{name = "add (same shape)", run = bench_add, repetitions = 20},
	{name = "add_into (same shape)", run = bench_add_into, repetitions = 20},
	{name = "add (broadcast row)", run = bench_add_broadcast, repetitions = 20},
	{name = "div_in_place", run = bench_div_in_place, repetitions = 20},
	{name = "sqrt_views_into (transposed)", run = bench_sqrt_transposed, repetitions = 20},
	{name = "sum_axis 0", run = bench_sum_axis, repetitions = 20},
	{name = "sum_all", run = bench_sum_all, repetitions = 20},
	{name = "matmul_into 256", run = bench_matmul_256, repetitions = 10},
	// TODO.md: cache-block large matmul, at least 2x faster.
	{name = "matmul_into 512", run = bench_matmul_512, repetitions = 5, baseline_ms = 17.6, target_speedup = 2},
	// TODO.md: per-row offsets in backward kernels, at least 5x faster.
	{name = "tape_mul + backward 1000x1000", run = bench_tape_mul_backward, repetitions = 5, baseline_ms = 130, target_speedup = 5},
}

main :: proc() {
	state: State
	setup(&state)
	defer teardown(&state)

	missed := 0
	fmt.printfln("%-32s %10s %10s %9s %8s  %s", "case", "median ms", "baseline", "speedup", "target", "verdict")
	for c in CASES {
		median := measure(c, &state)
		median_text := fmt.tprintf("%.3f", median)
		if c.baseline_ms <= 0 || PRINT_BASELINE {
			fmt.printfln("%-32s %10s %10s %9s %8s", c.name, median_text, "-", "-", "-")
			continue
		}
		speedup := c.baseline_ms / median
		verdict := "PASS"
		if speedup < c.target_speedup {
			verdict = "MISS"
			missed += 1
		}
		fmt.printfln("%-32s %10s %10s %9s %8s  %s", c.name, median_text,
			fmt.tprintf("%.3f", c.baseline_ms), fmt.tprintf("%.2fx", speedup), fmt.tprintf("%.1fx", c.target_speedup), verdict)
	}
	if missed > 0 {
		fmt.printfln("%d speed target(s) missed", missed)
		os.exit(1)
	}
}

measure :: proc(c: Case, state: ^State) -> f64 {
	c.run(state) // warm up caches and page in memory
	samples := make([]f64, c.repetitions)
	defer delete(samples)
	for &sample in samples {
		start := time.tick_now()
		c.run(state)
		sample = time.duration_milliseconds(time.tick_since(start))
	}
	slice.sort(samples)
	return samples[len(samples)/2]
}

must :: proc(err: tensor.Tensor_Error, loc := #caller_location) {
	if err != .None {
		fmt.eprintfln("%v: setup failed with %v", loc, err)
		os.exit(2)
	}
}

filled :: proc($T: typeid, shape: []int, value: T) -> tensor.Tensor(T) {
	t, err := tensor.new_tensor(T, shape)
	must(err)
	tensor.fill(&t, value)
	return t
}

setup :: proc(s: ^State) {
	s.a = filled(f32, []int{N, N}, 1.5)
	s.b = filled(f32, []int{N, N}, 2)
	s.row = filled(f32, []int{N}, 3)
	s.output = filled(f32, []int{N, N}, 0)
	s.square_256_a = filled(f32, []int{256, 256}, 1)
	s.square_256_b = filled(f32, []int{256, 256}, 1)
	s.square_256_out = filled(f32, []int{256, 256}, 0)
	s.square_512_a = filled(f32, []int{512, 512}, 1)
	s.square_512_b = filled(f32, []int{512, 512}, 1)
	s.square_512_out = filled(f32, []int{512, 512}, 0)
	s.grad_left = filled(f64, []int{N, N}, 0.5)
	s.grad_right = filled(f64, []int{N, N}, 2)
}

teardown :: proc(s: ^State) {
	for t in ([]^tensor.Tensor(f32){&s.a, &s.b, &s.row, &s.output, &s.square_256_a, &s.square_256_b, &s.square_256_out, &s.square_512_a, &s.square_512_b, &s.square_512_out}) {
		tensor.release(t)
	}
	tensor.release(&s.grad_left)
	tensor.release(&s.grad_right)
}

bench_add :: proc(s: ^State) {
	result, err := tensor.add(s.a, s.b)
	must(err)
	tensor.release(&result)
}

bench_add_into :: proc(s: ^State) {
	must(tensor.add_into(&s.output, s.a, s.b))
}

bench_add_broadcast :: proc(s: ^State) {
	result, err := tensor.add(s.a, s.row)
	must(err)
	tensor.release(&result)
}

bench_div_in_place :: proc(s: ^State) {
	must(tensor.div_in_place(&s.output, s.b))
}

bench_sqrt_transposed :: proc(s: ^State) {
	shape, strides: [2]int
	transposed, err := tensor.transpose_view(tensor.as_view(s.a), shape[:], strides[:])
	must(err)
	must(tensor.sqrt_views_into(&s.output, transposed))
}

bench_sum_axis :: proc(s: ^State) {
	result, err := tensor.sum_axis(s.a, 0)
	must(err)
	tensor.release(&result)
}

bench_sum_all :: proc(s: ^State) {
	result, err := tensor.sum_all(s.a)
	must(err)
	tensor.release(&result)
}

bench_matmul_256 :: proc(s: ^State) {
	must(tensor.matmul_into(&s.square_256_out, s.square_256_a, s.square_256_b))
}

bench_matmul_512 :: proc(s: ^State) {
	must(tensor.matmul_into(&s.square_512_out, s.square_512_a, s.square_512_b))
}

bench_tape_mul_backward :: proc(s: ^State) {
	tape, err := tensor.tape_init(f64)
	must(err)
	defer tensor.tape_release(&tape)
	left, left_err := tensor.tape_add_leaf_borrow(&tape, &s.grad_left)
	must(left_err)
	right, right_err := tensor.tape_add_leaf_borrow(&tape, &s.grad_right)
	must(right_err)
	product, product_err := tensor.tape_mul(&tape, left, right)
	must(product_err)
	loss, loss_err := tensor.tape_sum_all(&tape, product)
	must(loss_err)
	must(tensor.tape_backward(&tape, loss))
}
