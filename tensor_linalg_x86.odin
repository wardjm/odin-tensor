#+build amd64, i386
package tensor

// The product paths compiled for AVX, with and without FMA, which
// _product_kernel_serial picks between at run time.

import "core:sys/info"

// _product_instructions reports whether this CPU, and its operating system,
// support AVX, and AVX with FMA.
_product_instructions :: proc() -> (avx, fma: bool) {
	features := info.cpu_features()
	avx = .avx in features
	fma = avx && .fma in features
	return
}

@(enable_target_feature = "avx,fma")
_product_kernel_fma :: proc(output: []$T, a, b: Tensor_View(T), accumulate: bool, bias: []T) {
	#force_inline _product_paths(output, a, b, accumulate, bias, true, 32)
}

@(enable_target_feature = "avx")
_product_kernel_avx :: proc(output: []$T, a, b: Tensor_View(T), accumulate: bool, bias: []T) {
	#force_inline _product_paths(output, a, b, accumulate, bias, false, 32)
}
