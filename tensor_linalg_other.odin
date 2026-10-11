#+build !amd64
#+build !i386
package tensor

// Outside x86 the product paths have one compiled copy.

// _product_instructions reports that there are no x86 copies to pick.
_product_instructions :: proc() -> (avx, fma: bool) {
	return false, false
}
