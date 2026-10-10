package tensor

// The tape's recycling allocator.
//
// Training rebuilds the same graph every step, so each step allocates the same
// set of buffers that tape_reset just freed. Large blocks from the system heap
// go straight back to the kernel on free, and taking them again costs a page
// fault for every page touched. The recycler keeps freed blocks in per-size
// free lists and hands them out again, so once a graph has been built and
// reset twice, rebuilding it allocates nothing new. It holds its blocks until
// tape_release.

import "core:mem"

_Recycler :: struct {
	backing: mem.Allocator,
	// free maps a block size in bytes to blocks of that size awaiting reuse.
	free:    map[int][dynamic]rawptr,
}

// _recycler_create allocates a recycler from backing. Its allocator is valid
// until _recycler_destroy.
_recycler_create :: proc(backing: mem.Allocator) -> (recycler: ^_Recycler, err: Tensor_Error) {
	alloc_err: mem.Allocator_Error
	recycler, alloc_err = new(_Recycler, backing)
	if alloc_err != nil {
		return nil, .Allocation_Failed
	}
	recycler.backing = backing
	recycler.free = make(map[int][dynamic]rawptr, backing)
	return recycler, .None
}

_recycler_allocator :: proc(recycler: ^_Recycler) -> mem.Allocator {
	return mem.Allocator{procedure = _recycler_proc, data = recycler}
}

// _recycler_destroy returns every cached block, and the recycler itself, to
// the backing allocator. Blocks still in use must be freed first.
_recycler_destroy :: proc(recycler: ^_Recycler) {
	if recycler == nil {
		return
	}
	backing := recycler.backing
	for size, &blocks in recycler.free {
		for block in blocks {
			mem.free_with_size(block, size, backing)
		}
		delete(blocks)
	}
	delete(recycler.free)
	free(recycler, backing)
}

_recycler_proc :: proc(allocator_data: rawptr, mode: mem.Allocator_Mode, size, alignment: int, old_memory: rawptr, old_size: int, location := #caller_location) -> ([]byte, mem.Allocator_Error) {
	recycler := (^_Recycler)(allocator_data)
	backing := recycler.backing
	switch mode {
	case .Alloc, .Alloc_Non_Zeroed:
		if size > 0 {
			if blocks, found := &recycler.free[size]; found {
				for i := len(blocks)-1; i >= 0; i -= 1 {
					block := blocks[i]
					if uintptr(block) % uintptr(max(alignment, 1)) != 0 {
						continue
					}
					unordered_remove(blocks, i)
					if mode == .Alloc {
						mem.zero(block, size)
					}
					return ([^]byte)(block)[:size], nil
				}
			}
		}
		return backing.procedure(backing.data, mode, size, alignment, old_memory, old_size, location)
	case .Free:
		// A free without its size, as from free(), cannot be filed.
		if old_memory == nil || old_size <= 0 {
			return backing.procedure(backing.data, mode, size, alignment, old_memory, old_size, location)
		}
		_, blocks, inserted, map_err := map_entry(&recycler.free, old_size)
		if map_err == nil {
			if inserted {
				blocks^ = make([dynamic]rawptr, backing)
			}
			if _, append_err := append(blocks, old_memory); append_err == nil {
				return nil, nil
			}
		}
		return backing.procedure(backing.data, mode, size, alignment, old_memory, old_size, location)
	case .Resize:
		return mem.default_resize_bytes_align(([^]byte)(old_memory)[:old_size], size, alignment, _recycler_allocator(recycler), location)
	case .Resize_Non_Zeroed:
		return mem.default_resize_bytes_align_non_zeroed(([^]byte)(old_memory)[:old_size], size, alignment, _recycler_allocator(recycler), location)
	case .Query_Features:
		set := (^mem.Allocator_Mode_Set)(old_memory)
		if set != nil {
			set^ = {.Alloc, .Alloc_Non_Zeroed, .Free, .Resize, .Resize_Non_Zeroed, .Query_Features}
		}
		return nil, nil
	case .Free_All, .Query_Info:
	}
	return nil, .Mode_Not_Implemented
}
