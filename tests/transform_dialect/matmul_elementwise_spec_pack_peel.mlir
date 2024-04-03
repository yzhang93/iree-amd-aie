// RUN: iree-opt --iree-transform-dialect-interpreter %s | FileCheck %s

// This script shows an example lowering matmul for AIE device.
// In this strategy, we use pack operations for data movement from L3 to L2, and L2 to L1.
// In order to keep initialization in L1, the first iteration of scf.for loop is peeled.

func.func @matmul_elementwise_i32() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x512xi8>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x1024xi8>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xi32>>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xi8>> -> tensor<1024x512xi8>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xi8>> -> tensor<512x1024xi8>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xi32>> -> tensor<1024x1024xi32>
  %7 = tensor.empty() : tensor<1024x1024xi32>
  %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %9 = linalg.matmul ins(%4, %5 : tensor<1024x512xi8>, tensor<512x1024xi8>) outs(%8 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<1024x1024xi32>, tensor<1024x1024xi32>) outs(%7 : tensor<1024x1024xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %11 = arith.addi %in, %in_0 : i32
    linalg.yield %11 : i32
  } -> tensor<1024x1024xi32>
  flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xi32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.read_only}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul", "linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill, %matmul, %generic = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // First level tile the elementwise op to forall.
    %first_level_tiled_element, %outer_forall =
      transform.structured.tile_using_forall %generic tile_sizes [64, 64]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse matmul operation into the forall loop.
    %first_level_tiled_matmul, %0 = transform.structured.fuse_into_containing_op %matmul into %outer_forall :
        (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the forall loop.
    %fused_fill, %1 = transform.structured.fuse_into_containing_op %fill into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level pack the matmul.
    %first_level_tiled_transposed_l2_packed_matmul = transform.structured.pack %first_level_tiled_matmul packed_sizes = [64, 64, 32]
      : (!transform.any_op) -> (!transform.any_op)

    %lhs_l2_pack = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[0] : (!transform.any_op) -> (!transform.any_op)

    %rhs_transposed_l2_pack_op = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[1] : (!transform.any_op) -> (!transform.any_op)
    %first_level_tiled_l2_packed_matmul, %rhs_l2_pack, %rhs_unpack =
      transform.structured.pack_transpose %rhs_transposed_l2_pack_op with_compute_op(%first_level_tiled_transposed_l2_packed_matmul)
      outer_perm = [0, 1] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // First level pack the elementwise op.
    %first_level_tiled_packed_element = transform.structured.pack %first_level_tiled_element packed_sizes = [64, 64]
      : (!transform.any_op) -> (!transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Promote the fused fill to shared memory
    %result_l2 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
        {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !transform.any_op

    // Second level pack the matmul.
    %l1_packed = transform.structured.pack %first_level_tiled_l2_packed_matmul packed_sizes = [0, 0, 0, 4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    %l1_packed_lhs = transform.get_producer_of_operand %l1_packed[0]
      : (!transform.any_op) -> (!transform.any_op)
    %lhs_l1_packed_matmul, %lhs_l1_pack_op, %lhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_lhs with_compute_op(%l1_packed)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    %l1_packed_rhs = transform.get_producer_of_operand %lhs_l1_packed_matmul[1]
      : (!transform.any_op) -> (!transform.any_op)
    %operands_l1_packed_matmul, %rhs_l1_pack_op, %rhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_rhs with_compute_op(%lhs_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %l1_packed_output = transform.get_consumers_of_result %operands_l1_packed_matmul[0]
      : (!transform.any_op) -> (!transform.any_op)
    %l1_packed_matmul, %output_l1_pack_op, %output_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_output with_compute_op(%operands_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Promote the result to local memory
    %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %output_l1_pack_op
        {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // First level for loop.
    %first_level_tiled_reduction_matmul, %outer_for_loop =
      transform.structured.tile_using_for %l1_packed_matmul [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the pack operations in the outer for loop.
    %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %lhs_l1_pack_op into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %rhs_l1_pack_op into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_lhs_l2_pack, %4 = transform.structured.fuse_into_containing_op %lhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l2_pack, %5 = transform.structured.fuse_into_containing_op %rhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the lhs to shared memory
    %lhs_l2_pack_buffer, %lhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Promote the rhs to shared memory
    %rhs_l2_pack_buffer, %rhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Second level tile to forall with tile_sizes.
    %second_level_tiled_matmul, %inner_forall =
      transform.structured.tile_using_forall %first_level_tiled_reduction_matmul tile_sizes [0, 0, 0, 8, 4, 0]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the pack operations in inner forall loop.
    %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the LHS to local memory.
    %lhs_l1_pack_buffer, %lhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote the RHS to local memory.
    %rhs_l1_pack_buffer, %rhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack2
      {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

     // Hoist static alloc out of the loops
    %func = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.hoist_static_alloc %func : (!transform.any_op) -> ()

    // Peel the for loop
    %for_op = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.op<"scf.for">
    %peeled_outer_loop, %remainder = transform.loop.peel %for_op {peel_front = true}
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the fill operation to fuse
    %fill_op = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    // Get the consumers of the fill. It has to be a scf.forall op.
    %peeled_loop = transform.get_consumers_of_result %fill_op[0] : (!transform.any_op) -> (!transform.op<"scf.forall">)

    // Fuse the fill within the loop.
    %peel_fused, %13 = transform.structured.fuse_into_containing_op %fill_op into %peeled_loop : (!transform.any_op, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // --------------------------------------- For elementwise op ----------------------------------------------------//
    // Second level tile the elementwise op to forall.
    %second_level_tiled_element, %inner_forall_1 =
      transform.structured.tile_using_forall %first_level_tiled_packed_element tile_sizes [0, 0, 32, 32]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Second level pack the elementwise op.
    %second_level_tiled_packed_element = transform.structured.pack %second_level_tiled_element packed_sizes = [0, 0, 4, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %l1_element_output = transform.get_consumers_of_result %second_level_tiled_packed_element[0]
      : (!transform.any_op) -> (!transform.any_op)
    %l1_packed_element, %l1_element_pack_op, %l1_element_unpack_op =
      transform.structured.pack_transpose %l1_element_output with_compute_op(%second_level_tiled_packed_element)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Bufferize and drop HAL decriptor from memref ops.
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    //%14 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

