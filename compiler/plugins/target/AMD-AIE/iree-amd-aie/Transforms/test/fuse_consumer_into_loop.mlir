// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{fuse-depth=1}))' %s | FileCheck %s --check-prefix=FORALL-DEPTH-1
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{fuse-depth=2}))' %s | FileCheck %s --check-prefix=FORALL-DEPTH-2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{fuse-depth=1 use-scf-for=true}))' %s | FileCheck %s --check-prefix=FOR

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @fuse_consumer_into_scfforall(%arg0: tensor<1x1x4x8x4x8xi8>, %arg1: tensor<1x1x4x4x8x8xi8>, %arg2: tensor<1x1x8x16x4x8xi32>, %arg3: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
    %0 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %arg3) -> (tensor<1024x1024xi32>) {
      %1 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %arg2) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      }
      %2 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %1) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      }
      %3 = tensor.empty() : tensor<1x1x8x16x4x8xi32>
      %4 = tensor.empty() : tensor<64x64xi32>
      %5 = tensor.empty() : tensor<1x1x64x64xi32>
      %6 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x1x8x16x4x8xi32>, tensor<1x1x8x16x4x8xi32>) outs(%3 : tensor<1x1x8x16x4x8xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %7 = arith.addi %in, %in_1 : i32
        linalg.yield %7 : i32
      } -> tensor<1x1x8x16x4x8xi32>
      %unpack = tensor.unpack %6 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %5 : tensor<1x1x8x16x4x8xi32> -> tensor<1x1x64x64xi32>
      %unpack_0 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %4 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_0 into %arg6[%arg4, %arg5] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
      }
    }
    return %0 : tensor<1024x1024xi32>
  }
}
//      FORALL-DEPTH-1:   %[[FINAL:.*]] = scf.forall
// FORALL-DEPTH-1-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      FORALL-DEPTH-1:   {
//      FORALL-DEPTH-1:       %[[FIRST_LOOP:.*]] = scf.forall
//      FORALL-DEPTH-1:       {
//      FORALL-DEPTH-1:       }
//      FORALL-DEPTH-1:       %[[ELEM_OUT:.*]] = tensor.empty() : tensor<1x1x8x16x4x8xi32>
//      FORALL-DEPTH-1:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      FORALL-DEPTH-1:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
//      FORALL-DEPTH-1:       %[[SECOND_LOOP:.*]]:2 = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2) shared_outs(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]])
//      FORALL-DEPTH-1:       {
//      FORALL-DEPTH-1:            %[[MATMUL:.*]] = linalg.generic
//      FORALL-DEPTH-1:            %[[OPERAND1:.*]] = tensor.extract_slice %[[MATMUL]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-1:            %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OUT]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-1:            %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-1:            %[[FUSED_CONSUMER:.*]] = linalg.generic
// FORALL-DEPTH-1-SAME:                                         ins(%[[OPERAND1]], %[[OPERAND2]] :
// FORALL-DEPTH-1-SAME:                                         outs(%[[OPERAND3]] :
//      FORALL-DEPTH-1:                                    {
//      FORALL-DEPTH-1:                                         arith.addi  
//      FORALL-DEPTH-1:                                    }
//      FORALL-DEPTH-1:            scf.forall.in_parallel {
//      FORALL-DEPTH-1:                 tensor.parallel_insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-1:                 tensor.parallel_insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-1:            }
//      FORALL-DEPTH-1:        }
//      FORALL-DEPTH-1:        %[[FIRST_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[UNPACK_OUT]]
//      FORALL-DEPTH-1:        %[[SECOND_UNPACK:.*]] = tensor.unpack %[[FIRST_UNPACK]] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      FORALL-DEPTH-1:        scf.forall.in_parallel
//      FORALL-DEPTH-1:             tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      FORALL-DEPTH-1:        }
//      FORALL-DEPTH-1:   }
//      FORALL-DEPTH-1:   return %[[FINAL]]

//      FORALL-DEPTH-2: #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
//      FORALL-DEPTH-2: #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
//      FORALL-DEPTH-2:   %[[FINAL:.*]] = scf.forall
// FORALL-DEPTH-2-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      FORALL-DEPTH-2:   {
//      FORALL-DEPTH-2:       %[[FIRST_LOOP:.*]] = scf.forall
//      FORALL-DEPTH-2:       {
//      FORALL-DEPTH-2:       }
//      FORALL-DEPTH-2:       %[[ELEM_OUT:.*]] = tensor.empty() : tensor<1x1x8x16x4x8xi32>
//      FORALL-DEPTH-2:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      FORALL-DEPTH-2:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
//      FORALL-DEPTH-2:       %[[SECOND_LOOP:.*]]:3 = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2) shared_outs(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
//      FORALL-DEPTH-2:       {
//      FORALL-DEPTH-2:            %[[MATMUL:.*]] = linalg.generic
//      FORALL-DEPTH-2:            %[[FUSED_CONSUMER:.*]] = linalg.generic
//      FORALL-DEPTH-2:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//      FORALL-DEPTH-2:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV1]])
//      FORALL-DEPTH-2:            %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, 32, 32] [1, 1, 1, 1]
//      FORALL-DEPTH-2:            %[[TILED_UNPACK_SRC:.*]] = tensor.extract_slice %[[FUSED_CONSUMER]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-2:            %[[TILED_UNPACK:.*]] = tensor.unpack %[[TILED_UNPACK_SRC]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
//      FORALL-DEPTH-2:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//      FORALL-DEPTH-2:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV1]])
//      FORALL-DEPTH-2:            scf.forall.in_parallel {
//      FORALL-DEPTH-2:                 tensor.parallel_insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, 32, 32] [1, 1, 1, 1]
//      FORALL-DEPTH-2:                 tensor.parallel_insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-2:                 tensor.parallel_insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      FORALL-DEPTH-2:            }
//      FORALL-DEPTH-2:        }
//      FORALL-DEPTH-2:        %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      FORALL-DEPTH-2:        scf.forall.in_parallel
//      FORALL-DEPTH-2:             tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      FORALL-DEPTH-2:        }
//      FORALL-DEPTH-2:   }
//      FORALL-DEPTH-2:   return %[[FINAL]]

// -----

// FOR-LABEL: @fuse_consumer_into_scffor
func.func @fuse_consumer_into_scffor(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg3 = %c0 to %c64 step %c4 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
        %1 = scf.for %arg5 = %c0 to %c64 step %c4 iter_args(%arg6 = %arg4) -> (tensor<64x64xf32>) {
            %extracted_slice = tensor.extract_slice %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            scf.yield %inserted_slice : tensor<64x64xf32>
        }
        %2 = scf.for %arg5 = %c0 to %c64 step %c4 iter_args(%arg6 = %1) -> (tensor<64x64xf32>) {
            %extracted_slice = tensor.extract_slice %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            scf.yield %inserted_slice : tensor<64x64xf32>
        }
        %3 = tensor.empty() : tensor<64x64xf32>
        %4 = tensor.empty() : tensor<64x64xf32>
        %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %5 : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
}
//      FOR:   %[[FINAL:.*]] = scf.for %{{.*}} iter_args(%[[ITER_ARG_FINAL:.*]] = %{{.*}}) ->
//      FOR:   {
//      FOR:       %[[FIRST_LOOP:.*]] = scf.for
//      FOR:       {
//      FOR:       }
//      FOR:       %[[ELEM_OPERAND_2:.*]] = tensor.empty() : tensor<64x64xf32>
//      FOR:       %[[ELEM_OUT:.*]] = tensor.empty() : tensor<64x64xf32>
//      FOR:       %[[SECOND_LOOP:.*]]:2 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]])
//      FOR:       {
//      FOR:            %[[MATMUL:.*]] = linalg.matmul
//      FOR:            %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
//      FOR:            %[[OPERAND1:.*]] = tensor.extract_slice %[[YIELD_MATMUL]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
//      FOR:            %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OPERAND_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
//      FOR:            %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
//      FOR:            %[[FUSED_CONSUMER:.*]] = linalg.elemwise_binary
// FOR-SAME:                                        ins(%[[OPERAND1]], %[[OPERAND2]] :
// FOR-SAME:                                        outs(%[[OPERAND3]] :
//      FOR:            %[[YIELD_ELEM:.*]] = tensor.insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
//      FOR:            scf.yield %[[YIELD_MATMUL]], %[[YIELD_ELEM]]
//      FOR:       }
//      FOR:       scf.yield %[[SECOND_LOOP]]#1
//      FOR:   }
//      FOR:   return %[[FINAL]]
