#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0) -> (d0 * 8)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d2, d4, d5)>
module {
  func.func @matmul_elementwise_i32() {
    %c15 = arith.constant 15 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x4x4x8x8xi8, 2>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi8, 2>
    %alloc_1 = memref.alloc() : memref<1x1x32x64xi8, 1>
    %alloc_2 = memref.alloc() : memref<1x1x64x32xi8, 1>
    %alloc_3 = memref.alloc() : memref<1x1x8x16x4x8xi32, 2>
    %alloc_4 = memref.alloc() : memref<1x1x64x64xi32, 1>
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x512xi8>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x1024xi8>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xi32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xi8>> -> tensor<1024x512xi8>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xi8>> -> tensor<512x1024xi8>
    %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xi32>> -> tensor<1024x1024xi32>
    %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>> -> tensor<1024x1024xi32>
    %8 = scf.forall (%arg0, %arg1) in (16, 16) shared_outs(%arg2 = %7) -> (tensor<1024x1024xi32>) {
      %9 = affine.apply #map(%arg0)
      %10 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %4[%9, 0] [64, 512] [1, 1] : tensor<1024x512xi8> to tensor<64x512xi8>
      %extracted_slice_5 = tensor.extract_slice %5[0, %10] [512, 64] [1, 1] : tensor<512x1024xi8> to tensor<512x64xi8>
      %11 = tensor.empty() : tensor<1x1x64x64xi32>
      %12 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x1x64x64xi32, 1>
      %13 = bufferization.to_tensor %alloc_3 restrict writable : memref<1x1x8x16x4x8xi32, 2>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice[0, 0] [64, 32] [1, 1] : tensor<64x512xi8> to tensor<64x32xi8>
      %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x1x64x32xi8, 1>
      %pack = tensor.pack %extracted_slice_6 inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %14 : tensor<64x32xi8> -> tensor<1x1x64x32xi8>
      %extracted_slice_7 = tensor.extract_slice %extracted_slice_5[0, 0] [32, 64] [1, 1] : tensor<512x64xi8> to tensor<32x64xi8>
      %15 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x32x64xi8, 1>
      %pack_8 = tensor.pack %extracted_slice_7 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %15 : tensor<32x64xi8> -> tensor<1x1x32x64xi8>
      %16 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %13) -> (tensor<1x1x8x16x4x8xi32>) {
        %20 = affine.apply #map1(%arg4)
        %21 = affine.apply #map2(%arg3)
        %22 = affine.apply #map3(%arg3)
        %extracted_slice_18 = tensor.extract_slice %pack[0, 0, %22, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x32xi8> to tensor<1x1x32x32xi8>
        %23 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x8x4x8xi8, 2>
        %pack_19 = tensor.pack %extracted_slice_18 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %23 : tensor<1x1x32x32xi8> -> tensor<1x1x4x8x4x8xi8>
        %24 = affine.apply #map3(%arg4)
        %extracted_slice_20 = tensor.extract_slice %pack_8[0, 0, 0, %24] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x64xi8> to tensor<1x1x32x32xi8>
        %25 = bufferization.to_tensor %alloc restrict writable : memref<1x1x4x4x8x8xi8, 2>
        %pack_21 = tensor.pack %extracted_slice_20 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %25 : tensor<1x1x32x32xi8> -> tensor<1x1x4x4x8x8xi8>
        %extracted_slice_22 = tensor.extract_slice %arg5[0, 0, %20, %21, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %26 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_22 : tensor<1x1x4x8x4x8xi32>) -> tensor<1x1x4x8x4x8xi32>
        %27 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_19, %pack_21 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%26 : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_23: i8, %out: i32):
          %28 = arith.extsi %in : i8 to i32
          %29 = arith.extsi %in_23 : i8 to i32
          %30 = arith.muli %28, %29 : i32
          %31 = arith.addi %out, %30 : i32
          linalg.yield %31 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %27 into %arg5[0, 0, %20, %21, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %17 = scf.for %arg3 = %c1 to %c15 step %c1 iter_args(%arg4 = %16) -> (tensor<1x1x8x16x4x8xi32>) {
        %20 = affine.apply #map3(%arg3)
        %extracted_slice_18 = tensor.extract_slice %extracted_slice[0, %20] [64, 32] [1, 1] : tensor<64x512xi8> to tensor<64x32xi8>
        %21 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x1x64x32xi8, 1>
        %pack_19 = tensor.pack %extracted_slice_18 inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %21 : tensor<64x32xi8> -> tensor<1x1x64x32xi8>
        %extracted_slice_20 = tensor.extract_slice %extracted_slice_5[%20, 0] [32, 64] [1, 1] : tensor<512x64xi8> to tensor<32x64xi8>
        %22 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x32x64xi8, 1>
        %pack_21 = tensor.pack %extracted_slice_20 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %22 : tensor<32x64xi8> -> tensor<1x1x32x64xi8>
        %23 = scf.forall (%arg5, %arg6) in (2, 2) shared_outs(%arg7 = %arg4) -> (tensor<1x1x8x16x4x8xi32>) {
          %24 = affine.apply #map1(%arg6)
          %25 = affine.apply #map2(%arg5)
          %26 = affine.apply #map3(%arg5)
          %extracted_slice_22 = tensor.extract_slice %pack_19[0, 0, %26, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x32xi8> to tensor<1x1x32x32xi8>
          %27 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x8x4x8xi8, 2>
          %pack_23 = tensor.pack %extracted_slice_22 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %27 : tensor<1x1x32x32xi8> -> tensor<1x1x4x8x4x8xi8>
          %28 = affine.apply #map3(%arg6)
          %extracted_slice_24 = tensor.extract_slice %pack_21[0, 0, 0, %28] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x64xi8> to tensor<1x1x32x32xi8>
          %29 = bufferization.to_tensor %alloc restrict writable : memref<1x1x4x4x8x8xi8, 2>
          %pack_25 = tensor.pack %extracted_slice_24 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %29 : tensor<1x1x32x32xi8> -> tensor<1x1x4x4x8x8xi8>
          %extracted_slice_26 = tensor.extract_slice %arg7[0, 0, %24, %25, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
          %30 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_23, %pack_25 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice_26 : tensor<1x1x4x8x4x8xi32>) {
          ^bb0(%in: i8, %in_27: i8, %out: i32):
            %31 = arith.extsi %in : i8 to i32
            %32 = arith.extsi %in_27 : i8 to i32
            %33 = arith.muli %31, %32 : i32
            %34 = arith.addi %out, %33 : i32
            linalg.yield %34 : i32
          } -> tensor<1x1x4x8x4x8xi32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %30 into %arg7[0, 0, %24, %25, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        scf.yield %23 : tensor<1x1x8x16x4x8xi32>
      }
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [64, 32] [1, 1] : tensor<64x512xi8> to tensor<64x32xi8>
      %pack_10 = tensor.pack %extracted_slice_9 inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %14 : tensor<64x32xi8> -> tensor<1x1x64x32xi8>
      %extracted_slice_11 = tensor.extract_slice %extracted_slice_5[480, 0] [32, 64] [1, 1] : tensor<512x64xi8> to tensor<32x64xi8>
      %pack_12 = tensor.pack %extracted_slice_11 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %15 : tensor<32x64xi8> -> tensor<1x1x32x64xi8>
      %18 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %17) -> (tensor<1x1x8x16x4x8xi32>) {
        %20 = affine.apply #map1(%arg4)
        %21 = affine.apply #map2(%arg3)
        %22 = affine.apply #map3(%arg3)
        %extracted_slice_18 = tensor.extract_slice %pack_10[0, 0, %22, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x32xi8> to tensor<1x1x32x32xi8>
        %23 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x8x4x8xi8, 2>
        %pack_19 = tensor.pack %extracted_slice_18 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %23 : tensor<1x1x32x32xi8> -> tensor<1x1x4x8x4x8xi8>
        %24 = affine.apply #map3(%arg4)
        %extracted_slice_20 = tensor.extract_slice %pack_12[0, 0, 0, %24] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x64xi8> to tensor<1x1x32x32xi8>
        %25 = bufferization.to_tensor %alloc restrict writable : memref<1x1x4x4x8x8xi8, 2>
        %pack_21 = tensor.pack %extracted_slice_20 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %25 : tensor<1x1x32x32xi8> -> tensor<1x1x4x4x8x8xi8>
        %extracted_slice_22 = tensor.extract_slice %arg5[0, 0, %20, %21, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %26 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_19, %pack_21 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice_22 : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_23: i8, %out: i32):
          %27 = arith.extsi %in : i8 to i32
          %28 = arith.extsi %in_23 : i8 to i32
          %29 = arith.muli %27, %28 : i32
          %30 = arith.addi %out, %29 : i32
          linalg.yield %30 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %26 into %arg5[0, 0, %20, %21, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %unpack = tensor.unpack %18 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %12 : tensor<1x1x8x16x4x8xi32> -> tensor<1x1x64x64xi32>
      %extracted_slice_13 = tensor.extract_slice %6[%9, %10] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
      %extracted_slice_14 = tensor.extract_slice %arg2[%9, %10] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
      %pack_15 = tensor.pack %extracted_slice_13 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %11 : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
      %pack_16 = tensor.pack %extracted_slice_14 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %11 : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
      %19 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %pack_16) -> (tensor<1x1x64x64xi32>) {
        %20 = affine.apply #map3(%arg3)
        %21 = affine.apply #map3(%arg4)
        %extracted_slice_18 = tensor.extract_slice %unpack[0, 0, %20, %21] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x64xi32> to tensor<1x1x32x32xi32>
        %extracted_slice_19 = tensor.extract_slice %pack_15[0, 0, %20, %21] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x64xi32> to tensor<1x1x32x32xi32>
        %extracted_slice_20 = tensor.extract_slice %arg5[0, 0, %20, %21] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x64x64xi32> to tensor<1x1x32x32xi32>
        %22 = tensor.empty() : tensor<1x1x8x4x4x8xi32>
        %pack_21 = tensor.pack %extracted_slice_18 inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %22 : tensor<1x1x32x32xi32> -> tensor<1x1x8x4x4x8xi32>
        %pack_22 = tensor.pack %extracted_slice_19 inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %22 : tensor<1x1x32x32xi32> -> tensor<1x1x8x4x4x8xi32>
        %23 = tensor.empty() : tensor<1x1x4x8x4x8xi32>
        %pack_23 = tensor.pack %extracted_slice_20 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %23 : tensor<1x1x32x32xi32> -> tensor<1x1x4x8x4x8xi32>
        %24 = linalg.generic {indexing_maps = [#map7, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%pack_21, %pack_22 : tensor<1x1x8x4x4x8xi32>, tensor<1x1x8x4x4x8xi32>) outs(%pack_23 : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i32, %in_25: i32, %out: i32):
          %25 = arith.addi %in, %in_25 : i32
          linalg.yield %25 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        %unpack_24 = tensor.unpack %24 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %extracted_slice_20 : tensor<1x1x4x8x4x8xi32> -> tensor<1x1x32x32xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %unpack_24 into %arg5[0, 0, %20, %21] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x32xi32> into tensor<1x1x64x64xi32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %unpack_17 = tensor.unpack %19 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_14 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_17 into %arg2[%9, %10] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xi32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>>
    memref.dealloc %alloc_4 : memref<1x1x64x64xi32, 1>
    memref.dealloc %alloc_3 : memref<1x1x8x16x4x8xi32, 2>
    memref.dealloc %alloc_2 : memref<1x1x64x32xi8, 1>
    memref.dealloc %alloc_1 : memref<1x1x32x64xi8, 1>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi8, 2>
    memref.dealloc %alloc : memref<1x1x4x4x8x8xi8, 2>
    return
  }
}
