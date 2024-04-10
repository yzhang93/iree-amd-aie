// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-lowering-strategy{use-pass-pipeline=pack-peel})))' %s | FileCheck %s

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [0, 0, 0, 8, 4, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
hal.executable private @matmul_pack_peel_i8_i32 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x2048_i8_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_i8_i32() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>> -> tensor<2048x2048xi8>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>> -> tensor<2048x2048xi8>
        %5 = tensor.empty() : tensor<2048x2048xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
        %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xi8>, tensor<2048x2048xi8>) outs(%6 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [0, 0, 0, 8, 4, 0]]>
// CHECK{LITERAL}: #config1 = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
// CHECK{LITERAL}: #packingConfig1 = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64], transposePackIndices = [], unpackEmpty = [], innerPerm = [], outerPerm = []}, {packedSizes = [0, 0, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
hal.executable private @matmul_elementwise_pack_peel_i8_i32 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_example_dispatch_0_matmul_1024x1024x512_i8xi8xi32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_1024x1024x512_i8xi8xi32() {
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
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
        %9 = linalg.matmul ins(%4, %5 : tensor<1024x512xi8>, tensor<512x1024xi8>) outs(%8 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
        // CHECK:  linalg.generic {{.*}} attrs = {lowering_config = #config1, packing_config = #packingConfig1}
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<1024x1024xi32>, tensor<1024x1024xi32>) outs(%7 : tensor<1024x1024xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
          %11 = arith.addi %in, %in_0 : i32
          linalg.yield %11 : i32
        } -> tensor<1024x1024xi32>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xi32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xi32>>
        return
      }
    }
  }
}
