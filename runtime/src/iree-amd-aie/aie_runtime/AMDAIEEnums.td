// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_AMDAIEENUMS
#define IREE_AIE_RUNTIME_AMDAIEENUMS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def AMDAIE_AMDAIEDevice: I32EnumAttr<"AMDAIEDevice",
  "Enum with target AMDAIE devices.",
  [
    I32EnumAttrCase<"xcvc1902", 1>,
    I32EnumAttrCase<"xcve2302", 2>,
    I32EnumAttrCase<"xcve2802", 3>,
    I32EnumAttrCase<"npu1", 4>,
    I32EnumAttrCase<"npu1_1col", 5>,
    I32EnumAttrCase<"npu1_2col", 6>,
    I32EnumAttrCase<"npu1_3col", 7>,
    I32EnumAttrCase<"npu1_4col", 8>
  ]>
{
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def DMAChannelDir: I32EnumAttr<"DMAChannelDir",
  "DMA Channel direction. Either stream to memory (`S2MM`) or memory to stream "
  "(`MM2S`)",
  [
    I32EnumAttrCase<"S2MM", 0>,
    I32EnumAttrCase<"MM2S", 1>
  ]> {
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

#endif // IREE_AIE_RUNTIME_AMDAIEENUMS
