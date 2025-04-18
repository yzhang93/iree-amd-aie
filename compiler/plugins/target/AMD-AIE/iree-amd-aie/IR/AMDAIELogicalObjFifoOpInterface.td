// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE
#define IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CopyOpInterface.td"

//===----------------------------------------------------------------------===//
// Defines the interface for logical objectFifo operations.
//===----------------------------------------------------------------------===//

def LogicalObjFifoOpInterface : OpInterface<"LogicalObjFifoOpInterface"> {
  let description = [{
    Interface for operations creating a logical objectFifo.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        A utility to return the allocation size of a logical objFifo in bytes.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getAllocationSizeInBytes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        MemRefType memrefType = getMemrefType();
        size_t allocationSizeInBits =
          memrefType.getNumElements() * memrefType.getElementTypeBitWidth();
        if (allocationSizeInBits % 8 != 0) return std::nullopt;
        return allocationSizeInBits / 8 * getDepth();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the consumer copy-like operations of the logical objFifo.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::CopyOpInterface>",
      /*methodName=*/"getCopyLikeConsumers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getCopyLikeConsumers(
          ::mlir::cast<LogicalObjFifoOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the producer copy-like operations of the logical objFifo.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::CopyOpInterface>",
      /*methodName=*/"getCopyLikeProducers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getCopyLikeProducers(
          ::mlir::cast<LogicalObjFifoOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the buffer depth of the logical objectFifo. (E.g. 1 == "
               "single buffer, 2 == double buffer)",
      /*retTy=*/"uint8_t",
      /*methodName=*/"getDepth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getDepth();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the memory space attribute.",
      /*retTy=*/"mlir::Attribute",
      /*methodName=*/"getMemorySpace",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getMemorySpace();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the memory space attribute as an integer.",
      /*retTy=*/"uint8_t",
      /*methodName=*/"getMemorySpaceAsUInt",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getMemorySpaceAsUInt();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the memref type.",
      /*retTy=*/"mlir::MemRefType",
      /*methodName=*/"getMemrefType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getMemrefType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the assigned tiles.",
      /*retTy=*/"::llvm::SmallVector<mlir::Value>",
      /*methodName=*/"getTiles",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTiles();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        A utility to replace this logical objectFifo operation with a new one with new tiles.
      }],
      /*retTy=*/"::llvm::FailureOr<::mlir::iree_compiler::AMDAIE::LogicalObjFifoOpInterface>",
      /*methodName=*/"replaceWithNewTiles",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter,
                    "::mlir::ValueRange":$tiles),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.replaceWithNewTiles(rewriter, tiles);
      }]
    >,
  ];
}

#endif // IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE
