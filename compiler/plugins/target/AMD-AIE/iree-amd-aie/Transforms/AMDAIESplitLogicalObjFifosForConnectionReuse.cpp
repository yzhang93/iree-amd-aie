// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIELogicalObjFifoSplittingUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
// #include "llvm/Support/Debug.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos-for-connection-reuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESplitLogicalObjFifosForConnectionReusePass
    : public impl::AMDAIESplitLogicalObjFifosForConnectionReuseBase<
          AMDAIESplitLogicalObjFifosForConnectionReusePass> {
 public:
  using AMDAIESplitLogicalObjFifosForConnectionReuseBase::
      AMDAIESplitLogicalObjFifosForConnectionReuseBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosForConnectionReusePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  //   SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
  //       fetchDmaCpyNdOpsToSplitOrCombine(moduleOp);
  //
  //   if (failed(splitLogicalObjectFifos(rewriter, l2ToL1DmaOps, context))) {
  //     LLVM_DEBUG(llvm::dbgs()
  //                << "Failed to perform splitting of logicalobjectfifos");
  //     return signalPassFailure();
  //   }

  // Walk and collect all L3 to L2 Dma ops.
  SmallVector<AMDAIE::DmaCpyNdOp> l3ToL2DmaOps;
  WalkResult res = moduleOp->walk([&](AMDAIE::DmaCpyNdOp op) {
    std::optional<uint8_t> sourceMemSpace = op.getSourceMemorySpaceAsUInt();
    std::optional<uint8_t> targetMemSpace = op.getTargetMemorySpaceAsUInt();
    if (!sourceMemSpace || !targetMemSpace) {
      op.emitOpError() << "expected a source and target memory space";
      return WalkResult::interrupt();
    }
    if ((sourceMemSpace.value() == 1 && targetMemSpace.value() == 0) ||
        (sourceMemSpace.value() == 0 && targetMemSpace.value() == 1)) {
      l3ToL2DmaOps.push_back(op);
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  for (AMDAIE::DmaCpyNdOp dmaOp : l3ToL2DmaOps) {
    if (failed(splitDoublyStridedOp(rewriter, dmaOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of doubly strided op");
      return signalPassFailure();
    }
  }

  // Walk and split input and output objectfifos in L2 memory space.
  res = moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    if (op.getMemorySpaceAsUInt() != 1) return WalkResult::skip();
    if (failed(splitObjFifo(rewriter, op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosForConnectionReusePass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosForConnectionReusePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
