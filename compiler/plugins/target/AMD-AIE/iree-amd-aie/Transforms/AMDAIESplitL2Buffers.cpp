// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-l2-buffers"

namespace mlir::iree_compiler::AMDAIE {

namespace {

bool isElementwiseConsumerOfMatmul(linalg::LinalgOp linalgOp) {
  if (!isa<linalg::MatmulOp>(linalgOp) && !isMatmul(linalgOp)) {
    return false;
  }
  for (Operation *userOp : linalgOp->getUsers()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(userOp);
    if (linalgUser && isElementwise(linalgUser)) {
      return true;
    }
  }
  return false;
}

class AMDAIESplitL2BuffersPass
    : public impl::AMDAIESplitL2BuffersBase<AMDAIESplitL2BuffersPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  AMDAIESplitL2BuffersPass() = default;
  AMDAIESplitL2BuffersPass(const AMDAIESplitL2BuffersPass &pass){};
  void runOnOperation() override;
};

void AMDAIESplitL2BuffersPass::runOnOperation() {
  Operation *parentOp = getOperation();
  //  MLIRContext *context = &getContext();
  //  IRRewriter rewriter(context);
  //
  memref::AllocOp l2Alloc;
  IREE::LinalgExt::PackOp l2ToL1Pack;
  parentOp->walk([&](IREE::LinalgExt::PackOp packOp) {
    // Only get pack ops from L2 to L1 memory space
    Value output = packOp.getOutput();
    Operation *dstOp = output.getDefiningOp();

    uint32_t dstMemspace =
        cast<MemRefType>(output.getType()).getMemorySpaceAsInt();
    if (dstMemspace != 2) return WalkResult::advance();

    for (Operation *user : dstOp->getUsers()) {
      if (!isa<linalg::GenericOp>(user)) continue;
      if (!isElementwise(cast<linalg::LinalgOp>(user))) continue;

      // Now check the input of the pack op
      Value input = packOp.getInput();
      Operation *srcOp = input.getDefiningOp();
      if (!isa_and_present<memref::SubViewOp>(srcOp)) continue;

      auto subviewOp = cast<memref::SubViewOp>(user);
      Value srcSubview = subviewOp.getSource();
      Operation *origOp = srcSubview.getDefiningOp();
      uint32_t srcSubviewMem =
          cast<MemRefType>(srcSubview.getType()).getMemorySpaceAsInt();

      if (!isa<memref::AllocOp>(origOp) || srcSubviewMem != 1) continue;
      l2ToL1Pack = packOp;
      l2Alloc = cast<memref::AllocOp>(origOp);
    }
    return WalkResult::advance();
  });
  llvm::outs() << l2ToL1Pack << "\n";
  llvm::outs() << l2Alloc << "\n";
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitL2BuffersPass() {
  return std::make_unique<AMDAIESplitL2BuffersPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
