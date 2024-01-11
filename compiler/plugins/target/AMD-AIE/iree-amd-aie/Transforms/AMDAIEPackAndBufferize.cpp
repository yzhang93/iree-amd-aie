// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pack-and-bufferize"

namespace mlir::iree_compiler::AMDAIE {

namespace {

//static LogicalResult applyBufferizeToAllocation(RewriterBase &rewriter,
//                                                Operation *op,
//                                                Attribute memorySpace) {
//  linalg::BufferizeToAllocationOptions options;
//  options.memcpyOp =
//      linalg::BufferizeToAllocationOptions::MemcpyOp::MaterializeInDestination;
//  options.allocOp = linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloc;
//  options.bufferizeDestinationOnly = true;
//  options.emitDealloc = true;
//
//  // Bufferize ops.
//  Value buffer =
//      linalg::bufferizeToAllocation(rewriter, options, op, memorySpace);
//  if (!buffer) {
//    LLVM_DEBUG(llvm::dbgs() << "----- failed to bufferize operation -----\n");
//    return failure();
//  }
//  return success();
//}


class AMDAIEPackAndBufferizePass
    : public AMDAIEPackAndBufferizeBase<AMDAIEPackAndBufferizePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEPackAndBufferizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
//    linalg::MatumlOp matmulOp;
//    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
//        [&](TilingInterface op) {
//          // Find the next matmul op if it does not have loops.
//          if (op.getLoopIteratorTypes().empty() || !isa<linalg::MatmulOp>(op))
//            return WalkResult::advance();
//          matmulOp = cast<linalg::MatmulOp>(op);
//          return WalkResult::interrupt();
//        });
//    if (!matmulOp) {
//      LLVM_DEBUG(llvm::dbgs() << "----- skip, no matmul op -----\n");
//      return;
//    }

    funcOp->walk([&](linalg::ContractionOpInterface op) {
      auto linalgOp = llvm::cast<linalg::LinalgOp>(op.getOperation());

      IRRewriter rewriter(context);
      SmallVector<OpFoldResult> packedSizes={rewriter.getI64IntegerAttr(16),
                                             rewriter.getI64IntegerAttr(64),
                                             rewriter.getI64IntegerAttr(64)};
      rewriter.setInsertionPoint(linalgOp);
      FailureOr<linalg::PackResult> maybeResult = linalg::pack(rewriter, linalgOp, packedSizes);
      if (failed(maybeResult))
        return signalPassFailure();
    });

//    for (auto operand : matmulOp->getOperands()) {
//      auto memorySpace = rewriter.getIntegerAttr(i64Type, 1);
//      rewriter.setInsertionPointAfter(operand.getDefiningOp());
//      if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
//                                            memorySpace))) {
//        funcOp->emitOpError("failed bufferizing to allocations");
//        return signalPassFailure();
//      }
//    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEPackAndBufferizePass() {
  return std::make_unique<AMDAIEPackAndBufferizePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE