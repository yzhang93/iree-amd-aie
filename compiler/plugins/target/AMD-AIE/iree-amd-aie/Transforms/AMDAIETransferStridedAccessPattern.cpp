// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file composes more complex strided DMA ops by iteratively:
// 1. Combining ops in the same block.
// 2. Subsuming loop iterations into the strided access pattern.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-transfer-strided-access-pattern"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Copy a vector except for certain dim positions.
static SmallVector<OpFoldResult> copyExcludeDims(
    SmallVector<OpFoldResult> origVals, DenseSet<size_t> excludeDims) {
  if (excludeDims.size() == 0) return origVals;
  SmallVector<OpFoldResult> results;
  for (size_t i = 0; i < origVals.size(); i++) {
    if (!excludeDims.contains(i)) {
      results.push_back(origVals[i]);
    }
  }
  return results;
};

// Two dimensions (i and innermost) from the dma addressing can be combined if
// the conditions are satisfied: 1) stride[i] = innermost_stride *
// innermost_size; 2) offset[i] = 0.
static bool isL3AddressingCombinable(SmallVector<OpFoldResult> &dmaOffsets,
                                     SmallVector<OpFoldResult> &dmaSizes,
                                     SmallVector<OpFoldResult> &dmaStrides,
                                     size_t &dimForCombine) {
  // Offsets could be dynamic but sizes and strides should be static.
  std::optional<SmallVector<int64_t>> maybeSizes =
      getConstantIntValues(dmaSizes);
  std::optional<SmallVector<int64_t>> maybeStrides =
      getConstantIntValues(dmaStrides);
  if (!maybeSizes.has_value() || !maybeSizes.has_value()) {
    return false;
  }
  SmallVector<int64_t> sizeVals = maybeSizes.value();
  SmallVector<int64_t> strideVals = maybeStrides.value();

  // Get the index of the dim that can be potentially combined with the
  // innermost dim. If there is no such dim, return the last index of the
  // vector.
  auto getPos = [&](SmallVector<int64_t> values, int64_t target) {
    size_t i = 0;
    for (; i < values.size() - 1; i++) {
      if (values[i] == target) return i;
    }
    return i;
  };

  int64_t innerDimTotal = strideVals.back() * sizeVals.back();
  dimForCombine = getPos(strideVals, innerDimTotal);
  std::optional<int64_t> offsetAtPos =
      getConstantIntValue(dmaOffsets[dimForCombine]);

  if (dimForCombine >= (dmaSizes.size() - 1)) return false;
  if (!offsetAtPos.has_value() || offsetAtPos.value() != 0) return false;
  return true;
}

static bool isL2AddressingLinear(SmallVector<OpFoldResult> &dmaOffsets,
                                 SmallVector<OpFoldResult> &dmaSizes,
                                 SmallVector<OpFoldResult> &dmaStrides) {
  assert(dmaOffsets.size() == dmaSizes.size() &&
         dmaOffsets.size() == dmaStrides.size() &&
         "expected same number of source offsets and sizes");
  if (dmaOffsets.size() == 0) return true;
  if (dmaOffsets.size() != 1) return false;
  if (!isConstantIntValue(dmaOffsets[0], 0)) return false;
  if (!isConstantIntValue(dmaStrides[0], 1)) return false;
  return true;
}

// Check if all users of the connection op statisfy the conditions for
// optimization.
static bool checkConnectionUsers(AMDAIE::ConnectionOp connectionOp) {
  for (Operation *user : connectionOp->getUsers()) {
    // Check if L3 addressing is combinable.
    if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(user)) {
      // If the source is in L3, then the source addressing from NpuDmaCpyNdOp
      // and target addressing from NpuCircularDmaCpyNdOp can be potentially
      // changed.
      if (dmaOp.hasSourceAddressing() && dmaOp.hasTargetAddressing()) {
        return false;
      }
      if (!dmaOp.hasSourceAddressing() && !dmaOp.hasTargetAddressing()) {
        return false;
      }

      SmallVector<OpFoldResult> dmaOffsets;
      SmallVector<OpFoldResult> dmaSizes;
      SmallVector<OpFoldResult> dmaStrides;
      if (dmaOp.hasSourceAddressing()) {
        dmaOffsets = dmaOp.getSourceMixedOffsets();
        dmaSizes = dmaOp.getSourceMixedSizes();
        dmaStrides = dmaOp.getSourceMixedStrides();
      }
      if (dmaOp.hasTargetAddressing()) {
        dmaOffsets = dmaOp.getTargetMixedOffsets();
        dmaSizes = dmaOp.getTargetMixedSizes();
        dmaStrides = dmaOp.getTargetMixedStrides();
      }
      size_t dimForCombine = dmaSizes.size();
      if (!isL3AddressingCombinable(dmaOffsets, dmaSizes, dmaStrides,
                                    dimForCombine))
        return false;
    }
    // Check if L2 addressing should be linear.
    if (auto circularDma = dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(user)) {
      if (circularDma.hasSourceAddressing() &&
          circularDma.hasTargetAddressing()) {
        return false;
      }

      SmallVector<OpFoldResult> circularOffsets;
      SmallVector<OpFoldResult> circularSizes;
      SmallVector<OpFoldResult> circularStrides;
      if (circularDma.hasSourceAddressing()) {
        circularOffsets = circularDma.getSourceMixedOffsets();
        circularSizes = circularDma.getSourceMixedSizes();
        circularStrides = circularDma.getSourceMixedStrides();
      }
      if (circularDma.hasTargetAddressing()) {
        circularOffsets = circularDma.getTargetMixedOffsets();
        circularSizes = circularDma.getTargetMixedSizes();
        circularStrides = circularDma.getTargetMixedStrides();
      }
      if (!isL2AddressingLinear(circularOffsets, circularSizes,
                                circularStrides))
        return false;
    }
  }
  return true;
}

class AMDAIETransferStridedAccessPatternPass
    : public impl::AMDAIETransferStridedAccessPatternBase<
          AMDAIETransferStridedAccessPatternPass> {
 public:
  AMDAIETransferStridedAccessPatternPass() = default;
  AMDAIETransferStridedAccessPatternPass(
      const AMDAIETransferStridedAccessPatternPass &pass){};
  void runOnOperation() override;
};

void AMDAIETransferStridedAccessPatternPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);

  // Walk the NpuDmaCpyNdOp ops and get the defining connections between L2 and
  // L3 objectFifos. Then go through all users of each connection op and check
  // if there is optimization opportunity to transfer strided access pattern
  // from L3 to L2 side.
  DenseSet<AMDAIE::ConnectionOp> connectionOps;
  WalkResult walkRes = parentOp->walk([&](NpuDmaCpyNdOp dmaOp) {
    auto connectionOp = dmaOp.getConnectionOp();
    if (!connectionOp) {
      dmaOp.emitOpError() << "no connection op is found";
      return WalkResult::interrupt();
    }
    if (connectionOps.contains(connectionOp)) {
      return WalkResult::advance();
    }
    if (checkConnectionUsers(connectionOp)) {
      connectionOps.insert(connectionOp);
    }
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted()) return signalPassFailure();

  // Walk through all users of each connection op and change the dma addressing
  // from NpuDmaCpyNdOp and NpuCircularDmaCpyNdOp at the same time. Currently, a
  // connection op can have multiple NpuDmaCpyNdOp users (with different
  // offsets) but only one NpuCircularDmaCpyNdOp user.
  for (auto connectionOp : connectionOps) {
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNpuDmaUserOp =
        connectionOp.getNpuCircularDmaCpyNdUser();
    if (failed(maybeNpuDmaUserOp)) {
      connectionOp.emitOpError() << "failed to get circular NPU DMA op user";
      return signalPassFailure();
    }
    AMDAIE::NpuCircularDmaCpyNdOp circularDma = maybeNpuDmaUserOp.value();

    SmallVector<OpFoldResult> srcCircularOffsets =
        circularDma.getSourceMixedOffsets();
    SmallVector<OpFoldResult> srcCircularSizes =
        circularDma.getSourceMixedSizes();
    SmallVector<OpFoldResult> srcCircularStrides =
        circularDma.getSourceMixedStrides();
    SmallVector<OpFoldResult> tgtCircularOffsets =
        circularDma.getTargetMixedOffsets();
    SmallVector<OpFoldResult> tgtCircularSizes =
        circularDma.getTargetMixedSizes();
    SmallVector<OpFoldResult> tgtCircularStrides =
        circularDma.getTargetMixedStrides();

    auto getStrides = [&](SmallVector<int64_t> values) {
      SmallVector<OpFoldResult> res = {getAsIndexOpFoldResult(ctx, 1)};
      int64_t initial = values.back();
      for (size_t i = values.size() - 2; i > 0; i--) {
        initial *= values[i];
        res.push_back(getAsIndexOpFoldResult(ctx, initial));
      }
      return llvm::to_vector(llvm::reverse(res));
    };

    for (Operation *user : connectionOp->getUsers()) {
      if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(user)) {
        SmallVector<OpFoldResult> srcOffsets = dmaOp.getSourceMixedOffsets();
        SmallVector<OpFoldResult> srcSizes = dmaOp.getSourceMixedSizes();
        SmallVector<OpFoldResult> srcStrides = dmaOp.getSourceMixedStrides();
        SmallVector<OpFoldResult> tgtOffsets = dmaOp.getTargetMixedOffsets();
        SmallVector<OpFoldResult> tgtSizes = dmaOp.getTargetMixedSizes();
        SmallVector<OpFoldResult> tgtStrides = dmaOp.getTargetMixedStrides();

        // Generate new L3 source addressing, and new L2 target addressing.
        if (dmaOp.getSourceMemorySpaceAsUInt() == 0) {
          SmallVector<OpFoldResult> l3OrigSizes = srcSizes;
          SmallVector<OpFoldResult> l3OrigStrides = srcStrides;

          size_t dimForCombine = srcSizes.size();
          if (!isL3AddressingCombinable(srcOffsets, srcSizes, srcStrides,
                                        dimForCombine)) {
            return signalPassFailure();
          }

          // Generate new offsets and strides.
          DenseSet<size_t> excludeDims = {dimForCombine};
          srcOffsets = copyExcludeDims(srcOffsets, excludeDims);
          srcStrides = copyExcludeDims(srcStrides, excludeDims);

          // Generate new sizes, the innermost size is combined.
          std::optional<SmallVector<int64_t>> maybeSizes =
              getConstantIntValues(l3OrigSizes);
          std::optional<SmallVector<int64_t>> maybeStrides =
              getConstantIntValues(l3OrigStrides);
          if (!maybeSizes.has_value() || !maybeSizes.has_value()) {
            return signalPassFailure();
          }
          SmallVector<int64_t> sizeVals = maybeSizes.value();
          SmallVector<int64_t> strideVals = maybeStrides.value();

          int64_t innerDimTotal = strideVals.back() * sizeVals.back();
          int64_t newInnerSize = sizeVals[dimForCombine] * innerDimTotal;

          size_t lastIndex = l3OrigSizes.size() - 1;
          excludeDims.insert(lastIndex);
          srcSizes = copyExcludeDims(srcSizes, excludeDims);
          srcSizes.push_back(getAsIndexOpFoldResult(ctx, newInnerSize));

          SmallVector<OpFoldResult> newCircularOffsets(
              l3OrigSizes.size(), rewriter.getIndexAttr(0));
          tgtCircularOffsets = newCircularOffsets;
          tgtCircularSizes = copyExcludeDims(l3OrigSizes, excludeDims);
          tgtCircularSizes.push_back(
              getAsIndexOpFoldResult(ctx, sizeVals[dimForCombine]));
          tgtCircularSizes.push_back(
              getAsIndexOpFoldResult(ctx, innerDimTotal));
          tgtCircularStrides = getStrides(sizeVals);
          tgtCircularStrides.insert(
              tgtCircularStrides.begin() + dimForCombine,
              getAsIndexOpFoldResult(ctx, strideVals[dimForCombine]));
        }

        // Generate new L3 target addressing, and new L2 source addressing.
        if (dmaOp.getTargetMemorySpaceAsUInt() == 0) {
          SmallVector<OpFoldResult> l3OrigSizes = tgtSizes;
          SmallVector<OpFoldResult> l3OrigStrides = tgtStrides;

          size_t dimForCombine = tgtSizes.size();
          if (!isL3AddressingCombinable(tgtOffsets, tgtSizes, tgtStrides,
                                        dimForCombine)) {
            return signalPassFailure();
          }

          // Generate new offsets and strides.
          DenseSet<size_t> excludeDims = {dimForCombine};
          tgtOffsets = copyExcludeDims(tgtOffsets, excludeDims);
          tgtStrides = copyExcludeDims(tgtStrides, excludeDims);

          // Generate new sizes, the innermost size is combined.
          std::optional<SmallVector<int64_t>> maybeSizes =
              getConstantIntValues(l3OrigSizes);
          std::optional<SmallVector<int64_t>> maybeStrides =
              getConstantIntValues(l3OrigStrides);
          if (!maybeSizes.has_value() || !maybeSizes.has_value()) {
            return signalPassFailure();
          }
          SmallVector<int64_t> sizeVals = maybeSizes.value();
          SmallVector<int64_t> strideVals = maybeStrides.value();

          int64_t innerDimTotal = strideVals.back() * sizeVals.back();
          int64_t newInnerSize = sizeVals[dimForCombine] * innerDimTotal;

          size_t lastIndex = l3OrigSizes.size() - 1;
          excludeDims.insert(lastIndex);
          tgtSizes = copyExcludeDims(tgtSizes, excludeDims);
          tgtSizes.push_back(getAsIndexOpFoldResult(ctx, newInnerSize));

          SmallVector<OpFoldResult> newCircularOffsets(
              l3OrigSizes.size(), rewriter.getIndexAttr(0));
          srcCircularOffsets = newCircularOffsets;
          srcCircularSizes = copyExcludeDims(l3OrigSizes, excludeDims);
          srcCircularSizes.push_back(
              getAsIndexOpFoldResult(ctx, sizeVals[dimForCombine]));
          srcCircularSizes.push_back(
              getAsIndexOpFoldResult(ctx, innerDimTotal));
          srcCircularStrides = getStrides(sizeVals);
          srcCircularStrides.insert(
              srcCircularStrides.begin() + dimForCombine,
              getAsIndexOpFoldResult(ctx, strideVals[dimForCombine]));
        }

        // Replace the npu.dma_cpy_nd with the combined access pattern.
        rewriter.setInsertionPoint(dmaOp);
        dmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            dmaOp, dmaOp.getConnection(), dmaOp.getTarget(), tgtOffsets,
            tgtSizes, tgtStrides, dmaOp.getTargetBdId(), dmaOp.getSource(),
            srcOffsets, srcSizes, srcStrides, dmaOp.getSourceBdId());
      }
    }

    // Replace the npu.circular_dma_cpy_nd with the new access pattern.
    rewriter.setInsertionPoint(circularDma);
    circularDma = rewriter.replaceOpWithNewOp<AMDAIE::NpuCircularDmaCpyNdOp>(
        circularDma, circularDma.getConnection(), tgtCircularOffsets,
        tgtCircularSizes, tgtCircularStrides, srcCircularOffsets,
        srcCircularSizes, srcCircularStrides);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETransferStridedAccessPatternPass() {
  return std::make_unique<AMDAIETransferStridedAccessPatternPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
