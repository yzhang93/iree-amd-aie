// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIELogicalObjFifoSplittingUtils.h"

#include <numeric>

#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "iree-amdaie-logicalobjfifo-splitting-utils"

namespace mlir::iree_compiler::AMDAIE {

/// Hardcoded the transposed dimensions of L2 target dma for now.
/// The values are based on the results from ConvertToDma with option as
/// transposed on target, e.g., dma size [1, 1, 32, 32] -> [1, 32, 1, 32].
const static SmallVector<size_t> transposedL2Dims = {0, 2, 1, 3};

/// Utility to create a new logical objectfifo.
static AMDAIE::LogicalObjectFifoFromMemrefOp createNewLogicalObjectFifo(
    IRRewriter &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp oldLogicalObjectFifo,
    ArrayRef<int64_t> newSizes) {
  OpBuilder::InsertionGuard guard(rewriter);
  Value oldAllocOp = oldLogicalObjectFifo.getMemref();
  auto oldMemRefType = cast<MemRefType>(oldAllocOp.getType());
  MemRefType newAllocType = MemRefType::get(
      newSizes, oldMemRefType.getElementType(), MemRefLayoutAttrInterface{},
      oldMemRefType.getMemorySpace());
  assert(oldAllocOp.getDefiningOp() && "expected a defining op for the value");
  rewriter.setInsertionPoint(oldAllocOp.getDefiningOp());
  auto newAllocOp =
      rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(), newAllocType);
  auto newDeallocOp =
      rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(), newAllocOp);
  newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
  auto type = cast<MemRefType>(newAllocOp.getType());
  // Create new logical objectfifo.
  rewriter.setInsertionPoint(oldLogicalObjectFifo);
  auto newLogicalObjectFifo =
      rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
          newAllocOp.getResult(), oldLogicalObjectFifo.getTiles());
  return newLogicalObjectFifo;
}

/// Utility to create a new logical objectfifo based on shape defined by
/// `newSizesOpFoldResultArr`.
static AMDAIE::LogicalObjectFifoFromMemrefOp createNewLogicalObjectFifo(
    IRRewriter &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp oldLogicalObjectFifo,
    ArrayRef<OpFoldResult> newSizesOpFoldResultArr) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<int64_t> newSizes = llvm::map_to_vector(
      newSizesOpFoldResultArr,
      [](OpFoldResult sizeVal) { return getConstantIndexOrAssert(sizeVal); });
  return createNewLogicalObjectFifo(rewriter, oldLogicalObjectFifo, newSizes);
}

/// Utility to verify that the split dimensions for L2 are contiguous.
static LogicalResult checkIsRangeFromZero(
    SmallVector<size_t> &splitDimsSetForL2) {
  for (auto &&[dim, splitDim] : llvm::enumerate(splitDimsSetForL2)) {
    if (splitDim != dim) return failure();
  }
  return success();
}

/// This utility helps to perform the computation of offsets for L3 source.
///
/// Example:
/// For L3 -> L2 DmaCpyNd :-
/// From offset (0,0) we are extracting one 4x4 memref.
///                _______
///               |. . . .|
///               |. . . .|
///               |. . . .|
///               |. . . .|
///               ---------
/// After split we will extract four 2x2 memrefs.
/// So, the corresponding offsets will be :-
/// 1. Offset (0,0) - extract 2x2 memref
///       ___
///      |. .|. .
///      |. .|. .
///      -----
///       . . . .
///       . . . .
/// 2. Offset (0,2) - extract 2x2 memref
///           ___
///       . .|. .|
///       . .|. .|
///          -----
///       . . . .
///       . . . .
/// 3. Offset (2,0) - extract 2x2 memref
///       . . . .
///       . . . .
///       ___
///      |. .|. .
///      |. .|. .
///      -----
/// 4. Offset (2,2) - extract 2x2 memref
///       . . . .
///       . . . .
///           ___
///       . .|. .|
///       . .|. .|
///          -----
static FailureOr<OpFoldResult> addToOffset(IRRewriter &rewriter,
                                           OpFoldResult oldL3Offset,
                                           int64_t offsetToAdd) {
  auto createAffineMap = [&](AffineExpr affineExpr,
                             int64_t offsetToAdd) -> AffineMap {
    AffineExpr newAffineExpr = affineExpr + offsetToAdd;
    return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {newAffineExpr},
                          rewriter.getContext());
  };
  OpFoldResult newL3AsSourceOffset;
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto l3SourceOffsetAttr = dyn_cast<Attribute>(oldL3Offset)) {
    int64_t l3SourceOffsetIntVal =
        cast<IntegerAttr>(l3SourceOffsetAttr).getInt();
    int64_t newOffset = l3SourceOffsetIntVal + offsetToAdd;
    newL3AsSourceOffset = rewriter.getIndexAttr(newOffset);
  } else {
    auto l3SourceOffsetVal = cast<Value>(oldL3Offset);
    if (auto blockArg = dyn_cast<BlockArgument>(l3SourceOffsetVal)) {
      Operation *ownerOfBlockArg = blockArg.getOwner()->getParentOp();
      rewriter.setInsertionPointToStart(blockArg.getOwner());
      AffineExpr affineExpr = rewriter.getAffineDimExpr(0);
      AffineMap newAffineMap = createAffineMap(affineExpr, offsetToAdd);
      newL3AsSourceOffset =
          rewriter
              .create<affine::AffineApplyOp>(ownerOfBlockArg->getLoc(),
                                             newAffineMap, l3SourceOffsetVal)
              .getResult();
    } else {
      Operation *defOpOfL3SourceOffset = l3SourceOffsetVal.getDefiningOp();
      Location loc = defOpOfL3SourceOffset->getLoc();
      rewriter.setInsertionPoint(defOpOfL3SourceOffset);
      if (auto applyOp = dyn_cast_if_present<affine::AffineApplyOp>(
              defOpOfL3SourceOffset)) {
        AffineExpr affineExpr = applyOp.getAffineMap().getResult(0);
        AffineMap newAffineMap = createAffineMap(affineExpr, offsetToAdd);
        newL3AsSourceOffset =
            rewriter
                .create<affine::AffineApplyOp>(loc, newAffineMap,
                                               applyOp.getMapOperands())
                .getResult();
      } else if (auto constantOffset = getConstantIntValue(l3SourceOffsetVal)) {
        int64_t newOffset = *constantOffset + offsetToAdd;
        newL3AsSourceOffset = rewriter.getIndexAttr(newOffset);
      } else {
        return failure();
      }
    }
  }
  return newL3AsSourceOffset;
}

/// Given a L2->L1 DmaCpyNd op, find the unique L3->L2 DmaCpyNd op.
static FailureOr<AMDAIE::DmaCpyNdOp> fetchL3ToL2DmaCpyNdOp(
    AMDAIE::DmaCpyNdOp l2ToL1DmaOp) {
  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOp.getSourceObjectFifo();
  SmallVector<AMDAIE::DmaCpyNdOp> l3ToL2DmaOps;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
  for (Operation *objFifoUserOp : sourceObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
        dmaOp.getTargetObjectFifo() == sourceObjectFifo) {
      l3ToL2DmaOps.push_back(dmaOp);
    }
  }
  if (l3ToL2DmaOps.size() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "no corresponding L3->L2 dma op found for "
                            << sourceObjectFifo << "\n");
    return failure();
  }
  if (l3ToL2DmaOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "found more than one L3->L2 dma ops for "
                            << sourceObjectFifo << "\n");
    return failure();
  }
  l3ToL2DmaOp = l3ToL2DmaOps[0];
  return l3ToL2DmaOp;
}

/// A struct utility to encapsulate all the data required to perform splitting
/// of logicalobjectfifos.
struct SplittingLogicalObjectFifoData {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  SmallVector<size_t> splitDimsForL2;
  SmallVector<size_t> nonSplitDimsForL2;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
};

/// Utility to check whether splitting of logicalobjectfifos can be performed.
/// If possible, it populates the struct `SplittingLogicalObjectFifoData` with
/// the data required to perform the actual splitting.
static LogicalResult checkWhetherSplitIsPossible(
    SplittingLogicalObjectFifoData &splittingLogicalObjectFifoData) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
      splittingLogicalObjectFifoData.l2ToL1DmaOps;

  if (l2ToL1DmaOps.size() == 0) return failure();

  SmallVector<OpFoldResult> baseSourceOffsets =
      l2ToL1DmaOps[0].getSourceMixedOffsets();
  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOps[0].getSourceObjectFifo();
  auto sourceAllocOp =
      sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();
  if (!sourceAllocOp) {
    LLVM_DEBUG(llvm::dbgs() << "expected alloc op as the defining op of "
                            << sourceObjectFifo << "\n");
    return failure();
  }

  // We will now capture those dimensions where L2 memory was split. The way we
  // do this is by checking all L2->L1 DmaOps' source offset and marking those
  // dimensions which are not equal to at least one of the source offsets.
  DenseSet<size_t> splitDimsSetForL2;
  SmallVector<size_t> splitDimsForL2;
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    if (l2ToL1DmaOps[i].getSourceObjectFifo() != sourceObjectFifo) {
      LLVM_DEBUG(llvm::dbgs()
                 << l2ToL1DmaOps[i] << " does not have " << sourceObjectFifo
                 << " as the source objectfifo\n");
      return failure();
    }
    SmallVector<OpFoldResult> sourceOffsets =
        l2ToL1DmaOps[i].getSourceMixedOffsets();
    for (unsigned j = 0, m = baseSourceOffsets.size(); j < m; j++) {
      if (baseSourceOffsets[j] != sourceOffsets[j] &&
          !splitDimsSetForL2.contains(j)) {
        splitDimsForL2.push_back(j);
        splitDimsSetForL2.insert(j);
      }
    }
  }
  std::sort(splitDimsForL2.begin(), splitDimsForL2.end());

  if (failed(checkIsRangeFromZero(splitDimsForL2))) {
    LLVM_DEBUG(llvm::dbgs() << "cannot split L2 logicalobjectfifo because of "
                               "non-contiguous split dimensions inferred\n");
    return failure();
  }

  // Fetch the L3 -> L2 Dma Op corresponding to the L2 buffer as target.
  FailureOr<AMDAIE::DmaCpyNdOp> maybeL3ToL2DmaOp =
      fetchL3ToL2DmaCpyNdOp(l2ToL1DmaOps[0]);
  if (failed(maybeL3ToL2DmaOp)) return failure();
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp = maybeL3ToL2DmaOp.value();

  SmallVector<OpFoldResult, 4> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<size_t> nonSplitDimsForL2(staticL2AsTargetSizes.size() -
                                        splitDimsForL2.size());
  std::iota(nonSplitDimsForL2.begin(), nonSplitDimsForL2.end(),
            splitDimsForL2.size());

  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    SmallVector<OpFoldResult, 6> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      std::optional<int64_t> constantVal =
          getConstantIntValue(staticL2AsSourceOffsets[splitDim]);
      if (!constantVal) {
        LLVM_DEBUG(llvm::dbgs()
                   << "found a non-constant value for source offset at dim "
                   << splitDim << " for " << l2ToL1DmaOp << "\n");
        return failure();
      }
      constantVal = getConstantIntValue(staticL2AsTargetSizes[nonSplitdim]);
      if (!constantVal) {
        LLVM_DEBUG(llvm::dbgs()
                   << "found a non-constant value for target size at dim "
                   << nonSplitdim << " for " << l3ToL2DmaOp << "\n");
        return failure();
      }
    }
  }
  splittingLogicalObjectFifoData.splitDimsForL2 = splitDimsForL2;
  splittingLogicalObjectFifoData.nonSplitDimsForL2 = nonSplitDimsForL2;
  splittingLogicalObjectFifoData.l3ToL2DmaOp = l3ToL2DmaOp;
  return success();
}

/// Utility to determine if the strides of a dma copy operation might describe
/// a transposition of dimensions. Here we are only considering static strides.
/// If any of the static strides are in non-decreasing order from right to left,
/// then this might be a transpose.
static FailureOr<bool> isMaybeTransposed(Location loc,
                                         ArrayRef<OpFoldResult> strides) {
  std::optional<SmallVector<int64_t>> maybeStrides =
      getConstantIntValues(strides);
  if (!maybeStrides) {
    emitError(loc) << "expected static L2 strides";
    return failure();
  }
  SmallVector<int64_t> staticStrides = maybeStrides.value();
  return !std::is_sorted(staticStrides.rbegin(), staticStrides.rend());
}

static FailureOr<bool> isDmaTransposedOnSourceSide(AMDAIE::DmaCpyNdOp dmaOp) {
  return isMaybeTransposed(dmaOp->getLoc(), dmaOp.getSourceMixedStrides());
}

static FailureOr<bool> isDmaTransposedOnTargetSide(AMDAIE::DmaCpyNdOp dmaOp) {
  return isMaybeTransposed(dmaOp->getLoc(), dmaOp.getTargetMixedStrides());
}

/// Given a vector of L2->L1 Dma ops' perform the splitting :-
/// 1. Check if the splitting can be performed. If it can't, bail out.
/// 2. For the split dimension inferred set offset = 0 and size as 1 for L2 and
///    L3.
/// 3. Now traverse each L2->L1 Dma op and perform the following :-
///    a) Create a new L2 AllocOp based on the updated size (step 2 above) and
///       create a logicalobjectfifo using the same.
///    b) Split L3->L2 Dma op.
///    c) Split L2->L1 Dma op.
/// 4. Delete old L2->L1, L3->L2 and corresponding AllocOps.
LogicalResult splitLogicalObjectFifoForElementwiseOp(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context) {
  SplittingLogicalObjectFifoData splittingLogicalObjectFifoData;
  splittingLogicalObjectFifoData.l2ToL1DmaOps = l2ToL1DmaOps;
  if (failed(checkWhetherSplitIsPossible(splittingLogicalObjectFifoData))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot perform splitting of logicalobjectfifos");
    return success();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<size_t> splitDimsForL2 =
      splittingLogicalObjectFifoData.splitDimsForL2;
  SmallVector<size_t> nonSplitDimsForL2 =
      splittingLogicalObjectFifoData.nonSplitDimsForL2;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp = splittingLogicalObjectFifoData.l3ToL2DmaOp;

  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOps[0].getSourceObjectFifo();
  auto sourceAllocOp =
      sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();

  DenseSet<Operation *> toBeErased;
  toBeErased.insert(l3ToL2DmaOp);
  toBeErased.insert(sourceAllocOp);
  toBeErased.insert(sourceObjectFifo);

  SmallVector<OpFoldResult> staticL2AsTargetOffsets =
      l3ToL2DmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult> staticL3AsSourceOffsets =
      l3ToL2DmaOp.getSourceMixedOffsets();
  SmallVector<OpFoldResult> staticL3AsSourceSizes =
      l3ToL2DmaOp.getSourceMixedSizes();

  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);

  FailureOr<bool> maybeTransposed = isDmaTransposedOnTargetSide(l3ToL2DmaOp);
  if (failed(maybeTransposed)) return failure();
  bool dmaTransposeOnTarget = maybeTransposed.value();

  if (!dmaTransposeOnTarget) {
    // Update split dimensions' offset/size for L2 as target and L3 as source.
    // We can afford to do this here because it's going to be the same for all
    // L3->L2 splits. Here we are setting offset = 0 and size = 1.
    for (size_t dim : splitDimsForL2) {
      staticL2AsTargetOffsets[dim] = zeroVal;
      staticL2AsTargetSizes[dim] = oneVal;
      staticL3AsSourceOffsets[dim] = zeroVal;
      staticL3AsSourceSizes[dim] = oneVal;
    }
  } else {
    // The L2 target side has transposed dimensions, while the L3 source side
    // data are continuous and don't have `nonSplitDim`. Then the L3 source
    // sizes need to be modified to match the new L2 target sizes.
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      staticL2AsTargetOffsets[transposedL2Dims[splitDim]] = zeroVal;
      staticL2AsTargetSizes[transposedL2Dims[splitDim]] = oneVal;
      staticL3AsSourceSizes[splitDim] =
          staticL2AsTargetSizes[transposedL2Dims[nonSplitdim]];
    }
  }

  // Traverse each L2->L1 DmaCpyNd op and split them.
  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    SmallVector<OpFoldResult> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult> staticL2AsSourceSizes =
        l2ToL1DmaOp.getSourceMixedSizes();

    // Now we'll create a new L2 buffer based on the new shape inferred earlier
    // via `staticL2AsTargetSizes`.
    LogicalObjectFifoFromMemrefOp oldL2ObjectFifo =
        l2ToL1DmaOp.getSourceObjectFifo();
    // If the dma transpose is on the source(target) side, then the L2
    // target(source) side has the sizes in order.
    SmallVector<OpFoldResult> newL2Sizes =
        dmaTransposeOnTarget ? staticL2AsSourceSizes : staticL2AsTargetSizes;
    AMDAIE::LogicalObjectFifoFromMemrefOp source =
        createNewLogicalObjectFifo(rewriter, oldL2ObjectFifo, newL2Sizes);

    // --------------------------------------------
    // ---------- L3 -> L2 splitting --------------
    // --------------------------------------------
    // Update L3 source offsets for non-split dimensions. Refer doc comment of
    // `addToOffset` for the computation rationale involved.
    SmallVector<OpFoldResult> staticL3AsSourceOffsets =
        l3ToL2DmaOp.getSourceMixedOffsets();
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      std::optional<int64_t> constantOffset =
          getConstantIntValue(staticL2AsSourceOffsets[splitDim]);
      if (!constantOffset) {
        return l2ToL1DmaOp->emitOpError()
               << "found a non-constant value for source offset at dim "
               << splitDim;
      }
      std::optional<int64_t> constantSize =
          getConstantIntValue(newL2Sizes[nonSplitdim]);
      if (!constantSize) {
        return l3ToL2DmaOp->emitOpError()
               << "found a non-constant value for target size at dim "
               << nonSplitdim;
      }
      int64_t offsetToAdd = constantOffset.value() * constantSize.value();

      // If the dma transpose is on the target side, L3 source side data are
      // continuous and don't have `nonSplitDim`.
      size_t dim = dmaTransposeOnTarget ? splitDim : nonSplitdim;
      FailureOr<OpFoldResult> newOffset =
          addToOffset(rewriter, staticL3AsSourceOffsets[dim], offsetToAdd);
      if (failed(newOffset)) {
        // TODO: Ideally we should be able to handle even +, -, *, /, etc.
        //       But handle this later (if at all!) as such cases might not
        //       arise.
        return l3ToL2DmaOp->emitOpError()
               << "Unhandled expression for source offset at dim "
               << nonSplitdim;
      }
      staticL3AsSourceOffsets[dim] = *newOffset;
    }

    // Create new L3 -> L2 Dma Op.
    rewriter.setInsertionPoint(l3ToL2DmaOp);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        l3ToL2DmaOp.getLoc(), source, llvm::ArrayRef(staticL2AsTargetOffsets),
        llvm::ArrayRef(staticL2AsTargetSizes),
        l3ToL2DmaOp.getTargetMixedStrides(), l3ToL2DmaOp.getSource(),
        llvm::ArrayRef(staticL3AsSourceOffsets),
        llvm::ArrayRef(staticL3AsSourceSizes),
        l3ToL2DmaOp.getSourceMixedStrides());

    // --------------------------------------------
    // ---------- L2 -> L1 splitting --------------
    // --------------------------------------------
    // Update split dimensions' offset/size for L2 as target . Here we are
    // setting offset = 0 and size = 1.
    for (unsigned dim : splitDimsForL2) {
      staticL2AsSourceOffsets[dim] = zeroVal;
      staticL2AsSourceSizes[dim] = oneVal;
    }

    // Create new L2 -> L1 Input DmaOp.
    rewriter.setInsertionPoint(l2ToL1DmaOp);
    auto newL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        l2ToL1DmaOp.getLoc(), l2ToL1DmaOp.getTarget(),
        l2ToL1DmaOp.getTargetMixedOffsets(), l2ToL1DmaOp.getTargetMixedSizes(),
        l2ToL1DmaOp.getTargetMixedStrides(), source,
        llvm::ArrayRef(staticL2AsSourceOffsets),
        llvm::ArrayRef(staticL2AsSourceSizes),
        l2ToL1DmaOp.getSourceMixedStrides());
    rewriter.replaceOp(l2ToL1DmaOp, newL2ToL1DmaOp);

    // Remove old dealloc.
    memref::DeallocOp oldDeallocOp;
    for (Operation *userOp : sourceAllocOp->getUsers()) {
      if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp))
        oldDeallocOp = deallocUser;
    }
    if (oldDeallocOp) toBeErased.insert(oldDeallocOp);
  }

  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  return success();
}

/// Utility to get the `DmaCpyNdOp` producers and consumers of a given
/// objectFifo op.
LogicalResult getDmaCpyNdOpProducersAndConsumers(
    AMDAIE::LogicalObjectFifoFromMemrefOp op,
    SmallVector<AMDAIE::DmaCpyNdOp> &producers,
    SmallVector<AMDAIE::DmaCpyNdOp> &consumers) {
  for (Operation *userOp : op->getUsers()) {
    if (auto stridedCopyOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp)) {
      if (dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              stridedCopyOp.getTarget().getDefiningOp()) == op) {
        producers.push_back(stridedCopyOp);
      } else if (dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                     stridedCopyOp.getSource().getDefiningOp()) == op) {
        consumers.push_back(stridedCopyOp);
      } else {
        return op.emitOpError()
               << "has non-consumer, non-producer doubly strided copy op user";
      }
    } else {
      return op.emitOpError() << "has non-doubly strided copy op user";
    }
  }
  return success();
}

using OffsetIndexAndNewOffsetT = std::tuple<std::optional<size_t>, int64_t>;

/// Utility to return the index of the offsets array that refers to newly
/// splitted objectFifo and the respective offset value. Note that there might
/// not be a dimension with `stride == sizeAfterSplit`, in which case an offset
/// index can't be returned and the correct offset is `0`.
FailureOr<OffsetIndexAndNewOffsetT> getOffsetIndexAndOffset(
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, size_t sizeAfterSplit,
    function_ref<InFlightDiagnostic()> emitError) {
  SmallVector<size_t> offsetIndices;
  for (auto iter : llvm::enumerate(llvm::zip(strides, offsets))) {
    std::optional<int64_t> maybeStride =
        getConstantIntValue(std::get<0>(iter.value()));
    std::optional<int64_t> maybeOffset =
        getConstantIntValue(std::get<1>(iter.value()));
    if (maybeStride.has_value() && maybeOffset.has_value() &&
        maybeStride.value() == sizeAfterSplit && maybeOffset.value() != 0) {
      offsetIndices.push_back(iter.index());
    }
  }
  if (offsetIndices.size() > 1)
    return emitError() << "multiple offset indices found";
  int64_t size{1};
  int64_t offset{0};
  std::optional<size_t> maybeOffsetIdx;
  if (offsetIndices.size() == 1) {
    size_t offsetIdx = offsetIndices[0];
    maybeOffsetIdx = offsetIdx;
    std::optional<int64_t> maybeSize = getConstantIntValue(sizes[offsetIdx]);
    std::optional<int64_t> maybeOffset =
        getConstantIntValue(offsets[offsetIdx]);
    if (!maybeSize || !maybeOffset) {
      return emitError()
             << "expected a static target offset and size on index: "
             << offsetIdx;
    }
    size = maybeSize.value();
    offset = maybeOffset.value();
  }
  if (size != 1) {
    return emitError() << "only a static size of 1 is currently "
                          "supported on the split index";
  }
  return OffsetIndexAndNewOffsetT{maybeOffsetIdx, offset};
}

/// Split a logical objectFifo on the provided split dimension with the
/// specified splitting factor.
LogicalResult splitLogicalObjectFifo(IRRewriter &rewriter,
                                     AMDAIE::LogicalObjectFifoFromMemrefOp op,
                                     size_t splitDim,
                                     std::optional<size_t> maybeSplitFactor) {
  SmallVector<int64_t> memrefShape =
      llvm::to_vector(op.getMemrefType().getShape());
  int64_t splitFactor = maybeSplitFactor.has_value() ? maybeSplitFactor.value()
                                                     : memrefShape[splitDim];
  assert(
      memrefShape[splitDim] % splitFactor == 0 &&
      "the target size for splitting is not divisible by the splitting factor");
  memrefShape[splitDim] /= splitFactor;

  // Create `splitFactor` number of objectFifo ops.
  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> newObjFifos;
  newObjFifos.reserve(splitFactor);
  for (int i = 0; i < splitFactor; i++) {
    newObjFifos.push_back(
        createNewLogicalObjectFifo(rewriter, op, memrefShape));
  }

  // Get the producers and consumers of the current objectFifoOp.
  SmallVector<AMDAIE::DmaCpyNdOp> producers;
  SmallVector<AMDAIE::DmaCpyNdOp> consumers;
  if (failed(getDmaCpyNdOpProducersAndConsumers(op, producers, consumers))) {
    return failure();
  }

  // The split offset is the
  int64_t sizeAfterSplit =
      std::accumulate(memrefShape.begin() + splitDim + 1, memrefShape.end(), 1,
                      std::multiplies<>());
  // Update the producer dma ops.
  for (AMDAIE::DmaCpyNdOp producer : producers) {
    SmallVector<OpFoldResult> targetOffsets = producer.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizes = producer.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStrides = producer.getTargetMixedStrides();
    std::optional<size_t> maybeOffsetIdx;
    int64_t targetOffset{0};
    FailureOr<OffsetIndexAndNewOffsetT> maybeOffsetIdxAndNewOffset =
        getOffsetIndexAndOffset(targetOffsets, targetSizes, targetStrides,
                                sizeAfterSplit,
                                [&]() { return producer.emitOpError(); });
    if (failed(maybeOffsetIdxAndNewOffset)) {
      return producer.emitOpError()
             << "failed to find an offset index and new offset";
    }
    std::tie(maybeOffsetIdx, targetOffset) = maybeOffsetIdxAndNewOffset.value();
    assert(targetOffset < newObjFifos.size() &&
           "the targetOffset should be smaller than the number of objectFifos");
    if (maybeOffsetIdx.has_value())
      targetOffsets[maybeOffsetIdx.value()] = rewriter.getIndexAttr(0);
    AMDAIE::LogicalObjectFifoFromMemrefOp newObjFifo =
        newObjFifos[targetOffset];
    rewriter.setInsertionPoint(producer);
    auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        producer.getLoc(), newObjFifo, targetOffsets, targetSizes,
        targetStrides, producer.getSource(), producer.getSourceMixedOffsets(),
        producer.getSourceMixedSizes(), producer.getSourceMixedStrides());
    rewriter.replaceOp(producer, newDmaOp);
  }

  // Update the consumer dma ops.
  for (AMDAIE::DmaCpyNdOp consumer : consumers) {
    SmallVector<OpFoldResult> sourceOffsets = consumer.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes = consumer.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStrides = consumer.getSourceMixedStrides();
    std::optional<size_t> maybeOffsetIdx;
    int64_t sourceOffset{0};
    FailureOr<OffsetIndexAndNewOffsetT> maybeOffsetIdxAndNewOffset =
        getOffsetIndexAndOffset(sourceOffsets, sourceSizes, sourceStrides,
                                sizeAfterSplit,
                                [&]() { return consumer.emitOpError(); });
    if (failed(maybeOffsetIdxAndNewOffset)) {
      return consumer.emitOpError()
             << "failed to find an offset index and offset";
    }
    std::tie(maybeOffsetIdx, sourceOffset) = maybeOffsetIdxAndNewOffset.value();
    assert(sourceOffset < newObjFifos.size() &&
           "the sourceOffset should be smaller than the number of objectFifos");
    if (maybeOffsetIdx.has_value())
      sourceOffsets[maybeOffsetIdx.value()] = rewriter.getIndexAttr(0);
    AMDAIE::LogicalObjectFifoFromMemrefOp newObjFifo =
        newObjFifos[sourceOffset];
    rewriter.setInsertionPoint(consumer);
    auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        consumer.getLoc(), consumer.getTarget(),
        consumer.getTargetMixedOffsets(), consumer.getTargetMixedSizes(),
        consumer.getTargetMixedStrides(), newObjFifo, sourceOffsets,
        sourceSizes, sourceStrides);
    rewriter.replaceOp(consumer, newDmaOp);
  }
  return success();
}

/// Split doubly strided operations on a source and target split dimension with
/// the provided split factor.
LogicalResult splitDoublyStridedOp(IRRewriter &rewriter,
                                   AMDAIE::DoublyStridedOpInterface op,
                                   size_t sourceSplitDim, size_t targetSplitDim,
                                   std::optional<size_t> maybeSplitFactor) {
  if (!op->use_empty())
    return op.emitOpError() << "can't be split because it has uses";
  SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
  SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
  assert(sourceSplitDim < sourceOffsets.size() &&
         "the dimension to be split on should be smaller than the number of "
         "source dimensions");
  assert(targetSplitDim < targetOffsets.size() &&
         "the dimension to be split on should be smaller than the number of "
         "target dimensions");
  std::optional<int64_t> sourceSize =
      getConstantIntValue(sourceSizes[sourceSplitDim]);
  std::optional<int64_t> targetSize =
      getConstantIntValue(targetSizes[targetSplitDim]);
  if (!sourceSize) {
    return op.emitOpError()
           << "does not have a static source size on dim: " << sourceSplitDim;
  }
  if (!targetSize) {
    return op.emitOpError()
           << "does not have a static target size on dim: " << targetSplitDim;
  }
  int64_t splitFactor = maybeSplitFactor.has_value()
                            ? maybeSplitFactor.value()
                            : std::gcd(sourceSize.value(), targetSize.value());
  if (sourceSize.value() % splitFactor != 0 ||
      targetSize.value() % splitFactor != 0) {
    return op.emitOpError() << "the target or source size is not divisible by "
                               "the provided splitting factor: "
                            << splitFactor;
  }
  int64_t newSourceSize = sourceSize.value() / splitFactor;
  int64_t newTargetSize = targetSize.value() / splitFactor;
  sourceSizes[sourceSplitDim] = rewriter.getIndexAttr(newSourceSize);
  targetSizes[targetSplitDim] = rewriter.getIndexAttr(newTargetSize);
  rewriter.setInsertionPoint(op);
  for (int i = 0; i < splitFactor; ++i) {
    FailureOr<OpFoldResult> newSourceOffset = addToOffset(
        rewriter, sourceOffsets[sourceSplitDim], newSourceSize);  // i *
    FailureOr<OpFoldResult> newTargetOffset = addToOffset(
        rewriter, targetOffsets[targetSplitDim], newTargetSize);  // i *
    if (failed(newSourceOffset))
      return op.emitOpError() << "could not create a new source offset";
    if (failed(newTargetOffset))
      return op.emitOpError() << "could not create a new target offset";
    op.createDoublyStridedOp(rewriter, targetOffsets, targetSizes,
                             targetStrides, sourceOffsets, sourceSizes,
                             sourceStrides);
    sourceOffsets[sourceSplitDim] = newSourceOffset.value();
    targetOffsets[targetSplitDim] = newTargetOffset.value();
  }
  rewriter.eraseOp(op);
  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE