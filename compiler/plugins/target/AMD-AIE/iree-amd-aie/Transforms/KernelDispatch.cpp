// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"

namespace mlir::iree_compiler::AMDAIE {

using detail::findLargestFactor;

static SmallVector<int64_t> getPackedSize(linalg::LinalgOp linalgOp,
                                          const int packLevel, int m = 0,
                                          int n = 0, int k = 0) {
  // TODO (newling): consider emiting an error/warning if the default sizes are used as a
  // fallback.
  SmallVector<int64_t> defaultSizes;
  // TODO (nmeshram) : We should not need this and be able to fix the pack
  // config after we have padding support
  int minM = m ? findLargestFactor(m, 4) : 4;
  int minN = n ? findLargestFactor(n, 4) : 4;
  int minK = k ? findLargestFactor(k, 8) : 8;
  if (packLevel == 1) {
    defaultSizes = {{minM, minN, minK}};
  } else if (packLevel == 2) {
    defaultSizes = {{0, 0, 0, minM, minN, minK}};
  } else {
    linalgOp->emitError("invalid value of pack level.");
  }
  if (!isa<linalg::MatmulOp>(linalgOp)) {
    return defaultSizes;
  }

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto elTypeLhs = getElementType(linalgOp->getOperand(0));
  auto elTypeRhs = getElementType(linalgOp->getOperand(1));
  auto elTypeAcc = getElementType(linalgOp->getResult(0));

  auto maybeInstructionSize =
      getAIEMatmulInstructionSize(elTypeLhs, elTypeRhs, elTypeAcc);

  if (failed(maybeInstructionSize)) {
    return defaultSizes;
  }

  auto instructionSize = maybeInstructionSize.value();
  SmallVector<int64_t> packedSizes(3, 0);
  std::copy(instructionSize.begin(), instructionSize.end(),
            packedSizes.begin());
  if (packLevel == 2) {
    packedSizes.insert(packedSizes.begin(), {0, 0, 0});
  }
  return packedSizes;
}

static LogicalResult setRootConfigForPackPeelPipeline(func::FuncOp entryPointFn,
                                                      linalg::LinalgOp linalgOp,
                                                      AIEConfig cfg) {
  auto initType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  auto initShape = initType.getShape();
  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto lhsShape = lhsType.getShape();

  FailureOr<unsigned> maybeTilingScaleFactor =
      getTilingScaleFactor(initType.getElementType());
  if (failed(maybeTilingScaleFactor)) {
    return linalgOp.emitOpError("expected bitwidth 64/32/16/8");
  }
  unsigned tilingScaleFactor = maybeTilingScaleFactor.value();
  auto tileM0 = findLargestFactor((int)initShape[0], 32 * tilingScaleFactor);
  auto tileN0 = findLargestFactor((int)initShape[1], 32 * tilingScaleFactor);

  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  // Pack level => 1.
  auto packedK0 = findLargestFactor((int)lhsShape[1], 16 * tilingScaleFactor);
  SmallVector<int64_t> packedSizes = {tileM0, tileN0, packedK0};
  // Transpose B matrix from [K N n k] to [K N k n]
  SmallVector<int64_t> transposePackIndices = {1};
  // There is no corresponding unpack for the specified pack operation
  // 0 is used when unpack is empty
  SmallVector<bool> unpackEmpty = {false};
  SmallVector<SmallVector<int64_t>> innerPerm = {{1, 0}};
  SmallVector<SmallVector<int64_t>> outerPerm = {{0, 1}};
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  // Pack level => 2.
  // packed size for [M, N, K, m, n, k]
  const int packLevel = 2;
  packedSizes = getPackedSize(linalgOp, packLevel);
  // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
  // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
  // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
  transposePackIndices = {0, 1, 2};
  // Only the third pack operation has a corresponding unpack operation
  unpackEmpty = {false, false, true};
  innerPerm = {{0, 1}, {1, 0}, {0, 1}};
  outerPerm = {{0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}};
  auto packingConfigLevel2Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel1Attr, packingConfigLevel2Attr};
  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  // Currently, assume working on a 2x2 AIE array, so the second level tile
  // sizes should be (tileM0/2, tileN0/2). Considering the packing sizes, the
  // adjusted tile sizes should be (tileM0/2/packedM1, tileN0/2/packedN1).
  auto packedM1 = packedSizes[3];
  auto packedN1 = packedSizes[4];
  auto tileM1 = findLargestFactor((int)tileM0 / packedM1,
                                  16 * tilingScaleFactor / packedM1);
  auto tileN1 = findLargestFactor((int)tileN0 / packedN1,
                                  16 * tilingScaleFactor / packedN1);
  // Set tile size for K as constant 1, so that the packed outer K dimension
  // is 1.
  const int tileK = 1;

  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, tileK};
  SmallVector<int64_t> TileSizeLevel2 = {0, 0, 0, tileM1, tileN1, 0};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::None))) {
    return failure();
  }
  return success();
}

static LogicalResult setRootConfigForPadPackPipeline(func::FuncOp entryPointFn,
                                                     linalg::LinalgOp linalgOp,
                                                     AIEConfig cfg) {
  // Assume working on a 4x4 AIE array. Currently, the tile sizes are chosen
  // empirically for large GEMM sizes, which are [64*s, 64*s, 256] for the first
  // level and [16*s, 16*s, 16*s] for the second level, where 's' is the scaling
  // scaling factor based on the element type's bit width. Basic min/max
  // constraints are added to avoid failure for small GEMM sizes.
  auto initType = linalgOp.getDpsInitOperand(0)->get().getType();
  auto initShape = llvm::cast<ShapedType>(initType).getShape();
  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto lhsShape = lhsType.getShape();

  FailureOr<unsigned> maybeTilingScaleFactor =
      getTilingScaleFactor(lhsType.getElementType());
  if (failed(maybeTilingScaleFactor)) {
    return linalgOp.emitOpError("expected bitwidth 64/32/16/8");
  }
  unsigned tilingScaleFactor = maybeTilingScaleFactor.value();
  // TODO (nmeshram) : We should be able to use fixed tiling config after we
  // have padding support.
  auto tileM0 = findLargestFactor((int)initShape[0], 64 * tilingScaleFactor);
  auto tileN0 = findLargestFactor((int)initShape[1], 64 * tilingScaleFactor);
  auto tileM1 = findLargestFactor((int)tileM0, 16 * tilingScaleFactor);
  auto tileN1 = findLargestFactor((int)tileN0, 16 * tilingScaleFactor);
  auto tileK0 = findLargestFactor((int)lhsShape[1], 256);

  // Do packing first to allow larger k packing
  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  const int packLevel = 1;
  auto packedSizes =
      getPackedSize(linalgOp, packLevel, tileM1, tileN1, (int)tileK0);
  SmallVector<int64_t> transposePackIndices = {0, 1, 2};
  SmallVector<bool> unpackEmpty = {false, false, true};
  SmallVector<SmallVector<int64_t>> innerPerm = {{0, 1}, {1, 0}, {0, 1}};
  SmallVector<SmallVector<int64_t>> outerPerm = {{1, 0}, {1, 0}, {1, 0}};
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);
  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel1Attr};

  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // Finish rest of tiling
  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  auto tileK1 = findLargestFactor((int)tileK0 / (int)packedSizes[2],
                                  2 * tilingScaleFactor);
  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, tileK0};
  SmallVector<int64_t> TileSizeLevel2 = {tileM1, tileN1};
  SmallVector<int64_t> TileSizeLevel3 = {0, 0, tileK1};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1, TileSizeLevel2,
                                 TileSizeLevel3};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::None))) {
    return failure();
  }

  return success();
}

/// Sets the lowering configuration for a generic op implementing a
/// transposition.
static LogicalResult setTransposeLikeOpRootConfig(
    func::FuncOp entryPointFn, linalg::LinalgOp linalgOp,
    AIEPassPipeline usePassPipeline, AIEConfig cfg) {
  if (usePassPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg);
  else if (usePassPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg);
  return linalgOp.emitOpError("unhandled pass pipeline");
}

/// Utility to check if an elementwise op is fusable with its producer.
static bool isMatmulElementwiseFusion(linalg::GenericOp genericOp) {
  // Check if any of the defining op is a matmul-like op. To simplify the
  // problem, currently only check if it is a contraction op.
  for (auto operand : genericOp.getOperands()) {
    auto defOp = operand.getDefiningOp();
    if (defOp &&
        linalg::isaContractionOpInterface(cast<linalg::LinalgOp>(defOp))) {
      return true;
    }
  }
  return false;
}

/// Sets the lowering configuration for a generic op which is an elementwise op
/// and can be fused with its producer matmul-like op in the same dispatch.
static LogicalResult setElementwiseFusionRootConfig(
    func::FuncOp entryPointFn, linalg::LinalgOp linalgOp,
    AIEPassPipeline usePassPipeline, AIEConfig cfg) {
  if (usePassPipeline != AIEPassPipeline::PackPeelPipeline) return failure();

  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  // Only set the first level of tiling and packing configs, and these configs
  // should be consistent with the its producer's settings.
  auto initType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  auto initShape = initType.getShape();

  FailureOr<unsigned> maybeTilingScaleFactor =
      getTilingScaleFactor(initType.getElementType());
  if (failed(maybeTilingScaleFactor)) {
    return linalgOp.emitOpError("expected bitwidth 64/32/16/8");
  }
  unsigned tilingScaleFactor = maybeTilingScaleFactor.value();
  auto tileM0 = findLargestFactor((int)initShape[0], 32 * tilingScaleFactor);
  auto tileN0 = findLargestFactor((int)initShape[1], 32 * tilingScaleFactor);
  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  TileSizesListType tileSizes = {TileSizeLevel0};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::None))) {
    return failure();
  }

  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  SmallVector<int64_t> packedSizes = {tileM0, tileN0};
  auto packingConfigLevel1Attr =
      getPackingConfigPackingLevelAttr(context, packedSizes);
  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel1Attr};

  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);
  return success();
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::GenericOp genericOp,
                                   AIEPassPipeline usePassPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");

  if (isMatmulTranspose(genericOp) &&
      succeeded(setTransposeLikeOpRootConfig(entryPointFn, genericOp,
                                             usePassPipeline, cfg))) {
    return success();
  }

  if (isElementwise(genericOp) && isMatmulElementwiseFusion(genericOp) &&
      succeeded(setElementwiseFusionRootConfig(entryPointFn, genericOp,
                                               usePassPipeline, cfg))) {
    return success();
  }

  return failure();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::ContractionOpInterface contractionOp,
                                   AIEPassPipeline usePassPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig(contractionOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
  if (isa<linalg::MatmulTransposeBOp>(linalgOp)) {
    if (succeeded(setTransposeLikeOpRootConfig(entryPointFn, linalgOp,
                                               usePassPipeline, cfg))) {
      return success();
    }
    return failure();
  }
  unsigned numLoops = linalgOp.getNumLoops();
  {
    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    if (dims.size() != 1 || dims[0] != numLoops - 1) {
      return linalgOp.emitOpError(
          "expected to have exactly one reduction dim, and it is the innermost "
          "dim");
    }
  }

  // TODO (nmeshram) : This needs to be moved in a separate more generalized
  // logic. Also, need a flag to experiment between pad based and pack based
  // approach which will have different tile sizes and pass pipelines
  if (usePassPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg);
  if (usePassPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg);
  return linalgOp.emitOpError("unhandled pass pipeline");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(func::FuncOp entryPointFn, Operation *op,
                                       AIEPassPipeline usePassPipeline,
                                       AIEConfig cfg) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline, cfg);
        })
        .Case<linalg::ContractionOpInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline, cfg);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    AIEPassPipeline usePassPipeline, AIEConfig cfg) {
  // Make sure that lowering_config is not preset on any compute ops.
  Operation *elementwiseOp = nullptr;
  for (auto computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) return failure();
    if (isa<linalg::GenericOp>(computeOp) &&
        isElementwise(cast<linalg::LinalgOp>(computeOp))) {
      elementwiseOp = computeOp;
    }
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) return failure();
  Operation *rootOperation = rootOp.value();

  // TODO (nmeshram): Handle the case with no known root operation.
  if (!rootOperation) {
    return entryPointFn.emitError("Case with no root ops not yet supported.");
  }

  if (failed(setRootConfigImpl(entryPointFn, rootOperation, usePassPipeline,
                               cfg))) {
    return failure();
  }

  // For matmul + elementwise ops fusion, also set configs for elementwise op.
  if (elementwiseOp && elementwiseOp != rootOperation) {
    if (failed(setRootConfigImpl(entryPointFn, elementwiseOp, usePassPipeline,
                                 cfg))) {
      return failure();
    }
  }

  // TODO (nmeshram): // Set vector level tile sizes for other operations
  // individually.

  return success();
}

LogicalResult initAIELaunchConfig(ModuleOp moduleOp,
                                  AIEPassPipeline usePassPipeline,
                                  AIEConfig cfg) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    if (getTranslationInfo(exportOp)) continue;

    // TODO (nmeshram): Need a default pipeline for control flow cases.
    if (funcOp.getBody().empty() || !llvm::hasSingleElement(funcOp.getBody())) {
      return funcOp.emitError("control flow not yet supported.");
    }

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps,
                                               usePassPipeline, cfg))) {
      return failure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

}  // namespace mlir::iree_compiler::AMDAIE
