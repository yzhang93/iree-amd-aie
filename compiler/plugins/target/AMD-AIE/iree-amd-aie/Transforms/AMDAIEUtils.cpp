// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Generate a DenseMap key we can use for the element types (alternatives
// considered: implement tombstone for std::array, or use std::map instead of
// DenseMap).
constexpr uint32_t getElementTypeKey(uint32_t a, uint32_t b, uint32_t c) {
  return a + (b << 8) + (c << 16);
}

// Map from (LHS bitwidth, RHS bitwidth, Accumulator bitwidth) to the AIE
// instruction size (m, n, k) for the integer types with those bitwidths.
const auto& getIntegerMatmulInstructionSizeMap() {
  // Sanity check.
  static_assert(getElementTypeKey(1, 2, 3) == 1 + 2 * 256 + 3 * 65536);

  static DenseMap<uint32_t, std::array<uint32_t, 3>> matmulIntSizes{

      // `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
      {getElementTypeKey(8, 4, 32), {4, 8, 16}},

      // `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(8, 8, 32), {4, 8, 8}},

      // `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(16, 8, 32), {4, 8, 4}},

      // `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
      {getElementTypeKey(16, 16, 32), {4, 8, 2}},

      // `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
      // `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
      //   choosing the first i16 x i8 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 8, 64), {2, 8, 8}},

      // `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
      // `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
      //   choosing the first i16 x i16 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 16, 64), {2, 8, 4}},

      // `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
      {getElementTypeKey(32, 16, 64), {4, 4, 2}},
  };
  return matmulIntSizes;
}
}  // namespace

FailureOr<std::array<uint32_t, 3>> getAIEIntegerMatmulInstructionSize(
    uint32_t nBitsLhs, uint32_t nBitsRhs, uint32_t nBitsAcc) {
  const auto& mapForIntTypes = getIntegerMatmulInstructionSizeMap();
  auto it =
      mapForIntTypes.find(getElementTypeKey(nBitsLhs, nBitsRhs, nBitsAcc));
  if (it == mapForIntTypes.end()) {
    return failure();
  }
  return it->second;
}

FailureOr<std::array<uint32_t, 3>> getAIEMatmulInstructionSize(Type elTypeLhs,
                                                               Type elTypeRhs,
                                                               Type elTypeAcc) {
  bool allFloatingPoint = elTypeLhs.isa<FloatType>() &&
                          elTypeRhs.isa<FloatType>() &&
                          elTypeAcc.isa<FloatType>();

  bool allInteger = elTypeLhs.isa<IntegerType>() &&
                    elTypeRhs.isa<IntegerType>() &&
                    elTypeAcc.isa<IntegerType>();

  if (!allInteger && !allFloatingPoint) {
    return failure();
  }

  auto nBitsLhs = elTypeLhs.getIntOrFloatBitWidth();
  auto nBitsRhs = elTypeRhs.getIntOrFloatBitWidth();
  auto nBitsAcc = elTypeAcc.getIntOrFloatBitWidth();

  if (allFloatingPoint) {
    if (nBitsLhs == 16 && nBitsRhs == 16 && nBitsAcc == 32) {
      return std::array<uint32_t, 3>{4, 4, 8};
    }
    // There is only 1 floating point case in the table (handled above).
    return failure();
  }

  assert(allInteger);

  return getAIEIntegerMatmulInstructionSize(nBitsLhs, nBitsRhs, nBitsAcc);
}

FailureOr<unsigned> getTilingScaleFactor(Type elemType) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth %8 != 0) return failure();
  if (bitWidth > 64) return failure();
  return 64 / bitWidth;
}

// Find the largest factor of 'num' which is not larger than 'max'.
int detail::findLargestFactor(int num, int max) {
  assert(max > 0 && "No factors less than or equal to 0 exist");

  // Do O(1) instead of O(sqrt(num)) computation for this common case.
  if (num <= max) {
    return num;
  }

  int largestLowFactor = 1;
  for (int lowFactor = 2; lowFactor <= max; ++lowFactor) {
    const int highFactor = num / lowFactor;

    // This early exit is what makes this O(sqrt(num)) instead of O(num).
    if (highFactor < lowFactor) return largestLowFactor;

    const bool areActuallyFactors = num % lowFactor == 0;
    if (areActuallyFactors) {
      // We're certain that here lowFactor <= highFactor, and highFactor is
      // descending in this loop. So we can return immediately if highFactor is
      // good.
      if (highFactor <= max) return highFactor;
      largestLowFactor = lowFactor;
    }
  }
  return largestLowFactor;
}

/// TODO(avarma): This currently is skipping checking for ext* ops.
static bool bodyMatcherForMatmulTranspose(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_nonnull<arith::MulIOp, arith::MulFOp>(mulOp)) {
    return false;
  }
  auto lhsBlockArg = mulOp->getOperand(0).dyn_cast<BlockArgument>();
  auto rhsBlockArg = mulOp->getOperand(1).dyn_cast<BlockArgument>();
  auto outBlockArg = addOp->getOperand(0).dyn_cast<BlockArgument>();
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// `isMatmulTranspose` is a utility function that aims to indentify whether a
/// linalg.generic op is a matmul transpose op.
bool isMatmulTranspose(linalg::GenericOp genericOp) {
  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul transpose.
  Block *body = genericOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulTranspose(yieldVal, body)) {
    return false;
  }
  // Step 2. Check iterator types.
  SmallVector<utils::IteratorType> matmulTransposeIteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      genericOp.getIteratorTypesArray();
  if (matmulTransposeIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Step 3. Test the indexing maps.
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  // Extract dimensions for MxK * NxK -> MxN
  AffineExpr m = map2.getResult(0);
  AffineExpr n = map2.getResult(1);
  AffineExpr k = map0.getResult(1);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

}  // namespace mlir::iree_compiler::AMDAIE
