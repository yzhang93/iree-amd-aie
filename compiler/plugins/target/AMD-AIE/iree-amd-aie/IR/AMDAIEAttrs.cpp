// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIEAttrs.cpp.inc"
#include "iree-amd-aie/IR/AMDAIEEnums.cpp.inc"

static const char kPackingConfigAttrName[] = "packing_config";

namespace mlir::iree_compiler {

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of 64-bit
/// integer type whose values is obtained from `values`.
static ArrayAttr getIndexArrayAttr(MLIRContext *context,
                                   ArrayRef<int64_t> values) {
  return ArrayAttr::get(
      context, llvm::map_to_vector(values, [&](int64_t value) -> Attribute {
        return IntegerAttr::get(IndexType::get(context), APInt(64, value));
      }));
}

}  // namespace mlir::iree_compiler

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
// amdaie.packing_config
//===----------------------------------------------------------------------===//

PackingConfigAttr PackingConfigAttr::get(
    MLIRContext *context, PackingConfigListTypeRef packingConfigs) {
  Builder builder(context);
  SmallVector<PackingConfigPackingLevelAttr> packinglevels;
  for (auto [level, configs] : llvm::enumerate(packingConfigs)) {
    // Form `innerPerm` attribute which is a multi-dimensional array attribute.
    SmallVector<PermLevelAttr> innerPermLevels;
    for (auto row : configs.innerPerm) {
      innerPermLevels.push_back(PermLevelAttr::get(context, row));
    }
    auto innerPerm = PermLevelsAttr::get(context, innerPermLevels);

    // Form `outerPerm` attribute which is a multi-dimensional array attribute.
    SmallVector<PermLevelAttr> outerPermLevels;
    for (auto row : configs.outerPerm) {
      outerPermLevels.push_back(PermLevelAttr::get(context, row));
    }
    auto outerPerm = PermLevelsAttr::get(context, outerPermLevels);

    packinglevels.push_back(PackingConfigPackingLevelAttr::get(
        context, configs.packedSizes, configs.transposePackIndices,
        configs.unpackEmpty, innerPerm, outerPerm));
  }
  return get(context,
             PackingConfigPackingLevelsAttr::get(context, packinglevels));
}

PackConfig PackingConfigAttr::getPackingConfigVals(unsigned level) {
  Builder builder(getContext());
  auto levels = getPackingLevels();
  if (level >= levels.size()) return {};
  PackConfig packConfig;
  SmallVector<OpFoldResult> packedSizes;
  for (int64_t packedSize : levels[level].getPackedSizes()) {
    packedSizes.push_back(builder.getI64IntegerAttr(packedSize));
  }
  packConfig.packedSizes = packedSizes;

  SmallVector<int64_t> transposePackIndices;
  for (int64_t transposePackIndex : levels[level].getTransposePackIndices()) {
    transposePackIndices.push_back(transposePackIndex);
  }
  packConfig.transposePackIndices = transposePackIndices;

  SmallVector<int64_t> unpackEmpty;
  for (int64_t unpackEmptyVal : levels[level].getUnpackEmpty()) {
    unpackEmpty.push_back(unpackEmptyVal);
  }
  packConfig.unpackEmpty = unpackEmpty;

  // Fetch `innerPerm`.
  SmallVector<SmallVector<int64_t>> innerPerm;
  for (auto &permLevel : levels[level].getInnerPerm()) {
    SmallVector<int64_t> permLevelVal;
    for (int64_t permVal : permLevel.getPerm()) {
      permLevelVal.push_back(permVal);
    }
    innerPerm.push_back(permLevelVal);
  }
  packConfig.innerPerm = innerPerm;

  // Fetch `outerPerm`.
  SmallVector<SmallVector<int64_t>> outerPerm;
  for (auto &permLevel : levels[level].getOuterPerm()) {
    SmallVector<int64_t> permLevelVal;
    for (int64_t permVal : permLevel.getPerm()) {
      permLevelVal.push_back(permVal);
    }
    outerPerm.push_back(permLevelVal);
  }
  packConfig.outerPerm = outerPerm;

  return packConfig;
}

void AMDAIEDialect::initializeAMDAIEAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-amd-aie/IR/AMDAIEAttrs.cpp.inc"  // IWYU pragma: keeep
      >();
}

}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `amdaie.packing_config` attribute.
// ===----------------------------------------------------------------------===//

AMDAIE::PackingConfigAttr getPackingConfig(Operation *op) {
  return op->getAttrOfType<AMDAIE::PackingConfigAttr>(kPackingConfigAttrName);
}

void setPackingConfig(Operation *op, AMDAIE::PackingConfigAttr config) {
  op->setAttr(kPackingConfigAttrName, config);
}

}  // namespace mlir::iree_compiler
