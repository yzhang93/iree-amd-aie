// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREECodegenAttrs.h - Codegen dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_AMDAIE_DIALECT_PACKINGCONFIG_H_
#define IREE_COMPILER_AMDAIE_DIALECT_PACKINGCONFIG_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {
struct PackConfigTD {
  // Expected packed sizes for specified iterator dimensions
  ArrayRef<int64_t> packedSizes;
  // Indices of pack operations need to be transposed
  ArrayRef<int64_t> transposePackIndices;
  // Indicator of if there is a unpack op corresponding to a pack op
  ArrayRef<int64_t> unpackEmpty;
  // Attributes for inner dimension permutation
  ArrayRef<ArrayRef<int64_t>> innerPerm;
  // Attributes for outer dimension permutation
  ArrayRef<ArrayRef<int64_t>> outerPerm;
};
/// Typedef for packing config to use at different levels of packing.
using PackingConfigListType = SmallVector<PackConfigTD>;
using PackingConfigListTypeRef = ArrayRef<PackConfigTD>;
struct PackConfig {
  // Expected packed sizes for specified iterator dimensions
  SmallVector<OpFoldResult> packedSizes;
  // Indices of pack operations need to be transposed
  SmallVector<int64_t> transposePackIndices;
  // Indicator of if there is a unpack op corresponding to a pack op
  SmallVector<int64_t> unpackEmpty;
  // Attributes for inner dimension permutation
  SmallVector<SmallVector<int64_t>> innerPerm;
  // Attributes for outer dimension permutation
  SmallVector<SmallVector<int64_t>> outerPerm;
};

}  // namespace mlir::iree_compiler

// clang-format off
#include "iree-amd-aie/IR/AMDAIEEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIEAttrs.h.inc"
// clang-format on

namespace mlir::iree_compiler {
/// Returns the packing configuration set for an operation. Returns `nullptr`
/// if no value is set.  It expects that the attribute is stored using the
/// identifier `packing_config`.
AMDAIE::PackingConfigAttr getPackingConfig(Operation *op);

/// Sets the packing configuration, overwriting existing attribute values.
void setPackingConfig(Operation *op, AMDAIE::PackingConfigAttr config);

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_AMDAIE_DIALECT_PACKINGCONFIG_H_
