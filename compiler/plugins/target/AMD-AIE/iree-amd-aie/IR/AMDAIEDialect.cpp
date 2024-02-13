// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDialect.cpp.inc"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::AMDAIE {

void AMDAIEDialect::initialize() { initializeAMDAIEAttrs(); }

}  // namespace mlir::iree_compiler::AMDAIE
