// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS
#define IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def AMDAIE_MemSpace_L1 : I32EnumAttrCase<"L1", 1, "l1">;
def AMDAIE_MemSpace_L2 : I32EnumAttrCase<"L2", 2, "l2">;
def AMDAIE_MemSpaceAttr: I32EnumAttr<"AMDAIEMemSpace", "AIE Memory Space",
  [
    AMDAIE_MemSpace_L1,
    AMDAIE_MemSpace_L2,
  ]> {

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_PermLevelAttr :
  AttrDef<AMDAIE_Dialect, "PermLevel", []>
{
  let mnemonic = "perm_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t","">:$perm
  );
  
  let assemblyFormat = [{
    `[` $perm `]`
  }];
}

def AMDAIE_PermLevelsAttr :
  ArrayOfAttr<AMDAIE_Dialect, "PermLevels",
    "perm_levels", "PermLevelAttr", []>
{
}

def AMDAIE_PackingConfigPackingLevelAttr :
  AttrDef<AMDAIE_Dialect, "PackingConfigPackingLevel", []>
{
  let mnemonic = "packing_config_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t",
        "Expected packed sizes for specified iterator dimensions">:$packedSizes,
    ArrayRefParameter<"int64_t",
        "Indices of pack operations need to be transposed">:$transposePackIndices,
    ArrayRefParameter<"int64_t",
        "Indicator of if there is a unpack op corresponding to a pack op">:$unpackEmpty,
    AttrParameter<"PermLevelsAttr",
        "Attributes for inner dimension permutation">:$innerPerm,
    AttrParameter<"PermLevelsAttr",
        "Attributes for outer dimension permutation">:$outerPerm
  );
  
  let assemblyFormat = [{
    `{` `packedSizes` `=` `[` $packedSizes `]` `,` `transposePackIndices` `=` `[` $transposePackIndices `]` `,` `unpackEmpty` `=` `[` $unpackEmpty `]` `,` `innerPerm` `=` $innerPerm `,` `outerPerm` `=` $outerPerm `}`
  }];
}

def AMDAIE_PackingConfigPackingLevelsAttr :
  ArrayOfAttr<AMDAIE_Dialect, "PackingConfigPackingLevels",
    "packing_config_levels", "PackingConfigPackingLevelAttr", []>
{
}

def AMDAIE_PackingConfigAttr :
    AttrDef<AMDAIE_Dialect, "PackingConfig", []> {
  let mnemonic = "packing_config";
  let summary = [{drive lowering of an operation within dispatch region via a given packing config}];
  let description = [{
    Specifies the information that is used by the iree-pack-and-transpose
    pass to help in the lowering of an operation within a dispatch region.
  }];

  let assemblyFormat = [{
    `<` `packing_config` `=` $packingLevels `>`
  }];

  let parameters = (ins
    AttrParameter<"PackingConfigPackingLevelsAttr",
        "The packing config at different levels">:$packingLevels
  );
  let builders = [
    AttrBuilder<(ins "PackingConfigListTypeRef":$packingConfigs)>
  ];
  let extraClassDeclaration = [{
    // Returns the packing config for a level set for the op.
    PackConfig getPackingConfigVals(unsigned level);
  }];
}

#endif // IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS
