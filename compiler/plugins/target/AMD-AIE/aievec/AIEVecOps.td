//===- AIEVecOps.td - AIE vector op definitions -----------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_OPS
#define AIEVEC_OPS

// include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "AIEVecAttributes.td"
include "AIEVecTypeConstraints.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for AIE dialect ops.
class AIEVec_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEVec_Dialect, mnemonic, traits> {
  // For every AIE vector op, there needs to be a:
  //   * void ${C++ class of Op}::print(OpAsmPrinter &p)
  //   * LogicalResult ${C++ class of Op}::verify()
  //   * ParseResult ${C++ class of Op}::parse(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AIEVec_UPSOp:
  AIEVec_Op<"ups", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source,
          DefaultValuedAttr<ConfinedAttr<I8Attr, [IntNonNegative]>, "0">:$shift)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE ups";
  let description = [{
    AMD-specific upshift intrinsic. Moves data from AIE vector data type
    to accumulator data type. The adjustment in precision is controlled by
    the shift parameter.
    `$result = ups($source, $shift)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "int8_t":$shift),
    [{build($_builder, $_state, source.getType(), source, shift);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_CastOp:
  AIEVec_Op<"cast", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source, DefaultValuedAttr<BoolAttr, "false">:$isResAcc)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE cast";
  let description = [{
    AIE2 cast intrinsic. Cast values from source data type to result data types.
    `$result = cast($source, isResAcc)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "bool":$isResAcc),
    [{build($_builder, $_state, source.getType(), source, isResAcc);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_SRSOp:
  AIEVec_Op<"srs", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source, AnyInteger:$shift)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE srs";
  let description = [{
    AMD-specific shift-round-saturate intrinsic. Moves values from
    accumulator data type to AIE vector data types. The adjustment in
    precision is controlled by the shift parameter.
    `$result = srs($source, $shift)`
  }];
  let hasFolder = 1;
}

def AIEVec_MatMulOp:
  AIEVec_Op<"matmul", [
    Pure,
    AllRanksMatch<["lhs", "rhs", "acc"]>,
    AllTypesMatch<["acc", "result"]>,
    ShapesCompatibleWithContraction<"lhs", "rhs", "acc">,
    IsValidAIE2MatMulShapeAndType<"lhs", "rhs", "acc">
  ]>,
  Arguments<(ins AIE2MatMulLHS:$lhs,
                 AIE2MatMulRHS:$rhs,
                 AIE2MatMulACC:$acc)>,
  Results<(outs AIE2MatMulACC:$result)> {
  let summary = "AIE2 matrix-multiply and accummulate";
  let description = [{
    AMD AIEv2-specific intrinsic that performs a matrix multiplications
    between `lhs` and `rhs`, and accumulates the result in `acc`.

    Currently, this intrinsic supports the following type combinations:

         lhs                | rhs                | Accumulator
        :------------------:|:------------------:|:-----------------:
         `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
         `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
         `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
         `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
         `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
         `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
         `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
         `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
         `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
         `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`
  }];
  let assemblyFormat = [{$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,`
                         type($rhs) `into` type($acc)}];
  let hasVerifier = 0;
}

def AIEVec_ShuffleOp : AIEVec_Op<"shuffle",
    [Pure, AllTypesMatch<["lhs", "result"]>,
     OptionalTypesMatchWith<"result and rhs have the same type", "result", "rhs",
                            "::llvm::cast<Type>($_self)">]>,
  Arguments<(ins VectorOfBitWidthAndElementTypes<
                      512, [I8, I16, I32, I64, I128, I256,
                            I512, BF16, F32]>:$lhs,
                 Optional<VectorOfBitWidthAndElementTypes<
                      512, [I8, I16, I32, I64, I128, I256,
                            I512, BF16, F32]>>:$rhs,
                 AIEVec_ShuffleModeAttr:$mode)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 shuffle";
  let description = [{
    AMD AIEv2-specific vector shuffle. It performs a shuffle of the elements of
    1 or 2 input vectors using the specified shuffle mode. The shuffle mode is
    specified as:

      `t<width>_<r>x<c>(_(hi|lo))?`

    where `<width>` is the bitwidth of the vector element type, `<r>` and `<c>`
    are the number of rows and columns that will be transposed to perform the
    shuffle, and, for modes that require two 512-bit vectors, `hi` and `lo`
    indicate which part of the resulting extended 1024-bit vector will be
    assembled and returned.

    E.g.: `t32_4x8` would take two 512-bit vectors, `lhs` and `rhs`, with 16
    elements of 32 bits each. The resulting vector would contain either the
    least (`lo`) or most (`hi`) significant 16 elements of the 32 element vector
    that would result from selecting, out of the concatenated vectors `lhs:rhs`,
    8 blocks of 4 elements, each block taking one of every 8 elements starting
    from the block index.

    That is, for two `vector<16xi32>` operands containing:
    ```
    lhs = [0,   1,  2,  3, ..., 15]
    rhs = [17, 18, 19, 20, ..., 31]
    ```

    The first 8 blocks would be:
    ```
    b0 = [0,  8, 16, 24]
    b1 = [1,  9, 17, 25]
    b2 = [2, 10, 18, 26]
    b3 = [3, 11, 19, 27]
       ...
    b7 = [7, 15, 23, 31]
    ```

    `t32_4x8_lo` would return first four blocks:
    ```
    result = [0, 8, 16, 24, 1, 9, 17, 25, ..., 3, 11, 19, 27]
    ```

    And `t32_4x8_hi` would return the last four blocks:
    ```
    result = [4, 12, 20, 28, 5, 13, 21, 29, ..., 7, 15, 24, 31]
    ```

    It can be seen as flattened 4x8 matrix, split in two 16-element halfs, being
    tranposed to a 8x4 arrangement. In the example above:

    ```
    lhs = [ 0,  1,  2,  3,  4,  5,  6,  7]
          [ 8,  9, 10, 11, 12, 13, 14, 15]
    rhs = [16, 17, 18, 19, 20, 21, 22, 23]
          [24, 25, 26, 27, 28, 29, 30, 31]
    ```

    Would result in:
    ```
    t32_4x8_lo = [0,  8, 16, 24]
                 [1,  9, 17, 25]
                 [2, 10, 18, 26]
                 [3, 11, 19, 27]
    t32_4x8_hi = [4, 12, 20, 28]
                 [5, 13, 21, 29]
                 [6, 14, 22, 30]
                 [7, 15, 23, 31]
    ```

    A special mode, `t16_1x2_flip`, swaps each pair of elements in a vector with
    32 16-bit elements. E.g.:
    ```
    lhs = [0, 1, 2, 3, ..., 28, 29, 30, 31]
    ```
    Would result in:
    ```
    t16_1x2_flip = [1, 0, 3, 2, ..., 29, 28, 31, 30]
    ```

    The list of supported shuffle modes, required operands, and associated
    vector types are the following:

         Shuffle Mode       | Operands           | Types Supported
        :------------------:|:------------------:|:------------------:
         t8_8x4             | `lhs`              | `vector<64xi8>`
         t8_4x8             | ^                  | ^
         t8_8x8             | ^                  | ^
         t8_16x4            | ^                  | ^
         t8_4x16            | ^                  | ^
         t8_64x2_lo         | `lhs` & `rhs`      | ^
         t8_64x2_hi         | ^                  | ^
         t8_2x64_lo         | ^                  | ^
         t8_2x64_hi         | ^                  | ^
         t16_4x2            | `lhs`              | `vector<32xi16>` or `vector<32xbf16>`
         t16_2x4            | ^                  | ^
         t16_4x4            | ^                  | ^
         t16_8x2            | ^                  | ^
         t16_2x8            | ^                  | ^
         t16_8x4            | ^                  | ^
         t16_4x8            | ^                  | ^
         t16_16x2           | ^                  | ^
         t16_2x16           | ^                  | ^
         t16_1x2_flip       | ^                  | ^
         t16_32x2_lo        | `lhs` & `rhs`      | ^
         t16_32x2_hi        | ^                  | ^
         t16_2x32_lo        | ^                  | ^
         t16_2x32_hi        | ^                  | ^
         t16_16x4_lo        | ^                  | ^
         t16_16x4_hi        | ^                  | ^
         t16_4x16_lo        | ^                  | ^
         t16_4x16_hi        | ^                  | ^
         t32_4x4            | `lhs`              | `vector<16xi32>` or `vector<16xf32>`
         t32_16x2_lo        | `lhs` & `rhs`      | ^
         t32_16x2_hi        | ^                  | ^
         t32_2x16_lo        | ^                  | ^
         t32_2x16_hi        | ^                  | ^
         t32_8x4_lo         | ^                  | ^
         t32_8x4_hi         | ^                  | ^
         t32_4x8_lo         | ^                  | ^
         t32_4x8_hi         | ^                  | ^
         t64_8x2_lo         | ^                  | `vector<8xi64>`
         t64_8x2_hi         | ^                  | ^
         t64_2x8_lo         | ^                  | ^
         t64_2x8_hi         | ^                  | ^
         t128_4x2_lo        | ^                  | `vector<4xi128>`
         t128_4x2_hi        | ^                  | ^
         t128_2x4_lo        | ^                  | ^
         t128_2x4_hi        | ^                  | ^
         t256_2x2_lo        | ^                  | `vector<2xi256>`
         t256_2x2_hi        | ^                  | ^
         t512_1x2_lo        | ^                  | `vector<1xi512>`
         t512_1x2_hi        | ^                  | ^
  }];
  let assemblyFormat = [{$lhs (`,` $rhs^)? $mode attr-dict `:` type($result)}];
  let hasVerifier = 1;
}

def AIEVec_MulElemOp:
  AIEVec_Op<"mul_elem", [
    Pure,
    SameTypeOperands,
    SameOperandsShape,
    SameOperandsAndResultShape,
    isOperandResultTypePairValidForAIE2MulElem<"lhs", "rhs", "result">
  ]>,
  Arguments<(ins
    VectorOfLengthAndType<[16, 32], [I8, I16, I32, BF16, F32]>:$lhs,
    VectorOfLengthAndType<[16, 32], [I8, I16, I32, BF16, F32]>:$rhs)>,
  Results<(outs
    VectorOfLengthAndType<[16, 32], [I32, I64, F32]>:$result)> {
  let summary = "AIE2 vector element-wise multiply";
  let description = [{
    AMD-specific multiply operation that multiplies two 1-D vectors in the same channel.
    The vector sizes are at least 512 bits.
    `$result = `$lhs * $rhs`.
    Currently, the following are the supported type combinations:
        lhs                | rhs                | Accumulator
      :------------------:|:------------------:|:-----------------:
        `vector<32xi8>`    | `vector<32xi8>`    | `vector<32xi32>`
        `vector<32xi16>`   | `vector<32xi16>`   | `vector<32xi32>`
        `vector<16xi32>`   | `vector<16xi32>`   | `vector<16xi64>`
        `vector<16xbf16>`  | `vector<16xbf16>`  | `vector<16xf32>`
        `vector<16xf32>`   | `vector<16xf32>`   | `vector<16xf32>`'
  }];
}

def AIEVec_FMAElemOp :
  AIEVec_Op<"mac_elem", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc,
               DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 element-wise vector fused multiply-add";
  let description = [{
    AMD-specific multiply-add operation. It multiplies two 1-D vectors in the same channel,
    and adds the result to an accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the
    'fmsub' bool to true.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Value":$acc,
            "bool":$fmsub),
    [{build($_builder, $_state, acc.getType(), lhs, rhs, acc,
            fmsub);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attribute names
    llvm::StringRef getSubAttrName() { return "fmsub"; }
  }];
}

#endif // AIEVEC_OPS
