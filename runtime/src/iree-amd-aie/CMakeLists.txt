# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  add_subdirectory(driver/xrt)
endif()

if("xrt-lite" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  add_subdirectory(driver/xrt-lite)
endif()

# Flatbuffer schema generation does not require a driver but the generated
# flatbuffer header files are used by the compiler to create artefacts
# (.vmfb file), and so the schema sub-directory is required even when not
# building driver code.
add_subdirectory(schemas)

# Contains libiree_aie_runtime, i.e., suitably encapsulated calls to aie-rt.
# Note, "runtime" is currently a misnomer as all such calls are made at
# compile/configuration time; the name is inspired by aie-rt itself
# (where rt stands for runtime) and the fact that in the future, the plan is indeed
# to make these same calls at runtime.
add_subdirectory(aie_runtime)
