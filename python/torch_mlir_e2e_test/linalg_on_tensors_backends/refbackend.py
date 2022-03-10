# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ctypes
import numpy as np

from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.execution_engine import *
from torch_mlir.runtime import *
# Imported for side effects.
import torch_mlir.all_passes_registration
import torch_mlir.dialects.torch

from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report

from .abc import LinalgOnTensorsBackend

__all__ = [
    "RefBackendLinalgOnTensorsBackend",
]


def checkArgTypeIsSupported(ty):
    SUPPORTED = [np.float32, np.float64, np.int32, np.int64, np.bool_]
    assert ty in SUPPORTED, f"Only numpy arrays with dtypes in {SUPPORTED} are supported"


class RefBackendInvoker:
    def __init__(self, module):
        self.ee = ExecutionEngine(module)
        self.result = None

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri1(a):
            self.result = unranked_memref_to_numpy(a, np.bool_)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri32(a):
            self.result = unranked_memref_to_numpy(a, np.int32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri64(a):
            self.result = unranked_memref_to_numpy(a, np.int64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32(a):
            self.result = unranked_memref_to_numpy(a, np.float32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf64(a):
            self.result = unranked_memref_to_numpy(a, np.float64)

        @ctypes.CFUNCTYPE(None, ctypes.c_bool)
        def consume_return_i1(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.c_int)
        def consume_return_i64(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.c_float)
        def consume_return_f32(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.c_double)
        def consume_return_f64(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mri64(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.int64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mrf32(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.float32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf64_mrf64(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float64), unranked_memref_to_numpy(
                    arg1,
                    np.float64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mrf32_mrf32(arg0, arg1, arg2):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.float32), unranked_memref_to_numpy(arg2, np.float32)

        @ctypes.CFUNCTYPE(None,
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mri64_mri64_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mrf32(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61, arg62, arg63, arg64, arg65, arg66, arg67, arg68, arg69, arg70, arg71, arg72, arg73, arg74, arg75, arg76, arg77, arg78, arg79, arg80, arg81, arg82, arg83, arg84, arg85, arg86, arg87, arg88, arg89, arg90, arg91, arg92, arg93, arg94, arg95, arg96, arg97, arg98, arg99, arg100, arg101, arg102, arg103, arg104, arg105, arg106, arg107, arg108, arg109, arg110, arg111, arg112, arg113, arg114, arg115, arg116, arg117, arg118, arg119, arg120, arg121, arg122, arg123, arg124, arg125, arg126, arg127, arg128, arg129, arg130, arg131, arg132, arg133, arg134, arg135, arg136, arg137, arg138, arg139, arg140, arg141, arg142, arg143, arg144, arg145, arg146, arg147, arg148, arg149, arg150, arg151, arg152, arg153, arg154, arg155, arg156, arg157, arg158, arg159, arg160, arg161, arg162, arg163, arg164, arg165, arg166, arg167, arg168, arg169, arg170, arg171, arg172, arg173, arg174, arg175, arg176, arg177, arg178, arg179, arg180, arg181, arg182, arg183, arg184, arg185, arg186, arg187, arg188, arg189, arg190, arg191, arg192, arg193, arg194, arg195, arg196, arg197, arg198, arg199, arg200, arg201, arg202, arg203, arg204, arg205, arg206, arg207, arg208, arg209, arg210, arg211, arg212, arg213, arg214, arg215, arg216, arg217, arg218, arg219, arg220, arg221, arg222, arg223, arg224, arg225, arg226, arg227, arg228, arg229, arg230, arg231, arg232, arg233, arg234, arg235, arg236, arg237, arg238, arg239, arg240, arg241, arg242, arg243, arg244, arg245, arg246, arg247, arg248, arg249, arg250, arg251, arg252, arg253, arg254, arg255, arg256, arg257, arg258, arg259, arg260, arg261, arg262, arg263, arg264, arg265, arg266, arg267, arg268, arg269, arg270, arg271, arg272, arg273, arg274, arg275, arg276, arg277, arg278, arg279, arg280, arg281, arg282, arg283, arg284, arg285, arg286, arg287, arg288, arg289, arg290, arg291, arg292, arg293, arg294, arg295, arg296, arg297, arg298, arg299, arg300, arg301, arg302, arg303, arg304, arg305, arg306, arg307, arg308, arg309, arg310, arg311, arg312, arg313, arg314, arg315, arg316, arg317, arg318, arg319, arg320, arg321, arg322, arg323, arg324, arg325, arg326, arg327, arg328, arg329, arg330, arg331, arg332, arg333, arg334, arg335, arg336, arg337, arg338, arg339, arg340, arg341, arg342):
            self.result = (
                unranked_memref_to_numpy(arg0, np.float32),
                unranked_memref_to_numpy(arg1, np.float32),
                unranked_memref_to_numpy(arg2, np.float32),
                unranked_memref_to_numpy(arg3, np.float32),
                unranked_memref_to_numpy(arg4, np.float32),
                unranked_memref_to_numpy(arg5, np.float32),
                unranked_memref_to_numpy(arg6, np.float32),
                unranked_memref_to_numpy(arg7, np.float32),
                unranked_memref_to_numpy(arg8, np.float32),
                unranked_memref_to_numpy(arg9, np.float32),
                unranked_memref_to_numpy(arg10, np.float32),
                unranked_memref_to_numpy(arg11, np.float32),
                unranked_memref_to_numpy(arg12, np.float32),
                unranked_memref_to_numpy(arg13, np.float32),
                unranked_memref_to_numpy(arg14, np.float32),
                unranked_memref_to_numpy(arg15, np.float32),
                unranked_memref_to_numpy(arg16, np.float32),
                unranked_memref_to_numpy(arg17, np.float32),
                unranked_memref_to_numpy(arg18, np.float32),
                unranked_memref_to_numpy(arg19, np.float32),
                unranked_memref_to_numpy(arg20, np.float32),
                unranked_memref_to_numpy(arg21, np.float32),
                unranked_memref_to_numpy(arg22, np.float32),
                unranked_memref_to_numpy(arg23, np.float32),
                unranked_memref_to_numpy(arg24, np.float32),
                unranked_memref_to_numpy(arg25, np.float32),
                unranked_memref_to_numpy(arg26, np.float32),
                unranked_memref_to_numpy(arg27, np.float32),
                unranked_memref_to_numpy(arg28, np.float32),
                unranked_memref_to_numpy(arg29, np.float32),
                unranked_memref_to_numpy(arg30, np.float32),
                unranked_memref_to_numpy(arg31, np.float32),
                unranked_memref_to_numpy(arg32, np.float32),
                unranked_memref_to_numpy(arg33, np.float32),
                unranked_memref_to_numpy(arg34, np.float32),
                unranked_memref_to_numpy(arg35, np.float32),
                unranked_memref_to_numpy(arg36, np.float32),
                unranked_memref_to_numpy(arg37, np.float32),
                unranked_memref_to_numpy(arg38, np.float32),
                unranked_memref_to_numpy(arg39, np.float32),
                unranked_memref_to_numpy(arg40, np.float32),
                unranked_memref_to_numpy(arg41, np.float32),
                unranked_memref_to_numpy(arg42, np.float32),
                unranked_memref_to_numpy(arg43, np.float32),
                unranked_memref_to_numpy(arg44, np.float32),
                unranked_memref_to_numpy(arg45, np.float32),
                unranked_memref_to_numpy(arg46, np.float32),
                unranked_memref_to_numpy(arg47, np.float32),
                unranked_memref_to_numpy(arg48, np.float32),
                unranked_memref_to_numpy(arg49, np.float32),
                unranked_memref_to_numpy(arg50, np.float32),
                unranked_memref_to_numpy(arg51, np.float32),
                unranked_memref_to_numpy(arg52, np.float32),
                unranked_memref_to_numpy(arg53, np.float32),
                unranked_memref_to_numpy(arg54, np.int64),
                unranked_memref_to_numpy(arg55, np.int64),
                unranked_memref_to_numpy(arg56, np.int64),
                unranked_memref_to_numpy(arg57, np.float32),
                unranked_memref_to_numpy(arg58, np.float32),
                unranked_memref_to_numpy(arg59, np.float32),
                unranked_memref_to_numpy(arg60, np.float32),
                unranked_memref_to_numpy(arg61, np.float32),
                unranked_memref_to_numpy(arg62, np.float32),
                unranked_memref_to_numpy(arg63, np.float32),
                unranked_memref_to_numpy(arg64, np.float32),
                unranked_memref_to_numpy(arg65, np.float32),
                unranked_memref_to_numpy(arg66, np.float32),
                unranked_memref_to_numpy(arg67, np.float32),
                unranked_memref_to_numpy(arg68, np.float32),
                unranked_memref_to_numpy(arg69, np.float32),
                unranked_memref_to_numpy(arg70, np.float32),
                unranked_memref_to_numpy(arg71, np.float32),
                unranked_memref_to_numpy(arg72, np.float32),
                unranked_memref_to_numpy(arg73, np.float32),
                unranked_memref_to_numpy(arg74, np.float32),
                unranked_memref_to_numpy(arg75, np.float32),
                unranked_memref_to_numpy(arg76, np.float32),
                unranked_memref_to_numpy(arg77, np.float32),
                unranked_memref_to_numpy(arg78, np.float32),
                unranked_memref_to_numpy(arg79, np.float32),
                unranked_memref_to_numpy(arg80, np.float32),
                unranked_memref_to_numpy(arg81, np.float32),
                unranked_memref_to_numpy(arg82, np.float32),
                unranked_memref_to_numpy(arg83, np.float32),
                unranked_memref_to_numpy(arg84, np.float32),
                unranked_memref_to_numpy(arg85, np.float32),
                unranked_memref_to_numpy(arg86, np.float32),
                unranked_memref_to_numpy(arg87, np.float32),
                unranked_memref_to_numpy(arg88, np.float32),
                unranked_memref_to_numpy(arg89, np.float32),
                unranked_memref_to_numpy(arg90, np.float32),
                unranked_memref_to_numpy(arg91, np.float32),
                unranked_memref_to_numpy(arg92, np.float32),
                unranked_memref_to_numpy(arg93, np.float32),
                unranked_memref_to_numpy(arg94, np.float32),
                unranked_memref_to_numpy(arg95, np.float32),
                unranked_memref_to_numpy(arg96, np.float32),
                unranked_memref_to_numpy(arg97, np.float32),
                unranked_memref_to_numpy(arg98, np.float32),
                unranked_memref_to_numpy(arg99, np.float32),
                unranked_memref_to_numpy(arg100, np.float32),
                unranked_memref_to_numpy(arg101, np.float32),
                unranked_memref_to_numpy(arg102, np.float32),
                unranked_memref_to_numpy(arg103, np.float32),
                unranked_memref_to_numpy(arg104, np.float32),
                unranked_memref_to_numpy(arg105, np.float32),
                unranked_memref_to_numpy(arg106, np.float32),
                unranked_memref_to_numpy(arg107, np.float32),
                unranked_memref_to_numpy(arg108, np.float32),
                unranked_memref_to_numpy(arg109, np.float32),
                unranked_memref_to_numpy(arg110, np.float32),
                unranked_memref_to_numpy(arg111, np.float32),
                unranked_memref_to_numpy(arg112, np.float32),
                unranked_memref_to_numpy(arg113, np.float32),
                unranked_memref_to_numpy(arg114, np.float32),
                unranked_memref_to_numpy(arg115, np.float32),
                unranked_memref_to_numpy(arg116, np.float32),
                unranked_memref_to_numpy(arg117, np.float32),
                unranked_memref_to_numpy(arg118, np.float32),
                unranked_memref_to_numpy(arg119, np.float32),
                unranked_memref_to_numpy(arg120, np.float32),
                unranked_memref_to_numpy(arg121, np.float32),
                unranked_memref_to_numpy(arg122, np.float32),
                unranked_memref_to_numpy(arg123, np.float32),
                unranked_memref_to_numpy(arg124, np.float32),
                unranked_memref_to_numpy(arg125, np.float32),
                unranked_memref_to_numpy(arg126, np.float32),
                unranked_memref_to_numpy(arg127, np.float32),
                unranked_memref_to_numpy(arg128, np.float32),
                unranked_memref_to_numpy(arg129, np.float32),
                unranked_memref_to_numpy(arg130, np.float32),
                unranked_memref_to_numpy(arg131, np.float32),
                unranked_memref_to_numpy(arg132, np.float32),
                unranked_memref_to_numpy(arg133, np.float32),
                unranked_memref_to_numpy(arg134, np.float32),
                unranked_memref_to_numpy(arg135, np.float32),
                unranked_memref_to_numpy(arg136, np.float32),
                unranked_memref_to_numpy(arg137, np.float32),
                unranked_memref_to_numpy(arg138, np.float32),
                unranked_memref_to_numpy(arg139, np.float32),
                unranked_memref_to_numpy(arg140, np.float32),
                unranked_memref_to_numpy(arg141, np.float32),
                unranked_memref_to_numpy(arg142, np.float32),
                unranked_memref_to_numpy(arg143, np.float32),
                unranked_memref_to_numpy(arg144, np.float32),
                unranked_memref_to_numpy(arg145, np.float32),
                unranked_memref_to_numpy(arg146, np.float32),
                unranked_memref_to_numpy(arg147, np.float32),
                unranked_memref_to_numpy(arg148, np.float32),
                unranked_memref_to_numpy(arg149, np.float32),
                unranked_memref_to_numpy(arg150, np.float32),
                unranked_memref_to_numpy(arg151, np.float32),
                unranked_memref_to_numpy(arg152, np.float32),
                unranked_memref_to_numpy(arg153, np.float32),
                unranked_memref_to_numpy(arg154, np.float32),
                unranked_memref_to_numpy(arg155, np.float32),
                unranked_memref_to_numpy(arg156, np.float32),
                unranked_memref_to_numpy(arg157, np.float32),
                unranked_memref_to_numpy(arg158, np.float32),
                unranked_memref_to_numpy(arg159, np.float32),
                unranked_memref_to_numpy(arg160, np.float32),
                unranked_memref_to_numpy(arg161, np.float32),
                unranked_memref_to_numpy(arg162, np.float32),
                unranked_memref_to_numpy(arg163, np.float32),
                unranked_memref_to_numpy(arg164, np.float32),
                unranked_memref_to_numpy(arg165, np.float32),
                unranked_memref_to_numpy(arg166, np.float32),
                unranked_memref_to_numpy(arg167, np.float32),
                unranked_memref_to_numpy(arg168, np.float32),
                unranked_memref_to_numpy(arg169, np.float32),
                unranked_memref_to_numpy(arg170, np.float32),
                unranked_memref_to_numpy(arg171, np.float32),
                unranked_memref_to_numpy(arg172, np.float32),
                unranked_memref_to_numpy(arg173, np.float32),
                unranked_memref_to_numpy(arg174, np.float32),
                unranked_memref_to_numpy(arg175, np.float32),
                unranked_memref_to_numpy(arg176, np.float32),
                unranked_memref_to_numpy(arg177, np.float32),
                unranked_memref_to_numpy(arg178, np.float32),
                unranked_memref_to_numpy(arg179, np.float32),
                unranked_memref_to_numpy(arg180, np.float32),
                unranked_memref_to_numpy(arg181, np.float32),
                unranked_memref_to_numpy(arg182, np.float32),
                unranked_memref_to_numpy(arg183, np.float32),
                unranked_memref_to_numpy(arg184, np.float32),
                unranked_memref_to_numpy(arg185, np.float32),
                unranked_memref_to_numpy(arg186, np.float32),
                unranked_memref_to_numpy(arg187, np.float32),
                unranked_memref_to_numpy(arg188, np.float32),
                unranked_memref_to_numpy(arg189, np.float32),
                unranked_memref_to_numpy(arg190, np.float32),
                unranked_memref_to_numpy(arg191, np.float32),
                unranked_memref_to_numpy(arg192, np.float32),
                unranked_memref_to_numpy(arg193, np.float32),
                unranked_memref_to_numpy(arg194, np.float32),
                unranked_memref_to_numpy(arg195, np.float32),
                unranked_memref_to_numpy(arg196, np.float32),
                unranked_memref_to_numpy(arg197, np.float32),
                unranked_memref_to_numpy(arg198, np.float32),
                unranked_memref_to_numpy(arg199, np.float32),
                unranked_memref_to_numpy(arg200, np.float32),
                unranked_memref_to_numpy(arg201, np.float32),
                unranked_memref_to_numpy(arg202, np.float32),
                unranked_memref_to_numpy(arg203, np.float32),
                unranked_memref_to_numpy(arg204, np.float32),
                unranked_memref_to_numpy(arg205, np.float32),
                unranked_memref_to_numpy(arg206, np.float32),
                unranked_memref_to_numpy(arg207, np.float32),
                unranked_memref_to_numpy(arg208, np.float32),
                unranked_memref_to_numpy(arg209, np.float32),
                unranked_memref_to_numpy(arg210, np.float32),
                unranked_memref_to_numpy(arg211, np.float32),
                unranked_memref_to_numpy(arg212, np.float32),
                unranked_memref_to_numpy(arg213, np.float32),
                unranked_memref_to_numpy(arg214, np.float32),
                unranked_memref_to_numpy(arg215, np.float32),
                unranked_memref_to_numpy(arg216, np.float32),
                unranked_memref_to_numpy(arg217, np.float32),
                unranked_memref_to_numpy(arg218, np.float32),
                unranked_memref_to_numpy(arg219, np.float32),
                unranked_memref_to_numpy(arg220, np.float32),
                unranked_memref_to_numpy(arg221, np.float32),
                unranked_memref_to_numpy(arg222, np.float32),
                unranked_memref_to_numpy(arg223, np.float32),
                unranked_memref_to_numpy(arg224, np.float32),
                unranked_memref_to_numpy(arg225, np.float32),
                unranked_memref_to_numpy(arg226, np.float32),
                unranked_memref_to_numpy(arg227, np.float32),
                unranked_memref_to_numpy(arg228, np.float32),
                unranked_memref_to_numpy(arg229, np.float32),
                unranked_memref_to_numpy(arg230, np.float32),
                unranked_memref_to_numpy(arg231, np.float32),
                unranked_memref_to_numpy(arg232, np.float32),
                unranked_memref_to_numpy(arg233, np.float32),
                unranked_memref_to_numpy(arg234, np.float32),
                unranked_memref_to_numpy(arg235, np.float32),
                unranked_memref_to_numpy(arg236, np.float32),
                unranked_memref_to_numpy(arg237, np.float32),
                unranked_memref_to_numpy(arg238, np.float32),
                unranked_memref_to_numpy(arg239, np.float32),
                unranked_memref_to_numpy(arg240, np.float32),
                unranked_memref_to_numpy(arg241, np.float32),
                unranked_memref_to_numpy(arg242, np.float32),
                unranked_memref_to_numpy(arg243, np.float32),
                unranked_memref_to_numpy(arg244, np.float32),
                unranked_memref_to_numpy(arg245, np.float32),
                unranked_memref_to_numpy(arg246, np.float32),
                unranked_memref_to_numpy(arg247, np.float32),
                unranked_memref_to_numpy(arg248, np.float32),
                unranked_memref_to_numpy(arg249, np.float32),
                unranked_memref_to_numpy(arg250, np.float32),
                unranked_memref_to_numpy(arg251, np.float32),
                unranked_memref_to_numpy(arg252, np.float32),
                unranked_memref_to_numpy(arg253, np.float32),
                unranked_memref_to_numpy(arg254, np.float32),
                unranked_memref_to_numpy(arg255, np.float32),
                unranked_memref_to_numpy(arg256, np.float32),
                unranked_memref_to_numpy(arg257, np.float32),
                unranked_memref_to_numpy(arg258, np.float32),
                unranked_memref_to_numpy(arg259, np.float32),
                unranked_memref_to_numpy(arg260, np.float32),
                unranked_memref_to_numpy(arg261, np.float32),
                unranked_memref_to_numpy(arg262, np.float32),
                unranked_memref_to_numpy(arg263, np.float32),
                unranked_memref_to_numpy(arg264, np.float32),
                unranked_memref_to_numpy(arg265, np.float32),
                unranked_memref_to_numpy(arg266, np.float32),
                unranked_memref_to_numpy(arg267, np.float32),
                unranked_memref_to_numpy(arg268, np.float32),
                unranked_memref_to_numpy(arg269, np.float32),
                unranked_memref_to_numpy(arg270, np.float32),
                unranked_memref_to_numpy(arg271, np.float32),
                unranked_memref_to_numpy(arg272, np.float32),
                unranked_memref_to_numpy(arg273, np.float32),
                unranked_memref_to_numpy(arg274, np.float32),
                unranked_memref_to_numpy(arg275, np.float32),
                unranked_memref_to_numpy(arg276, np.float32),
                unranked_memref_to_numpy(arg277, np.float32),
                unranked_memref_to_numpy(arg278, np.float32),
                unranked_memref_to_numpy(arg279, np.float32),
                unranked_memref_to_numpy(arg280, np.float32),
                unranked_memref_to_numpy(arg281, np.float32),
                unranked_memref_to_numpy(arg282, np.float32),
                unranked_memref_to_numpy(arg283, np.float32),
                unranked_memref_to_numpy(arg284, np.float32),
                unranked_memref_to_numpy(arg285, np.float32),
                unranked_memref_to_numpy(arg286, np.float32),
                unranked_memref_to_numpy(arg287, np.float32),
                unranked_memref_to_numpy(arg288, np.float32),
                unranked_memref_to_numpy(arg289, np.float32),
                unranked_memref_to_numpy(arg290, np.float32),
                unranked_memref_to_numpy(arg291, np.float32),
                unranked_memref_to_numpy(arg292, np.float32),
                unranked_memref_to_numpy(arg293, np.float32),
                unranked_memref_to_numpy(arg294, np.float32),
                unranked_memref_to_numpy(arg295, np.float32),
                unranked_memref_to_numpy(arg296, np.float32),
                unranked_memref_to_numpy(arg297, np.float32),
                unranked_memref_to_numpy(arg298, np.float32),
                unranked_memref_to_numpy(arg299, np.float32),
                unranked_memref_to_numpy(arg300, np.float32),
                unranked_memref_to_numpy(arg301, np.float32),
                unranked_memref_to_numpy(arg302, np.float32),
                unranked_memref_to_numpy(arg303, np.float32),
                unranked_memref_to_numpy(arg304, np.float32),
                unranked_memref_to_numpy(arg305, np.float32),
                unranked_memref_to_numpy(arg306, np.float32),
                unranked_memref_to_numpy(arg307, np.float32),
                unranked_memref_to_numpy(arg308, np.float32),
                unranked_memref_to_numpy(arg309, np.float32),
                unranked_memref_to_numpy(arg310, np.float32),
                unranked_memref_to_numpy(arg311, np.float32),
                unranked_memref_to_numpy(arg312, np.float32),
                unranked_memref_to_numpy(arg313, np.float32),
                unranked_memref_to_numpy(arg314, np.float32),
                unranked_memref_to_numpy(arg315, np.float32),
                unranked_memref_to_numpy(arg316, np.float32),
                unranked_memref_to_numpy(arg317, np.float32),
                unranked_memref_to_numpy(arg318, np.float32),
                unranked_memref_to_numpy(arg319, np.float32),
                unranked_memref_to_numpy(arg320, np.float32),
                unranked_memref_to_numpy(arg321, np.float32),
                unranked_memref_to_numpy(arg322, np.float32),
                unranked_memref_to_numpy(arg323, np.float32),
                unranked_memref_to_numpy(arg324, np.float32),
                unranked_memref_to_numpy(arg325, np.float32),
                unranked_memref_to_numpy(arg326, np.float32),
                unranked_memref_to_numpy(arg327, np.float32),
                unranked_memref_to_numpy(arg328, np.float32),
                unranked_memref_to_numpy(arg329, np.float32),
                unranked_memref_to_numpy(arg330, np.float32),
                unranked_memref_to_numpy(arg331, np.float32),
                unranked_memref_to_numpy(arg332, np.float32),
                unranked_memref_to_numpy(arg333, np.float32),
                unranked_memref_to_numpy(arg334, np.float32),
                unranked_memref_to_numpy(arg335, np.float32),
                unranked_memref_to_numpy(arg336, np.float32),
                unranked_memref_to_numpy(arg337, np.float32),
                unranked_memref_to_numpy(arg338, np.float32),
                unranked_memref_to_numpy(arg339, np.float32),
                unranked_memref_to_numpy(arg340, np.float32),
                unranked_memref_to_numpy(arg341, np.int64),
                unranked_memref_to_numpy(arg342, np.float32)
            )

        self.ee.register_runtime("refbackend_consume_func_return_mri1",
                                 consume_return_mri1)

        self.ee.register_runtime("refbackend_consume_func_return_mri32",
                                 consume_return_mri32)

        self.ee.register_runtime("refbackend_consume_func_return_mri64",
                                 consume_return_mri64)

        self.ee.register_runtime("refbackend_consume_func_return_mrf32",
                                 consume_return_mrf32)

        self.ee.register_runtime("refbackend_consume_func_return_mrf64",
                                 consume_return_mrf64)

        self.ee.register_runtime("refbackend_consume_func_return_i1",
                                 consume_return_i1)

        self.ee.register_runtime("refbackend_consume_func_return_i64",
                                 consume_return_i64)

        self.ee.register_runtime("refbackend_consume_func_return_f32",
                                 consume_return_f32)

        self.ee.register_runtime("refbackend_consume_func_return_f64",
                                 consume_return_f64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mri64",
            consume_return_mrf32_mri64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mrf32",
            consume_return_mrf32_mrf32)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf64_mrf64",
            consume_return_mrf64_mrf64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mrf32_mrf32",
            consume_return_mrf32_mrf32_mrf32)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mri64_mri64_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mrf32",
            consume_return_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mri64_mri64_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mrf32_mri64_mrf32)

    def __getattr__(self, function_name: str):
        def invoke(*args):
            ffi_args = []
            for arg in args:
                checkArgTypeIsSupported(arg.dtype)
                ffi_args.append(
                    ctypes.pointer(
                        ctypes.pointer(get_unranked_memref_descriptor(arg))))

            self.ee.invoke(function_name, *ffi_args)
            result = self.result
            assert result is not None, "Invocation didn't produce a result"
            self.result = None
            return result

        return invoke


LOWERING_PIPELINE = ",".join([
    "builtin.func(refback-generalize-tensor-pad)",
    # Bufferize.
    "builtin.func(scf-bufferize)",
    "builtin.func(tm-tensor-bufferize)",
    "builtin.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "builtin.func(tensor-bufferize)",
    "builtin.func(finalizing-bufferize)",
    # Munge to make it ExecutionEngine compatible.
    # Specifically, we rewrite calling convention boundaries to be in terms
    # of unranked memref, and we rewrite the return to actually be a
    # callback that consumes the return (the final munged function always
    # returns void at the C level -- we get the return value by providing the
    # callback).
    "refback-munge-calling-conventions",
    # Insert global variable and instruction sequence for getting the next
    # global seed used in stateful rng.
    "refback-insert-rng-globals",
    # Lower to LLVM
    "builtin.func(tm-tensor-to-loops)",
    "builtin.func(refback-munge-memref-copy)",
    "builtin.func(convert-linalg-to-loops)",
    "builtin.func(lower-affine)",
    "convert-scf-to-cf",
    "builtin.func(refback-expand-ops-for-llvm)",
    "builtin.func(arith-expand)",
    "builtin.func(convert-math-to-llvm)",
    "convert-linalg-to-llvm",
    "convert-memref-to-llvm",
    "builtin.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
])


class RefBackendLinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""
    def __init__(self):
        super().__init__()

    def compile(self, imported_module: Module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        run_pipeline_with_repro_report(
            imported_module, LOWERING_PIPELINE,
            "Lowering Linalg-on-Tensors IR to LLVM with RefBackend")
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        return RefBackendInvoker(module)
