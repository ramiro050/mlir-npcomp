//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace {
struct PdlExamplePass : public PdlExampleBase<PdlExamplePass> {
  PdlExamplePass() = default;
  PdlExamplePass(StringRef pdlLibrary) { this->pdlLibrary = pdlLibrary.str(); }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }
  void runOnOperation() override {
    if (pdlLibrary.empty())
      return;

    ModuleOp m = getOperation();
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(pdlLibrary);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file " << pdlLibrary << ":"
                   << ec.message() << "\n";
      return signalPassFailure();
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    OwningOpRef<ModuleOp> libraryModule =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &getContext());
    if (!libraryModule) {
      llvm::errs() << "Error can't load file " << pdlLibrary << "\n";
      return signalPassFailure();
    }

    RewritePatternSet patternList(&getContext());
    // Process the pattern module.
    // TODO: how to handle release? I was getting an error of double free when I
    // don't release here.
    ModuleOp libraryModuleReleased(libraryModule.release());
    libraryModuleReleased.getOperation()->remove();
    patternList.add(PDLPatternModule(libraryModuleReleased));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsAndFoldGreedily(m.getBodyRegion(),
                                       std::move(patternList));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createPdlExamplePass(StringRef pdlLibrary) {
  return std::make_unique<PdlExamplePass>(pdlLibrary);
}
