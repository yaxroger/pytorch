/** \brief This file defines freezing Torchscript module API.
 *
 * This API has python-binding and can be invoked directly or as a part of
 * general optimization pipeline.
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

/** \brief Freeze Module, i.e., Assume all atrributes are constants.
 *
 * Clone module, inline into forward method, Propagate attribute,  and optimize
 * cloned module agressively.
 */

namespace torch {
namespace jit {

TORCH_API script::Module FreezeModule(const script::Module& module);

} // namespace jit
} // namespace torch
