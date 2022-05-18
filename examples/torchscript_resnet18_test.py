"""
Example of taking TorchScript nn Module and compiling it using torch-mlir.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python torchscript_to_linalg.py`.
"""

import torch
import torchvision.models as models
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))


example_inputs = [torch.rand(1,3,224,224, dtype=torch.float32)]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.resnet = models.resnet18()
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


module = MyModule()
script_module = torch.jit.script(module)
_print_title("TorchScript")
print(script_module.graph)

torch_module = torch_mlir.compile(script_module, example_inputs,
                                  output_type=torch_mlir.OutputType.TORCH)
_print_title("Torch-MLIR-backend")
#print(torch_module)

linalg_module = torch_mlir.compile(script_module, example_inputs,
                                   output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

_print_title("Linalg-MLIR")
#print(linalg_module)

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(linalg_module)
jit_module = backend.load(compiled)

_print_title("Running Compiled Graph")
print('Expected output:')
print(script_module.forward(*example_inputs))
print('Output from compiled MLIR:')
numpy_inputs = list(map(lambda x: x.numpy(), example_inputs))
print(torch.tensor(jit_module.forward(*numpy_inputs)))
