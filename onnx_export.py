import onnx
from onnx import optimizer
import torch
# from sys import argv
from model import CBDNet as Model
import pdb

if __name__ == "__main__":
    onnx_file = "model.onnx"
    weight_file = "checkpoint/CBDNet.pth"

    # 1. Load model
    print("Loading model ...")
    model = Model()
    map_location = (lambda storage, loc:storage)
    if torch.cuda.is_available():
        map_location = None
    ckpt = torch.load(weight_file, map_location=map_location)
    model.load_state_dict(ckpt)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, onnx_file,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=True,
                    export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"
