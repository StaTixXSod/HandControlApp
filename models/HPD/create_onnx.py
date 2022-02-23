import torch
from HandPoseDetector import HandPoseModel

model = HandPoseModel(input_features=63, num_classes=4)
model.load_state_dict(
      torch.load("HandPoseDetector.pth",
                 map_location=torch.device("cpu"))
      )
model.eval()
posture = torch.randn(63)

torch.onnx.export(model,
                  posture,
                  "HandPoseDetectorOnnx.onnx",
                  export_params=True,
                  opset_version=10)

