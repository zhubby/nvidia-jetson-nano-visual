# Models

Place YOLO TensorRT engine files here.

Expected default:

```text
models/yolov8n_fp16.engine
```

TensorRT engines are hardware, TensorRT, CUDA, and JetPack specific. Generate the engine on the Jetson Nano or inside a matching L4T/JetPack container. Do not commit `.engine`, `.onnx`, or `.pt` binaries to git.

For a first deployment, use a COCO-pretrained nano model at 640 image size and FP16 precision. If FPS is below target, try a smaller input size such as 416 or 320.
