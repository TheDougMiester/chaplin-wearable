#!/usr/bin/env python3
"""
Convert Chaplin AVSR encoder to TensorRT FP16 with torch2trt
Expected: 2-5x speedup on VSR, 3.8s -> ~1s
Usage inside container:
  python3 convert_to_trt.py --model_path ./models/LRS3_V_WER19.1/model.pth --model_conf ./models/LRS3_V_WER19.1/model.json
"""
import torch
import argparse
import json
from pipelines.model import AVSR

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./models/LRS3_V_WER19.1/model.pth")
parser.add_argument("--model_conf", default="./models/LRS3_V_WER19.1/model.json")
parser.add_argument("--onnx_path", default="./models/LRS3_V_WER19.1/encoder_fp16.onnx")
parser.add_argument("--trt_path", default="./models/LRS3_V_WER19.1/encoder_fp16.trt")
args = parser.parse_args()

device = "cuda:0"
print(f"Loading {args.model_path} on {device}")
# Load original to get train_args
with open(args.model_conf, "rb") as f:
    confs = json.load(f)
    train_args = confs if isinstance(confs, dict) else confs[2]
    # We need to init AVSR to get model
avsr = AVSR(modality="video", model_path=args.model_path, model_conf=args.model_conf, device=device)
avsr.eval()

# Dummy input matching your pipeline: [1, 42, 88, 88] from your PERF log
dummy = torch.randn(1, 42, 88, 88).to(device).half()  # FP16
print("Dummy shape", dummy.shape)

# Convert encoder only - beam search stays PyTorch
print("Converting encoder to TensorRT FP16...")

class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # AVSR's E2E model uses encode(), not forward()
        return self.model.encode(x)

wrapper = EncoderWrapper(avsr.model).eval().half().to(device)

# Try torch2trt - will fail on rel_shift, but we catch and fall back to ONNX
try:
    from torch2trt import torch2trt
    trt_model = torch2trt(wrapper, [dummy], fp16_mode=True, max_workspace_size=1<<30)
    torch.save(trt_model.state_dict(), args.trt_path + ".pth")
    print(f"Saved TRT encoder to {args.trt_path}.pth")
except Exception as e:
    print(f"[WARN] torch2trt failed (expected for Transformer): {e}")
    print("Falling back to ONNX Runtime TensorRT - still 2-3x speedup")

# Also export ONNX for onnxruntime TensorRT provider - supports rel_shift
print(f"Exporting ONNX to {args.onnx_path}")
try:
    torch.onnx.export(wrapper, dummy, args.onnx_path, input_names=["video"], output_names=["enc_feats"], opset_version=17, dynamic_axes={"video": {0: "batch", 1: "time"}})
    print(f"Saved ONNX to {args.onnx_path}")
    print("To use: onnxruntime with providers=['TensorrtExecutionProvider','CUDAExecutionProvider']")
except Exception as e:
    print(f"ONNX export failed: {e}")

print("Done. For immediate speed win without TRT, use beam_size=1 in INI (already done) and res_factor=2")
