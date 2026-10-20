# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-edge-litert", "numpy", "pillow"]
# ///
"""Classify a parrot on the Coral USB Edge TPU with MobileNet V2.

Usage: uv run scripts/coral_smoke_test.py [path/to/libedgetpu.so.1]
"""
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
LIB = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "out/direct/k8/libedgetpu.so.1")
CACHE = Path.home() / ".cache/coral_smoke_test"
BASE = "https://github.com/google-coral/test_data/raw/master/"
FILES = ["mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", "inat_bird_labels.txt", "parrot.jpg"]


def fetch(name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(BASE + name, path)
    return path


def make_delegate():
    # The first load after plug-in uploads firmware and the device re-enumerates,
    # which makes that first attempt fail. Retry once.
    try:
        return load_delegate(LIB)
    except ValueError:
        time.sleep(1)
        return load_delegate(LIB)


model, labels, image = (fetch(f) for f in FILES)
labels = labels.read_text().splitlines()

interpreter = Interpreter(model_path=str(model), experimental_delegates=[make_delegate()])
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

_, h, w, _ = inp["shape"]
x = np.asarray(Image.open(image).convert("RGB").resize((w, h)), dtype=np.uint8)[None]
interpreter.set_tensor(inp["index"], x)

# The first invoke also uploads the model weights to the TPU, so time a few runs.
times = []
for _ in range(10):
    t0 = time.perf_counter()
    interpreter.invoke()
    times.append((time.perf_counter() - t0) * 1000)

scores = interpreter.get_tensor(out["index"])[0]
scale, zero = out["quantization"]
probs = (scores.astype(np.float32) - zero) * scale
print(f"\nFirst invoke: {times[0]:.1f} ms, then median {np.median(times[1:]):.1f} ms")
for i in np.argsort(probs)[::-1][:3]:
    print(f"  {probs[i]:.3f}  {labels[i]}")
