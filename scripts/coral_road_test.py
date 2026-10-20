# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-edge-litert", "numpy", "pillow"]
# ///
"""Street-scene test on the Coral USB Edge TPU, running two models side by side:

- DeepLab (Cityscapes): per-pixel road / sidewalk / traffic sign / car / ... mask
- SSDLite MobileDet (COCO): boxes for cars, motorcycles, buses, people, stop signs...

Writes an overlay image (mask + boxes) and prints what was found.

Usage: uv run scripts/coral_road_test.py [image path or URL] [-o overlay.jpg] [--lib libedgetpu.so.1]
"""
import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache/coral_smoke_test"
CORAL = "https://github.com/google-coral/test_data/raw/master/"
SEG_MODEL = "deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite"
DET_MODEL = "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
DET_LABELS = "coco_labels.txt"
# Busy street with motorbikes, cars and an overhead sign. CC BY-SA 4.0, PattayaPatrol (Wikimedia Commons).
DEFAULT_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
    "DZ6_0541_Busy_city_street_scene_with_motorbikes_and_cars_passing_beneath_an_overhead_sign_"
    "that_reads_Central_storefronts_and_colorful_signs_lining_both_sides_of_the_road.jpg/1280px-thumbnail.jpg"
)

# Cityscapes train IDs and their standard colours.
CITYSCAPES = [
    ("road", (128, 64, 128)), ("sidewalk", (244, 35, 232)), ("building", (70, 70, 70)),
    ("wall", (102, 102, 156)), ("fence", (190, 153, 153)), ("pole", (153, 153, 153)),
    ("traffic light", (250, 170, 30)), ("traffic sign", (220, 220, 0)), ("vegetation", (107, 142, 35)),
    ("terrain", (152, 251, 152)), ("sky", (70, 130, 180)), ("person", (220, 20, 60)),
    ("rider", (255, 0, 0)), ("car", (0, 0, 142)), ("truck", (0, 0, 70)), ("bus", (0, 60, 100)),
    ("train", (0, 80, 100)), ("motorcycle", (0, 0, 230)), ("bicycle", (119, 11, 32)),
]
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"}


def fetch(url, name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "coral-road-test/0.1"})
        path.write_bytes(urllib.request.urlopen(req).read())
    return path


def make_interpreter(model, lib):
    # The first load after plug-in uploads firmware and the device re-enumerates,
    # which makes that first attempt fail. Retry once.
    try:
        delegate = load_delegate(lib)
    except ValueError:
        time.sleep(1)
        delegate = load_delegate(lib)
    interpreter = Interpreter(model_path=str(model), experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    return interpreter


def timed_invoke(interpreter, runs=5):
    # The first invoke also uploads the model weights to the TPU.
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    return times[0], float(np.median(times[1:]))


def segment(interpreter, img):
    """Letterbox to the model input (keeping aspect), return a class-id mask at image size."""
    inp = interpreter.get_input_details()[0]
    _, h, w, _ = inp["shape"]
    scale = min(w / img.width, h / img.height)
    rw, rh = int(img.width * scale), int(img.height * scale)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:rh, :rw] = np.asarray(img.resize((rw, rh)))
    interpreter.set_tensor(inp["index"], canvas[None])
    timings = timed_invoke(interpreter)
    mask = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0][:rh, :rw]
    mask = Image.fromarray(mask.astype(np.uint8)).resize(img.size, Image.NEAREST)
    return np.asarray(mask), timings


def detect(interpreter, img, threshold):
    """Return [(label_id, score, (x0, y0, x1, y1) in image pixels)]."""
    inp = interpreter.get_input_details()[0]
    _, h, w, _ = inp["shape"]
    interpreter.set_tensor(inp["index"], np.asarray(img.resize((w, h)), dtype=np.uint8)[None])
    timings = timed_invoke(interpreter)
    boxes, classes, scores, count = (interpreter.get_tensor(o["index"]) for o in interpreter.get_output_details())
    dets = []
    for i in range(int(count[0])):
        if scores[0, i] < threshold:
            continue
        ymin, xmin, ymax, xmax = np.clip(boxes[0, i], 0, 1)
        dets.append((int(classes[0, i]), float(scores[0, i]),
                     (xmin * img.width, ymin * img.height, xmax * img.width, ymax * img.height)))
    return dets, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("-o", "--out", default="road_overlay.jpg")
    parser.add_argument("--lib", default=str(REPO / "out/direct/k8/libedgetpu.so.1"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = Path(args.image) if Path(args.image).exists() else fetch(args.image, "road_" + args.image.rsplit("/", 1)[-1])
    img = Image.open(image).convert("RGB")
    seg = make_interpreter(fetch(CORAL + SEG_MODEL, SEG_MODEL), args.lib)
    det = make_interpreter(fetch(CORAL + DET_MODEL, DET_MODEL), args.lib)
    labels = fetch(CORAL + DET_LABELS, DET_LABELS).read_text().splitlines()

    mask, seg_t = segment(seg, img)
    dets, det_t = detect(det, img, args.threshold)

    print(f"\nSegmentation: first {seg_t[0]:.1f} ms, then median {seg_t[1]:.1f} ms")
    ids, counts = np.unique(mask, return_counts=True)
    for i, c in sorted(zip(ids, counts), key=lambda x: -x[1]):
        if c / mask.size >= 0.005:
            print(f"  {100 * c / mask.size:5.1f}%  {CITYSCAPES[i][0]}")

    print(f"\nDetection: first {det_t[0]:.1f} ms, then median {det_t[1]:.1f} ms")
    names, counts = np.unique([labels[d[0]] for d in dets], return_counts=True)
    for name, c in sorted(zip(names, counts), key=lambda x: -x[1]):
        print(f"  {c:3d} x {name}")

    palette = np.array([c for _, c in CITYSCAPES], dtype=np.uint8)
    overlay = Image.blend(img, Image.fromarray(palette[mask]), 0.45)
    draw = ImageDraw.Draw(overlay)
    for label_id, score, box in dets:
        colour = (255, 255, 0) if labels[label_id] in ROAD_CLASSES else (255, 255, 255)
        draw.rectangle(box, outline=colour, width=3)
        draw.text((box[0] + 4, box[1] + 2), f"{labels[label_id]} {score:.2f}", fill=colour)
    overlay.save(args.out)
    print(f"\nOverlay written to {args.out}")


if __name__ == "__main__":
    main()
