"""
COCO keypoints JSON -> YOLO Pose label files (.txt, one per image).

Output line per object:
    class cx cy bw bh  kp1x kp1y vis1  kp2x kp2y vis2  kp3x kp3y vis3
All coordinates are normalized to 0-1 by image width/height.
Visibility: 0 = not labeled, 1 = labeled but occluded, 2 = visible.

Usage:
    1. Set COCO_PATH and OUTPUT_DIR below (or pass them as arguments)
    2. python coco_to_yolo_pose.py
       python coco_to_yolo_pose.py person_keypoints.json labels
"""
import json
import os
import sys
from collections import defaultdict

COCO_PATH = "person_keypoints.json"   # your COCO JSON file
OUTPUT_DIR = "labels"                 # where the .txt files are saved
CLASS_ID = 0                          # YOLO class id for "bowler"
NUM_KEYPOINTS = 3                     # shoulder, elbow, wrist
WRITE_EMPTY_FILES = True              # empty .txt for images without annotations (background)


def clamp01(v):
    return max(0.0, min(1.0, v))


def convert(coco_path, output_dir):
    with open(coco_path) as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    n_files, n_objects = 0, 0
    sample = None

    for img in coco["images"]:
        w, h = img["width"], img["height"]
        lines = []

        for ann in anns_by_image.get(img["id"], []):
            # COCO bbox = [x_top_left, y_top_left, width, height] in pixels
            bx, by, bw, bh = ann["bbox"]
            cx = clamp01((bx + bw / 2) / w)
            cy = clamp01((by + bh / 2) / h)
            nw = clamp01(bw / w)
            nh = clamp01(bh / h)

            parts = [str(CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]

            kps = ann["keypoints"]
            for i in range(NUM_KEYPOINTS):
                x, y, v = kps[3 * i], kps[3 * i + 1], int(kps[3 * i + 2])
                if v == 0:
                    parts += ["0.000000", "0.000000", "0"]   # missing joint
                else:
                    parts += [f"{clamp01(x / w):.6f}", f"{clamp01(y / h):.6f}", str(v)]

            lines.append(" ".join(parts))
            n_objects += 1

        if not lines and not WRITE_EMPTY_FILES:
            continue

        stem = os.path.splitext(os.path.basename(img["file_name"]))[0]
        with open(os.path.join(output_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1

        if lines and sample is None:
            sample = (stem + ".txt", lines[0])

    print(f"Saved {n_files} label files ({n_objects} objects) to '{output_dir}/'")
    if sample:
        print(f"Sample: {sample[0]}\n{sample[1]}")


if __name__ == "__main__":
    coco_path = sys.argv[1] if len(sys.argv) > 1 else COCO_PATH
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    convert(coco_path, output_dir)