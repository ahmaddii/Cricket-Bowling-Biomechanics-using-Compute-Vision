"""
Convert a CVAT "CVAT for images 1.1" annotations.xml (separate Points labels:
shoulder, elbow, wrist) into COCO keypoints JSON.

Usage:
    python cvat_to_coco_keypoints.py annotations.xml person_keypoints.json
"""
import json
import sys
import xml.etree.ElementTree as ET

KEYPOINT_NAMES = ["shoulder", "elbow", "wrist"]  # order in the output
SKELETON = [[1, 2], [2, 3]]                      # shoulder-elbow, elbow-wrist (1-based)
BBOX_PAD = 40                                    # pixels added around the keypoints

# Mislabeled points found in the export. Key = (current label lowercase, "x,y" as in the XML).
# Better to also fix these in CVAT, but this keeps the output correct either way.
FIXES = {
    "clip_frame_000457_t9.14s_000031.jpg": {("elbow", "537.30,196.88"): "wrist"},
    "clip_frame_001354_t27.08s_000080.jpg": {("shoulder", "1030.05,388.00"): "wrist"},
}


def convert(xml_path, out_path):
    root = ET.parse(xml_path).getroot()

    images, annotations = [], []
    ann_id = 1

    for img_id, img in enumerate(root.iter("image"), start=1):
        width, height = int(img.get("width")), int(img.get("height"))
        images.append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": img.get("name"),
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        })

        # group points per person: the i-th occurrence of each label
        # belongs to the i-th person in this frame
        people = []  # list of dicts: {label: (x, y, occluded)}
        counts = {k: 0 for k in KEYPOINT_NAMES}

        for pts in img.findall("points"):
            label = pts.get("label").strip().lower()  # merges "Shoulder"/"shoulder"
            label = FIXES.get(img.get("name"), {}).get((label, pts.get("points")), label)
            if label not in KEYPOINT_NAMES:
                continue
            x, y = [float(v) for v in pts.get("points").split(";")[0].split(",")]
            occluded = pts.get("occluded") == "1"
            idx = counts[label]
            counts[label] += 1
            while len(people) <= idx:
                people.append({})
            people[idx][label] = (x, y, occluded)

        if len(people) > 1:
            print(f"WARNING: {img.get('name')} has duplicate labels -> {len(people)} people. "
                  f"Check for a mislabeled point.")

        for person in people:
            keypoints, xs, ys, n_visible = [], [], [], 0
            for name in KEYPOINT_NAMES:
                if name in person:
                    x, y, occ = person[name]
                    v = 1 if occ else 2  # 1 = labeled but occluded, 2 = visible
                    keypoints += [round(x, 2), round(y, 2), v]
                    xs.append(x)
                    ys.append(y)
                    n_visible += 1
                else:
                    keypoints += [0, 0, 0]  # not annotated

            if n_visible == 0:
                continue

            x0 = max(0, min(xs) - BBOX_PAD)
            y0 = max(0, min(ys) - BBOX_PAD)
            x1 = min(width, max(xs) + BBOX_PAD)
            y1 = min(height, max(ys) + BBOX_PAD)
            w, h = x1 - x0, y1 - y0

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": n_visible,
                "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "",
                 "url": "", "version": "", "year": ""},
        "categories": [{
            "id": 1,
            "name": "bowler",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }],
        "images": images,
        "annotations": annotations,
    }

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"{len(images)} images, {len(annotations)} annotations -> {out_path}")
    missing = sum(1 for a in annotations if a["num_keypoints"] < len(KEYPOINT_NAMES))
    print(f"{missing} annotations have at least one missing keypoint (stored as 0,0,0)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python cvat_to_coco_keypoints.py annotations.xml output.json")
    convert(sys.argv[1], sys.argv[2])
