import cv2
import os
from matplotlib import pyplot as plt
from ultralytics import YOLO
import json
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

video_root = "hope-dataset/hope_video"
obj_dir = "hope-dataset/meshes/eval"
img_out_dir = "hope-yolo/images/train"
lbl_out_dir = "hope-yolo/labels/train"

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(lbl_out_dir, exist_ok=True)

CLASSES = [
    "AlphabetSoup", "BBQSauce", "Butter", "Cherries", "ChocolatePudding", "Cookies",
    "Corn", "CreamCheese", "GranolaBars", "GreenBeans", "Ketchup", "MacaroniAndCheese",
    "Mayo", "Milk", "Mushrooms", "Mustard", "OrangeJuice", "Parmesan", "Peaches",
    "PeasAndCarrots", "Pineapple", "Popcorn", "Raisins", "SaladDressing", "Spaghetti",
    "TomatoSauce", "Tuna", "Yogurt"
]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

def load_obj_vertices(obj_path):
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.strip().split()[1:4])))
    return np.array(verts)

def project_points(points_3d, pose, K):
    n = points_3d.shape[0]
    points_3d_h = np.hstack([points_3d, np.ones((n, 1))])
    proj_matrix = K @ pose[:3]
    points_2d_h = proj_matrix @ points_3d_h.T
    points_2d = points_2d_h[:2] / points_2d_h[2]
    return points_2d.T

def process_video_scene(scene_path):
    frames = sorted([f for f in os.listdir(scene_path) if f.endswith("_rgb.jpg")])
    frames = frames[::10]  # every 10th frame

    for rgb_file in frames:
        img_id = rgb_file.split("_")[0]
        json_path = os.path.join(scene_path, f"{img_id}.json")
        img_path = os.path.join(scene_path, rgb_file)

        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            K = np.array(data["camera"]["intrinsics"])
            h, w = data["camera"]["height"], data["camera"]["width"]

            label_lines = []
            for obj in data["objects"]:
                class_name = obj["class"]
                if class_name not in CLASS_TO_ID:
                    continue

                obj_path = os.path.join(obj_dir, f"{class_name}.obj")
                if not os.path.exists(obj_path):
                    continue

                verts = load_obj_vertices(obj_path)
                pose = np.array(obj["pose"])
                proj_2d = project_points(verts, pose, K)

                x_min = np.clip(np.min(proj_2d[:, 0]), 0, w)
                x_max = np.clip(np.max(proj_2d[:, 0]), 0, w)
                y_min = np.clip(np.min(proj_2d[:, 1]), 0, h)
                y_max = np.clip(np.max(proj_2d[:, 1]), 0, h)

                if x_max - x_min < 1 or y_max - y_min < 1:
                    continue

                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                box_width = (x_max - x_min) / w
                box_height = (y_max - y_min) / h

                class_id = CLASS_TO_ID[class_name]
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

            # Save .jpg and .txt with consistent name
            out_name = f"{Path(scene_path).name}_{img_id}"
            out_img_path = os.path.join(img_out_dir, f"{out_name}.jpg")
            out_lbl_path = os.path.join(lbl_out_dir, f"{out_name}.txt")

            shutil.copy(img_path, out_img_path)
            with open(out_lbl_path, 'w') as f:
                f.write("\n".join(label_lines))

        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

# Run for all scenes 
video_scenes = [os.path.join(video_root, d) for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))]
print(f"Processing {len(video_scenes)} video scenes...")

for scene_path in tqdm(video_scenes):
    process_video_scene(scene_path)

# print("Images:", len(os.listdir("hope-yolo/images/train")))
# print("Labels:", len(os.listdir("hope-yolo/labels/train")))

model = YOLO("yolov8s.pt")
model.train(data="hope.yaml", epochs=50, imgsz=640)

metrics = model.val(data="hope.yaml")

results = model.predict(source="hope-yolo/images/val", conf=0.3, save=True)