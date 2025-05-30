import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

#Config
IMG_DIR = "hope-yolo/images/train"
# IMG_DIR = "hope-yolo/images/val"
OUT_DIR = "hope-yolo/npz_outputs"
MODEL_PATH = "runs/detect/train4/weights/best.pt"
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
CONF_THRESH = 0.3
FPN_LAYERS = [15, 18, 21]  # YOLOv8s FPN (P3, P4, P5)

os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = [
    "AlphabetSoup", "BBQSauce", "Butter", "Cherries", "ChocolatePudding", "Cookies",
    "Corn", "CreamCheese", "GranolaBars", "GreenBeans", "Ketchup", "MacaroniAndCheese",
    "Mayo", "Milk", "Mushrooms", "Mustard", "OrangeJuice", "Parmesan", "Peaches",
    "PeasAndCarrots", "Pineapple", "Popcorn", "Raisins", "SaladDressing", "Spaghetti",
    "TomatoSauce", "Tuna", "Yogurt"
]

# Yolov8 wrapper with hooked FPN fesetures
class YOLOWithFeatures(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.saved_features = []
        self.target_layers = [self.model.model[i] for i in FPN_LAYERS]

        def hook(module, input, output):
            self.saved_features.append(output.detach().cpu())

        for layer in self.target_layers:
            layer.register_forward_hook(hook)

    def forward_with_features(self, img_tensor):
        self.saved_features = []
        preds = self(img_tensor)[0]
        if len(self.saved_features) > 3:
            self.saved_features = self.saved_features[:3]
        return preds, self.saved_features

# loading Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOWithFeatures(MODEL_PATH).to(device)
print(f"Loaded model on {device} | FPN layers: {FPN_LAYERS}")

# processing all images
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])
print(f"Processing {len(img_files)} images in {IMG_DIR}")

for img_name in tqdm(img_files):
    img_path = os.path.join(IMG_DIR, img_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        result, feats = model.forward_with_features(img_tensor)

    boxes, labels, confs = [], [], []

    if result.boxes is not None:
        for b in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = b[:6]
            if conf < CONF_THRESH:
                continue
            cx = (x1 + x2) / 2 / INPUT_SIZE
            cy = (y1 + y2) / 2 / INPUT_SIZE
            w = (x2 - x1) / INPUT_SIZE
            h = (y2 - y1) / INPUT_SIZE
            boxes.append([cx, cy, w, h])
            labels.append(int(cls))
            confs.append(float(conf))

    # preparing feature maps
    feat_np = []
    for f in feats:
        arr = f.numpy()
        if arr.ndim == 3:  # Shape [C, H, W]
            arr = arr[np.newaxis, ...] 
        feat_np.append(arr)

    # to validate feature shapes
    for feat, stride in zip(feat_np, STRIDES):
        expected = INPUT_SIZE // stride
        assert feat.shape[2] == expected and feat.shape[3] == expected, \
            f"Feature map at stride {stride} has incorrect shape: {feat.shape}"

    # prepare obj array for the features
    features_array = np.empty(3, dtype=object)
    for i, arr in enumerate(feat_np):
        features_array[i] = arr

    # save .npz files
    save_path = os.path.join(OUT_DIR, f"{Path(img_name).stem}.npz")
    np.savez_compressed(
        save_path,
        boxes=np.array(boxes, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        confidences=np.array(confs, dtype=np.float32),
        features=features_array,
        strides=np.array(STRIDES, dtype=np.int32),
        channels=np.array([arr.shape[1] for arr in feat_np], dtype=np.int32),
        image_path=np.array(img_path, dtype=object)
    )

print(f"Saved all {len(img_files)} .npz outputs to: {OUT_DIR}")
