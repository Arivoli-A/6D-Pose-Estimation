import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualize_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    img_path = data['image_path'].item()
    img_bgr = cv2.imread(img_path)
    h0, w0 = img_bgr.shape[:2]
    INPUT_SIZE = 640

    # === Reapply letterbox padding ===
    scale = INPUT_SIZE / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_left = (INPUT_SIZE - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    boxes = data['boxes']
    labels = data['labels']

    for box, label in zip(boxes, labels):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * INPUT_SIZE)
        y1 = int((cy - bh / 2) * INPUT_SIZE)
        x2 = int((cx + bw / 2) * INPUT_SIZE)
        y2 = int((cy + bh / 2) * INPUT_SIZE)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(Path(npz_path).name)
    plt.show()

# Example usage
visualize_npz("hope-yolo/npz_outputs/scene_0000_0000.npz")
