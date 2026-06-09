from ultralytics import YOLO
import cv2
import os

# ======================
# MODEL LOAD
# ======================
MODEL_PATH = "runs/detect/train-12/weights/best.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = model.names  # IMPORTANT


# ======================
# PREDICTION FUNCTION
# ======================
def predict(image_path):
    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls)

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })

    return detections


# ======================
# RISK ENGINE (IMPROVED)
# ======================
def compute_risk(detections):

    if not detections:
        return {"risk_score": 0, "severity": "none"}

    severity_map = {
        "crazing": 0.6,
        "inclusion": 0.8,
        "patches": 0.7,
        "pitted_surface": 0.9,
        "rolled_in_scale": 0.9,
        "scratches": 0.5
    }

    total_defects = len(detections)

    weighted_score = sum(
        d["confidence"] * severity_map.get(d["class_name"], 0.5)
        for d in detections
    ) * 50

    risk_score = int((total_defects * 10) + weighted_score)

    if risk_score < 40:
        severity = "low"
    elif risk_score < 75:
        severity = "medium"
    else:
        severity = "critical"

    return {
        "risk_score": risk_score,
        "severity": severity
    }


# ======================
# VISUALIZATION
# ======================
def save_visualization(image_path, detections, output_path="inference_outputs/result.jpg"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(image_path)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])

        label = f"{d['class_name']} {d['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, img)
    print(f"📸 Saved visualization to: {output_path}")


# ======================
# FULL PIPELINE
# ======================
def run_pipeline(image_path):

    detections = predict(image_path)
    risk = compute_risk(detections)

    save_visualization(image_path, detections)

    return {
        "defects": detections,
        "risk": risk
    }


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    image_path = "data/processed_tiled/images/val/inclusion_1_0.jpg"

    result = run_pipeline(image_path)

    print("\n===== RESULT =====")
    print(result)