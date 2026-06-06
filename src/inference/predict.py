from ultralytics import YOLO
import cv2
import os

def main():
    # Load trained model
    model = YOLO("runs/detect/models/yolo11s_neu-4/weights/best.pt")

    # Input image
    image_path = "data/processed/images/val/crazing_4_0.jpg"
    image = cv2.imread(image_path)

    # Run inference
    results = model(image)

    # Parse results
    result = results[0]

    # Plot predictions on image
    annotated_img = result.plot()

    # Save output
    os.makedirs("inference_outputs", exist_ok=True)
    output_path = "inference_outputs/result.jpg"
    cv2.imwrite(output_path, annotated_img)

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()