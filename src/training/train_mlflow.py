import os
import time
import mlflow
from ultralytics import YOLO


def run():

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("defect-yolo-experiments")

    # -----------------------------
    # Experiment config
    # -----------------------------
    config = {
        "model": "yolo11s.pt",
        "data": "data/processed_tiled/dataset.yaml", # not done yet.
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "lr0": 0.01,
        "dataset": "tiled_640"
    }

    run_name = f"{config['model']}_tiled-640_{config['imgsz']}_{config['epochs']}ep"

    # -----------------------------
    # MLflow run
    # -----------------------------
    with mlflow.start_run(run_name=run_name):

        # log hyperparameters
        mlflow.log_params(config)

        # -----------------------------
        # Load model
        # -----------------------------
        model = YOLO(config["model"])

        start_time = time.time()

        # -----------------------------
        # TRAINING
        # -----------------------------
        results = model.train(
            data=config["data"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            lr0=config["lr0"],
            workers=0   # IMPORTANT: avoids multiprocessing crash
        )

        train_time = time.time() - start_time

        # -----------------------------
        # VALIDATION
        # -----------------------------
        metrics = model.val(data=config["data"])

        # -----------------------------
        # MLflow metrics logging
        # -----------------------------
        mlflow.log_metrics({
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50_95": metrics.box.map,
            "train_time_sec": train_time
        })

        # -----------------------------
        # SAFE artifact logging
        # -----------------------------
        save_dir = results.save_dir
        print(f"📁 YOLO save directory: {save_dir}")

        best_model = os.path.join(save_dir, "weights/best.pt")
        cm = os.path.join(save_dir, "confusion_matrix.png")
        pr = os.path.join(save_dir, "PR_curve.png")

        if os.path.exists(best_model):
            mlflow.log_artifact(best_model)

        if os.path.exists(cm):
            mlflow.log_artifact(cm)

        if os.path.exists(pr):
            mlflow.log_artifact(pr)

        print("✅ MLflow logging completed successfully")


# -----------------------------
# SAFE ENTRY POINT (CRITICAL)
# -----------------------------
if __name__ == "__main__":
    run()