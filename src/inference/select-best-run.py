import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

exp = mlflow.get_experiment_by_name(
    "defect-yolo-experiments"
)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id]
)

cols = [
    "tags.mlflow.runName",
    "metrics.metrics/mAP50-95B",
    "metrics.metrics/mAP50B",
    "metrics.metrics/precisionB",
    "metrics.metrics/recallB",
    "params.model",
    "params.dataset",
    "params.imgsz",
    "params.epochs"
]

df = runs[cols].sort_values(
    "metrics.metrics/mAP50-95B",
    ascending=False
)

print(df.to_string())