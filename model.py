import os
import json
from ultralytics import YOLO
from datetime import datetime

def train_fault_detector(dataset_yaml_path="./uploads/dataset/data.yaml",
                         output_dir="./uploads/runs/detect/fault_detector",
                         epochs=10,
                         batch_size=10):
    """
    Trains YOLO model.
    Args:
        dataset_yaml_path (str): Path to data.yaml.
        output_dir (str): Output directory for runs.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
    Returns:
        dict: Metrics summary.
    """
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

    print("🚀 Starting YOLOv8 training on pipeline fault dataset...")

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")  # Pretrained nano model

    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=416,
        batch=batch_size,
        name="fault_detector",
        project=output_dir,
        save_period=5 if epochs > 5 else -1,
        patience=5,
        workers=0
    )

    # Save metrics summary
    metrics_summary = {
        "best_mAP50": float(results.results_dict.get("metrics/mAP50", 0)) * 100,
        "best_mAP50-95": float(results.results_dict.get("metrics/mAP50-95", 0)) * 100,
        "best_epoch": getattr(results, "best_epoch", 0),
        "training_time": str(datetime.now() - results.start_time).split('.')[0] if hasattr(results, 'start_time') else "Unknown"
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("✅ Training complete. Model saved in:", output_dir)
    return model, metrics_summary

def validate_and_predict(model, dataset_yaml_path, output_dir):
    """
    Validates model and runs predictions on valid set.
    """
    print("🔍 Evaluating model on validation set...")
    model.val(data=dataset_yaml_path)  # evaluate on validation data

    print("🔮 Running example predictions on validation images...")
    val_images_path = os.path.join(os.path.dirname(dataset_yaml_path), "valid/images")

    if os.path.exists(val_images_path):
        results = model.predict(
            source=val_images_path,
            conf=0.5,
            save=True,
            project=output_dir,
            name="predictions"
        )
        print("✅ Predictions saved to:", os.path.join(output_dir, "predictions"))
    else:
        print("⚠️ No valid/images found; skipping predictions.")

def pipeline(dataset_yaml_path="./uploads/dataset/data.yaml",
             output_dir="./uploads/runs/detect/fault_detector",
             epochs=10,
             batch_size=10):
    """
    Full pipeline: train + validate/predict.
    """
    model, metrics = train_fault_detector(dataset_yaml_path, output_dir, epochs, batch_size)
    validate_and_predict(model, dataset_yaml_path, output_dir)
    return metrics