import json
from datetime import datetime

from PyQt6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QScrollArea, QWidget, QFrame, QLabel

LOG_FILE = "predictions_log.json"


def log_prediction(video_path, prediction_type, prediction_value, sperm_volume):
    """
    Logs a prediction result to a JSON file.
    
    :param video_path: Path to the video file being analyzed.
    :param prediction_type: Type of prediction (e.g., "motility", "count", etc.).
    :param prediction_value: 
        - For "motility" predictions, this should be a dictionary with keys: "total", "motile", "progressive", and "immotile".
        - For other types, this should be a simple value corresponding to the prediction.
    :param sperm_volume: The volume of the sperm inputted by the user.
    :return: None
    """
    # Create a log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_path": video_path,
        "sperm_volume": sperm_volume,
        "prediction_type": prediction_type
    }

    # Handle motility prediction tuple
    if prediction_type == "motility":
        log_entry["prediction_value"] = {
            "total": prediction_value["total"],
            "motile": prediction_value["motile"],
            "progressive": prediction_value["progressive"],
            "immotile": prediction_value["immotile"]
        }
    else:
        log_entry["prediction_value"] = prediction_value

    # Read existing log entries, if the log file exists
    try:
        with open(LOG_FILE, "r") as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    # Append the new log entry
    logs.append(log_entry)

    # Write the updated log back to the file
    with open(LOG_FILE, "w") as file:
        json.dump(logs, file, indent=4)


