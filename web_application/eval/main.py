import os

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluator import YOLOTrackEvaluator
from yolo.yolo_track import MODEL_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


def get_video_paths(root_folder: str) -> list[str]:
    video_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(root, file))
    return video_paths


def process_videos(root_folder: str) -> None:
    """
    Args:
        root_folder: This should be the root folder for the VISEM training folder and it looks like this:
                     "VISEM-Tracking/VISEM_Tracking_Train_v4/Train"

    Returns: Nothing. Saves the dataframe to a CSV file.

    """
    videos_paths = get_video_paths(root_folder)

    evaluator = YOLOTrackEvaluator(model_path=MODEL_PATH, confidence=0.1)
    results = []

    for video_path in tqdm(videos_paths):
        motility_data = evaluator.evaluate_video(video_path, video_length=4)
        video_name = video_path.split("\\")[-1].split(".")[0]
        result = {
            "ID": video_name,
            "Total": motility_data["total"],
            "Motile (%)": (motility_data["motile"] / motility_data["total"]) * 100 if motility_data["total"] > 0 else 0,
            "Progressive (%)": (motility_data["progressive"] / motility_data["total"]) * 100 if motility_data[
                                                                                                    "total"] > 0 else 0,
            "Immotile (%)": (motility_data["immotile"] / motility_data["total"]) * 100 if motility_data[
                                                                                              "total"] > 0 else 0,
        }
        results.append(result)
        print(result)

    predicted_df = pd.DataFrame(results)
    predicted_df.to_csv("predicted_results.csv")


def correct_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    df.rename(columns={'Motile (%)': 'Total Motile (%)', 'Progressive (%)': 'Progressive motility (%)',
                       'Immotile (%)': 'Immotile sperm (%)'}, inplace=True)
    df['Non progressive sperm motility (%)'] = df['Total Motile (%)'] - df['Progressive motility (%)']
    del df['Index']
    df = df[['ID', 'Total Motile (%)', 'Progressive motility (%)', 'Non progressive sperm motility (%)',
             'Immotile sperm (%)']]
    df.to_csv("corrected_results.csv", float_format='%.2f')


def statistics(path_to_pred, path_to_ground_truth):
    matplotlib.use('TkAgg')
    predictions = pd.read_csv(path_to_pred)
    ground_truth = pd.read_csv(path_to_ground_truth)

    merged_data = pd.merge(predictions, ground_truth, on="ID", suffixes=("_pred", "_gt"))

    assert len(predictions) == len(ground_truth)
    assert "Progressive motility (%)_pred" in merged_data.columns
    assert "Progressive motility (%)_gt" in merged_data.columns
    assert "Non progressive sperm motility (%)_pred" in merged_data.columns
    assert "Non progressive sperm motility (%)_gt" in merged_data.columns
    assert "Immotile sperm (%)_pred" in merged_data.columns
    assert "Immotile sperm (%)_gt" in merged_data.columns

    metrics = {}

    columns_to_evaluate = ["Progressive motility (%)", "Non progressive sperm motility (%)", "Immotile sperm (%)"]

    for column in columns_to_evaluate:
        pred_col = f"{column}_pred"
        gt_col = f"{column}_gt"

        # Compute errors
        mae = mean_absolute_error(merged_data[gt_col], merged_data[pred_col])
        mse = mean_squared_error(merged_data[gt_col], merged_data[pred_col])
        corr_pearson, _ = pearsonr(merged_data[gt_col], merged_data[pred_col])
        corr_spearman, _ = spearmanr(merged_data[gt_col], merged_data[pred_col])

        # Store results
        metrics[column] = {
            "MAE": mae,
            "MSE": mse,
            "Pearson Correlation": corr_pearson,
            "Spearman Correlation": corr_spearman
        }

    for metric, values in metrics.items():
        print(f"\nMetrics for {metric}:")
        for k, v in values.items():
            print(f"{k}: {v:.2f}")

    for column in columns_to_evaluate:
        pred_col = f"{column}_pred"
        gt_col = f"{column}_gt"

        plt.figure(figsize=(8, 6))
        plt.scatter(merged_data[gt_col], merged_data[pred_col], alpha=0.7, label="Predictions")
        plt.plot([merged_data[gt_col].min(), merged_data[gt_col].max()],
                 [merged_data[gt_col].min(), merged_data[gt_col].max()],
                 color="red", linestyle="--", label="Ideal")
        plt.title(f"Predicted vs. Ground Truth: {column}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid()
        plt.show()

    mae_data = []
    for column in columns_to_evaluate:
        pred_col = f"{column}_pred"
        gt_col = f"{column}_gt"
        mae = abs(merged_data[gt_col] - merged_data[pred_col])
        mae_data.append(mae)

    boxplot_data = pd.DataFrame({
        "Metric": columns_to_evaluate,
        "MAE": mae_data
    }).explode("MAE")

    plt.figure(figsize=(12, 6))
    boxplot_data.boxplot(by="Metric", column="MAE", grid=False)
    plt.title("MAE Distribution by Metric")
    plt.suptitle("")
    plt.xlabel("Metric")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(axis="y")
    plt.show()


def main():
    statistics('data_csv/corrected_results.csv', 'data_csv/ground_truth.csv')


if __name__ == "__main__":
    main()
