import os

import pandas as pd
from tqdm import tqdm

from evaluator import YOLOTrackEvaluator
from yolo.yolo_track import MODEL_PATH


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


def main():
    


if __name__ == "__main__":
    main()
