import os
import shutil


import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch

from torchvision import transforms
from PIL import Image



def transform_image(frame):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(frame).convert("RGB")
    transformed_image = transform(image)

    # print(f"Shape of transformed {frame}: {transformed_image.shape}")

    transformed_image = transformed_image.unsqueeze(0)

    # print(f"Shape of unsqueezed tensor: {transformed_image.shape}")

    return transformed_image


def extract_frames(video_path, output_dir, frame_rate=1, get_first=True):
    """
    Extracts frames from videos and saves them as .jpg files with names based on ID and frame index.
    
    Args:
        video_dir (str): Directory containing the video files.
        output_dir (str): Directory to save the extracted frames.
        csv_file (str): Path to the CSV file with video IDs and filenames.
        frame_rate (int): Number of frames to extract per second of video.
        get_first (bool): If True, only extract the first frame per video. If False, extract multiple frames.
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    frame_interval = max(1, fps // frame_rate)
    frame_idx = 0
    extracted = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if get_first and not extracted:
            frame_name = os.path.join(output_dir, f"frame_0.jpg")
            cv2.imwrite(frame_name, frame)
            extracted = True
            break

        if frame_idx % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_name, frame)

        frame_idx += 1

    cap.release()

    print(f"Frames extracted and saved to {output_dir}")


def get_prediction(video_path: str, output_dir: str, model):

    extract_frames(video_path, output_dir, frame_rate=1, get_first=False)

    # Get all jpg files in the output_dir
    frames = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')]
    transformed_images = [transform_image(frame) for frame in frames]

    outputs = [model(img).item() for img in transformed_images]

    # print(outputs)

    print(f"Mean: {np.mean(outputs)}")

    # delete temporary folder
    shutil.rmtree(output_dir)

    return np.mean(outputs)





