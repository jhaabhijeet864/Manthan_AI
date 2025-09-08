import cv2
import os
from pathlib import Path

# Define paths
RAW_DATA_DIR = Path("data/raw_data/noaa_images")
OUTPUT_DIR = Path("models/data/raw_data/images")
POSITIVE_DIR = OUTPUT_DIR / "positive"
NEGATIVE_DIR = OUTPUT_DIR / "negative"

# Ensure output directories exist
POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, output_folder, prefix="frame", frame_interval=30):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save the extracted frames.
        prefix (str): Prefix for the saved frame filenames.
        frame_interval (int): Interval between frames to save (e.g., every 30th frame).
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every `frame_interval` frame
        if frame_count % frame_interval == 0:
            frame_filename = f"{prefix}_{saved_count:04d}.jpg"
            frame_path = output_folder / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames to {output_folder}")

def main():
    # Process positive videos
    positive_videos = list(RAW_DATA_DIR.glob("Test/*.mp4")) + list(RAW_DATA_DIR.glob("Training_and_validation/Positive_fish/*.mp4"))
    for video in positive_videos:
        extract_frames(video, POSITIVE_DIR, prefix="positive")

    # Process negative videos
    negative_videos = list(RAW_DATA_DIR.glob("Negatives (seabed)/*.mp4"))
    for video in negative_videos:
        extract_frames(video, NEGATIVE_DIR, prefix="negative")

    print("Frame extraction complete!")

if __name__ == "__main__":
    main()
