import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save the extracted frames.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame as image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")

if __name__ == "__main__":
    # Define paths
    positive_video_path = "d:/Coding Journey/Hackathons/Dronathon/Prototype_Mak1/data/raw_data/noaa_images/Test/Test_ROV_video_h264_full.mp4"
    negative_video_path = "d:/Coding Journey/Hackathons/Dronathon/Prototype_Mak1/data/raw_data/noaa_images/Negatives (seabed)/neg_sb-0001.jpg"

    positive_output_folder = "d:/Coding Journey/Hackathons/Dronathon/Prototype_Mak1/ai_models/data/raw_data/images/positive"
    negative_output_folder = "d:/Coding Journey/Hackathons/Dronathon/Prototype_Mak1/ai_models/data/raw_data/images/negative"

    # Extract frames
    extract_frames(positive_video_path, positive_output_folder)
    extract_frames(negative_video_path, negative_output_folder)
