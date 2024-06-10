import os
import cv2
import numpy as np
import logging
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Paths
DATASET_PATH = '/Users/daniel/Downloads/DL_Local/HMDB Dataset/hmdb51_org'
OUTPUT_PATH = 'data/processed'
BUCKET_NAME = 'bucket-classify-videos-hmdb51'

# Parameters
IMG_SIZE = (224, 224)
FRAME_SKIP = 5
BATCH_SIZE = 10  # Number of videos to process in each batch

def upload_to_gcs(local_path, bucket_name, gcs_path):
    logging.info(f"Starting upload of {local_path} to gs://{bucket_name}/{gcs_path}")
    client = storage.Client.from_service_account_json('/Users/daniel/Downloads/DL_Local/classify-videos-hmdb51-3db90d2bc1bb.json')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")

def extract_frames(video_path, frame_skip=FRAME_SKIP):
    logging.info(f"Extracting frames from video: {video_path}")
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return frames
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame / 255.0  # Normalize
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

def process_video(video_path, label, output_path, bucket_name):
    logging.info(f"Processing video: {video_path}")
    frames = extract_frames(video_path)
    if len(frames) > 0:
        frames_filename = f"{os.path.splitext(video_path)[0]}_frames.npy"
        frames_path = os.path.join(output_path, frames_filename)
        np.save(frames_path, frames)
        
        label_filename = f"{os.path.splitext(video_path)[0]}_label.npy"
        label_path = os.path.join(output_path, label_filename)
        np.save(label_path, label)
        
        # Upload to GCS
        upload_to_gcs(frames_path, bucket_name, f"processed/{frames_filename}")
        upload_to_gcs(label_path, bucket_name, f"processed/{label_filename}")
        
        # Remove local files to save space
        os.remove(frames_path)
        os.remove(label_path)
        
        logging.info(f"Processed and uploaded video {video_path}")

def preprocess_videos(dataset_path=DATASET_PATH, output_path=OUTPUT_PATH, bucket_name=BUCKET_NAME, batch_size=BATCH_SIZE):
    labels = os.listdir(dataset_path)
    video_paths = []
    
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        
        # Skip non-directory files
        if not os.path.isdir(label_path):
            continue
        
        videos = os.listdir(label_path)
        
        for video in videos:
            video_path = os.path.join(label_path, video)
            
            # Ensure the path points to a file and is a video
            if not os.path.isfile(video_path) or not video_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                continue
            
            video_paths.append((video_path, label))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, video_path, label, output_path, bucket_name) for video_path, label in video_paths]
        for future in futures:
            future.result()

if __name__ == "__main__":
    preprocess_videos()