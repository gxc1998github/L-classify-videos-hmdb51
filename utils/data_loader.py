import os
import numpy as np

# Paths
PROCESSED_DATA_PATH = 'data/processed'

def load_data(processed_data_path=PROCESSED_DATA_PATH):
    X = []
    y = []

    for filename in os.listdir(processed_data_path):
        if filename.endswith('_frames.npy'):
            frames_path = os.path.join(processed_data_path, filename)
            label_path = frames_path.replace('_frames.npy', '_label.npy')

            frames = np.load(frames_path)
            label = np.load(label_path)

            X.append(frames)
            y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    print(f"Loaded {len(X)} samples")