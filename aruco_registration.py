import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2.aruco as aruco
from tqdm import tqdm
import json

# Function to detect ArUco markers in image tiles and compute centroids
def detect_aruco_markers_in_tiles(frame, tile_size=1200, overlap=200):
    h, w = frame.shape[:2]
    detected_corners = []
    detected_ids = []
    centroids = {}

    # Collect detections from all tiles
    all_corners = []
    all_ids = []
    all_centroids = {}

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            start_x = max(0, x - overlap)
            start_y = max(0, y - overlap)
            end_x = min(w, x + tile_size + overlap)
            end_y = min(h, y + tile_size + overlap)

            expanded_tile = frame[start_y:end_y, start_x:end_x]
            
            # Convert to grayscale and detect markers in the expanded tile
            gray = cv2.cvtColor(expanded_tile, cv2.COLOR_BGR2GRAY)
            # corners, ids, _ = aruco_detector.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            corners, ids, _ = aruco_detector.detectMarkers(gray)
            if ids is not None:
                for corner, id_ in zip(corners, ids):
                    id_scalar = id_[0]
                    if id_scalar in VALID_IDS:
                        # Adjust corner coordinates relative to the original frame
                        corner[:, :, 0] += start_x
                        corner[:, :, 1] += start_y

                        all_corners.append(corner)
                        all_ids.append(id_scalar)

                        # Compute centroid
                        centroid = np.mean(corner[0], axis=0)
                        if id_scalar in all_centroids:
                            # Average centroids if marker appears in multiple tiles
                            all_centroids[id_scalar].append(centroid)
                        else:
                            all_centroids[id_scalar] = [centroid]

    # Merge detections from all tiles
    for id_scalar, centroids_list in all_centroids.items():
        if len(centroids_list) > 1:
            # Average centroids detected in multiple tiles
            merged_centroid = np.mean(centroids_list, axis=0)
        else:
            merged_centroid = centroids_list[0]
        centroids[id_scalar] = merged_centroid

    # Draw detected markers on the frame
    if all_ids:
        detected_ids = np.array(all_ids)
        detected_corners = all_corners
        cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
    else:
        detected_ids = None

    return frame, centroids, detected_ids



# Define the path to the input video and the ArUCo type
# Base path
base_path = '/Volumes/SSD_various/DATA/sharks/2024_sequences/'
# File paths
video_path = f'{base_path}sequence_20240303_070126703_DJI_0257_trim.MP4'
output_video_path = f'{base_path}registered_clips/sequence_20240303_070126703_DJI_0257_trim_ema_alpha09.MP4'

# Load the ArUCo dictionary
aruco_type = cv2.aruco.DICT_4X4_50
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(aruco_type)
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT,ARUCO_PARAMETERS)

# Specify the valid marker IDs
VALID_IDS = [0, 1, 3]

# Smoothing factor for exponential moving average
alpha = 0.9  # Adjust between 0 and 1; lower values mean more smoothing

# Initialize smoothed centroids history
smoothed_centroids_history = {}
unsmoothed_centroids_history = {}
smoothed_centroids_history_full = {}
# Frame range for processing
start_frame = 500  # starting frame
end_frame = 700   # ending frame



# Open the input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Unable to open video file.")
    exit()

# Set the starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Read the first frame and detect markers
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Unable to read the first frame.")
    cap.release()
    out.release()
    exit()

first_frame_detected, first_centroids, first_ids = detect_aruco_markers_in_tiles(first_frame)
if first_ids is None or len(first_ids) < 2:
    print("[ERROR] Not enough valid markers detected in the first frame.")
    cap.release()
    out.release()
    exit()

# Initialize smoothed centroids with the first frame's centroids
for id_, centroid in first_centroids.items():
    smoothed_centroids_history[id_] = centroid.copy()

# Write the first frame to the output video
out.write(first_frame_detected)

# Initialize previous valid transformation
prev_transformation = None

# Buffer to store frames needing interpolation
buffered_frames = []
buffered_indices = []

frame_count = start_frame

while cap.isOpened() and frame_count <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    detected_frame, centroids, ids = detect_aruco_markers_in_tiles(frame)
    if ids is not None and len(ids) >= 2:
        smoothed_centroids = {}
        for id_ in ids:
            centroid = centroids[id_]
    
            # Initialize history for unsmoothed centroids if not present
            if id_ not in unsmoothed_centroids_history:
                unsmoothed_centroids_history[id_] = []
            unsmoothed_centroids_history[id_].append(centroid.tolist())
    
            # Apply exponential smoothing for smoothed centroids
            if id_ in smoothed_centroids_history:
                smoothed_centroid = alpha * centroid + (1 - alpha) * smoothed_centroids_history[id_]
                smoothed_centroids_history[id_] = smoothed_centroid.copy()
            else:
                smoothed_centroid = centroid.copy()
                smoothed_centroids_history[id_] = smoothed_centroid.copy()
    
            # Initialize history for smoothed centroids if not present
            if id_ not in smoothed_centroids_history_full:
                smoothed_centroids_history_full[id_] = []
            smoothed_centroids_history_full[id_].append(smoothed_centroid.tolist())
    
            # Store smoothed centroids for the current frame
            smoothed_centroids[id_] = smoothed_centroid
        
    else:
        smoothed_centroids = {}
        print(f"[INFO] Frame {frame_count}: Not enough valid markers detected.")

    # Find common markers between current frame and the first frame
    matched_ids = set(first_ids).intersection(set(ids)) if ids is not None else set()
    if len(matched_ids) >= 2:
        # Use centroids of matched markers for transformation estimation
        pts_src = []
        pts_dst = []
        for id_ in matched_ids:
            pts_src.append(smoothed_centroids[id_])
            pts_dst.append(first_centroids[id_])

        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        if len(matched_ids) >= 3:
            # Compute affine transformation
            M, inliers = cv2.estimateAffine2D(pts_src, pts_dst, method=cv2.RANSAC)
        else:
            # Compute similarity transformation
            M, inliers = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.RANSAC)

        if M is not None:
            # We have a valid transformation
            # If we have buffered frames, interpolate transformations
            if buffered_frames:
                # Interpolate transformations between prev_transformation and M
                num_buffered = len(buffered_frames)
                for i, (buf_frame, buf_index) in enumerate(zip(buffered_frames, buffered_indices)):
                    t = (i + 1) / (num_buffered + 1)
                    # Linear interpolation of transformation matrices
                    interpolated_M = (1 - t) * prev_transformation + t * M
                    warped_frame = cv2.warpAffine(buf_frame, interpolated_M, (width, height))
                    out.write(warped_frame)
                # Clear the buffers
                buffered_frames = []
                buffered_indices = []

            # Apply transformation to current frame
            warped_frame = cv2.warpAffine(frame, M, (width, height))
            out.write(warped_frame)

            # Update previous valid transformation
            prev_transformation = M.copy()
        else:
            print(f"[INFO] Frame {frame_count}: Transformation could not be computed.")
            # Buffer the frame
            buffered_frames.append(frame)
            buffered_indices.append(frame_count)
    else:
        print(f"[INFO] Frame {frame_count}: Not enough matched markers.")
        # Buffer the frame
        buffered_frames.append(frame)
        buffered_indices.append(frame_count)

    if frame_count % 100 == 0:
        print(f"Processed frame {frame_count}")

    frame_count += 1

# After processing all frames, handle any remaining buffered frames
if buffered_frames and prev_transformation is not None:
    print("[INFO] Processing remaining buffered frames at the end of the video.")
    for buf_frame in buffered_frames:
        # Use the last known valid transformation
        warped_frame = cv2.warpAffine(buf_frame, prev_transformation, (width, height))
        out.write(warped_frame)
elif buffered_frames:
    print("[INFO] No valid transformation available for remaining buffered frames. Writing original frames.")
    for buf_frame in buffered_frames:
        out.write(buf_frame)

# Release video objects
cap.release()
out.release()
print("[INFO] Video processing complete.")

# Save the data to a JSON file
output_data = {
    "non_smoothed_centroids": {str(k): np.array(v).tolist() for k, v in unsmoothed_centroids_history.items()},
    "smoothed_centroids": {str(k): np.array(v).tolist() for k, v in smoothed_centroids_history_full.items()},
}

with open('centroids_data_moving_ema.json', 'w') as f:
    json.dump(output_data, f)

print("[INFO] Centroids data saved to 'centroids_data_moving_ema.json'")
