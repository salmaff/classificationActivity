import cv2
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Function to calculate MSE using scikit-learn
def calculate_mse(imageA, imageB):
    return mean_squared_error(imageA.ravel(), imageB.ravel())

# Function to calculate PSNR using OpenCV
def calculate_psnr(imageA, imageB):
    mse = calculate_mse(imageA, imageB)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to process and filter each frame with various enhancements + median filtering
def process_and_filter_frame(frame, frame_idx, frame_output_folder, filename):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(img_gray)

    min_val, max_val = np.min(img_gray), np.max(img_gray)
    contrast_stretched_img = ((img_gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_gray)

    median_gray = cv2.medianBlur(img_gray, 3)
    median_equalized = cv2.medianBlur(equalized_img, 3)
    median_contrast = cv2.medianBlur(contrast_stretched_img, 3)
    median_clahe = cv2.medianBlur(clahe_img, 3)

    os.makedirs(frame_output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(frame_output_folder, 'grayscale.jpg'), img_gray)
    cv2.imwrite(os.path.join(frame_output_folder, 'equalized.jpg'), equalized_img)
    cv2.imwrite(os.path.join(frame_output_folder, 'contrast_stretched.jpg'), contrast_stretched_img)
    cv2.imwrite(os.path.join(frame_output_folder, 'clahe.jpg'), clahe_img)
    cv2.imwrite(os.path.join(frame_output_folder, 'median_grayscale.jpg'), median_gray)
    cv2.imwrite(os.path.join(frame_output_folder, 'median_equalized.jpg'), median_equalized)
    cv2.imwrite(os.path.join(frame_output_folder, 'median_contrast_stretched.jpg'), median_contrast)
    cv2.imwrite(os.path.join(frame_output_folder, 'median_clahe.jpg'), median_clahe)

    results = {}
    for label, enhanced_img in [
        ("Grayscale", median_gray),
        ("Histogram Equalized", median_equalized),
        ("Contrast Stretched", median_contrast),
        ("CLAHE", median_clahe),
    ]:
        mse = calculate_mse(frame, cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR))
        psnr = calculate_psnr(frame, cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR))
        results[label] = (mse, psnr)

    return results

# Function to process each video frame by frame
def process_video(video_path, video_output_folder):
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    total_mse = {"Grayscale": 0, "Histogram Equalized": 0, "Contrast Stretched": 0, "CLAHE": 0}
    total_psnr = {"Grayscale": 0, "Histogram Equalized": 0, "Contrast Stretched": 0, "CLAHE": 0}
    num_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_output_folder = os.path.join(video_output_folder, f'frame_{frame_idx}')
        results = process_and_filter_frame(frame, frame_idx, frame_output_folder, f'video_frame_{frame_idx}')

        for label, (mse, psnr) in results.items():
            total_mse[label] += mse
            total_psnr[label] += psnr

        num_frames += 1
        print(f'Processed frame {frame_idx} for video {os.path.basename(video_path)}')
        frame_idx += 1

    cap.release()

    avg_mse = {label: total_mse[label] / num_frames for label in total_mse}
    avg_psnr = {label: total_psnr[label] / num_frames for label in total_psnr}

    with open(os.path.join(video_output_folder, 'average_metrics.txt'), 'w') as f:
        f.write(f"Average MSE and PSNR for video {os.path.basename(video_path)}:\n")
        for label in avg_mse:
            f.write(f"{label} - Avg MSE: {avg_mse[label]:.4f}, Avg PSNR: {avg_psnr[label]:.4f}\n")

    return avg_mse, avg_psnr

# Function to automate processing of all videos in a folder
def process_all_videos(data_folder, output_base_folder, all_metrics):
    for video_filename in os.listdir(data_folder):
        if video_filename.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(data_folder, video_filename)
            video_name = os.path.splitext(video_filename)[0]
            video_output_folder = os.path.join(output_base_folder, video_name)

            print(f'Starting to process video: {video_filename}')
            avg_mse, avg_psnr = process_video(video_path, video_output_folder)
            print(f'Finished processing video: {video_filename}')
            all_metrics[video_name] = {'Avg MSE': avg_mse, 'Avg PSNR': avg_psnr}

# New function to process two folders containing videos
def process_two_video_folders(folder1, folder2, output_base_folder):
    all_metrics = {}
    process_all_videos(folder1, output_base_folder, all_metrics)
    process_all_videos(folder2, output_base_folder, all_metrics)

    # Write all metrics to a single text file
    with open(os.path.join(output_base_folder, 'all_average_metrics.txt'), 'w') as f:
        for video_name, metrics in all_metrics.items():
            f.write(f"Video: {video_name}\n")
            for label, values in metrics.items():
                f.write(f"{label} - {values}\n")
            f.write("\n")

# Paths for input data folders and output
data_folder_1 = 'D:/Salma S1/Kuliah/Semester 6/Computer Vision/UTS/PlayingPiano'
data_folder_2 = 'D:/Salma S1/Kuliah/Semester 6/Computer Vision/UTS/JugglingBalls'
output_base_folder = 'D:/Salma S1/Kuliah/Semester 6/Computer Vision/UTS/processed_output'

# Create output base folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Process videos from two folders
process_two_video_folders(data_folder_1, data_folder_2, output_base_folder)
