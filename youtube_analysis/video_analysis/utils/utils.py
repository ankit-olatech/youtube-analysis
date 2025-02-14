import cv2
import numpy as np
from moviepy import VideoFileClip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import os
import pytesseract
from PIL import Image, ImageOps
from collections import defaultdict
import nltk
import string
import random
# Ensure you have the necessary NLTK resources
nltk.download('punkt')

def summarize_keywords(keywords_list):
    """
    Summarizes a list of extracted keywords by selecting key themes and generating a human-like summary.

    Args:
        keywords_list (list): A list of keywords extracted from video frames.

    Returns:
        str: A concise and readable summary of the key themes.
    """

    if not keywords_list:
        return "No meaningful content detected."

    # Step 1: Preprocess Keywords (Lowercase and Remove Stopwords)
    stop_words = set(stopwords.words('english'))
    cleaned_keywords = [word.lower() for word in keywords_list if word.isalnum() and word.lower() not in stop_words]

    if not cleaned_keywords:
        return "No meaningful keywords found for summarization."

    # Step 2: Calculate Word Frequency
    word_frequencies = defaultdict(int)
    for word in cleaned_keywords:
        word_frequencies[word] += 1

    # Step 3: Identify Top Keywords (Most Frequent)
    sorted_keywords = sorted(word_frequencies, key=word_frequencies.get, reverse=True)
    top_keywords = sorted_keywords[:5]  # Get the 5 most frequent keywords

    # Step 4: Generate a Human-Like Summary
    if len(top_keywords) < 3:
        summary = f"The main focus appears to be on {', '.join(top_keywords)}."
    else:
        summary = (
            f"This content primarily discusses {top_keywords[0]}, "
            f"with significant emphasis on {top_keywords[1]} and {top_keywords[2]}. "
            f"Additionally, it touches upon {', '.join(top_keywords[3:])}."
        )

    return summary
def extract_frames(video_path, frame_interval=7):
    """
    Extract frames from the video at a specified interval.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    print(frames)
    return frames, frame_rate
def extract_text_from_frames(frames):

    texts = []

    for frame in frames:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("GRAY SCALE", gray_frame)
        # print("TESTING IMAGE",pytesseract.image_to_string(Image.open('test.png')))

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(gray_frame)
        texts.append(text)
    print(set(texts))
    cleaned_texts = {text.replace("\n", "").strip() for text in texts}

    summary = summarize_keywords(str(cleaned_texts))
    print("SUMMARY", summary)

    print("EXTRACTED TEXTS",texts)
    # cleaned_texts = {text.replace(" ", "").replace("\n", "").strip() for text in texts}

    return cleaned_texts, summary # to return only distinct values

def detect_key_moments(frames, threshold=30):
    """
    Detect key moments in the video based on frame differences.
    """
    key_moments = []
    prev_frame = None

    for i, frame in enumerate(frames):
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame) #diff in absolute value of two frame's pixels denoting a change in image
            diff_mean = np.mean(diff)
            if diff_mean > threshold:           #Threshold value denotes the min.diff value after which the pixels are differentiated enough to cause an image change
                key_moments.append(i)
        prev_frame = frame

    return key_moments



# THUMBNAIL ANALYSIS


def analyze_thumbnail(thumbnail_path):
    """
    Analyze a thumbnail for compliance with YouTube's guidelines and provide optimization suggestions.
    """
    try:
        # Open the image
        img = Image.open(thumbnail_path)
        print("THUMBNAIL IMAGE", img)

        # Check resolution and aspect ratio
        width, height = img.size
        resolution = f"{width}x{height}"
        aspect_ratio = width / height

        # Check file size
        file_size = os.path.getsize(thumbnail_path) / (1024 * 1024)  # Size in MB

        # Check image format
        image_format = img.format

        # Check for text overlay
        # print(pytesseract.image_to_string(Image.open(img)))
        # Compare extracted thumbnail text to that of the title to compare the relevance of thumbnail wrt to content
        if pytesseract.image_to_string(img) != None:
            has_text = True

        # Check for faces (using OpenCV)
        img_cv = cv2.imread(thumbnail_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        has_faces = len(faces) > 0

        # Generate suggestions
        suggestions = []
        if aspect_ratio != 16 / 9:
            suggestions.append("Aspect ratio should be 16:9.")
        if file_size > 2:
            suggestions.append("File size should not exceed 2MB.")
        if not has_text:
            suggestions.append("Consider adding text overlay for better engagement.")
        if not has_faces:
            suggestions.append("Consider including faces for better engagement.")

        # Return analysis results
        return {
            'resolution': resolution,
            'aspect_ratio': f"{aspect_ratio:.2f}:1",
            'file_size': f"{file_size:.2f} MB",
            'format': image_format,
            'has_text': has_text,
            'has_faces': has_faces,
            'suggestions': suggestions,
        }
    except Exception as e:
        return {'error': f'Error analyzing thumbnail: {str(e)}'}