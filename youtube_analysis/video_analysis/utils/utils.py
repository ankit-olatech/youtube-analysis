import cv2
import numpy as np
from moviepy import VideoFileClip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import os
import pytesseract
from PIL import Image

def extract_frames(video_path, frame_interval=10):
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
    print("EXTRACTED TEXTS",texts)
    # cleaned_texts = {text.replace(" ", "").replace("\n", "").strip() for text in texts}
    cleaned_texts = {text.replace("\n", "").strip() for text in texts}

    return cleaned_texts # to return only distinct values

def detect_key_moments(frames, threshold=30):
    """
    Detect key moments in the video based on frame differences.
    """
    key_moments = []
    prev_frame = None

    for i, frame in enumerate(frames):
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            diff_mean = np.mean(diff)
            if diff_mean > threshold:
                key_moments.append(i)
        prev_frame = frame

    return key_moments

def summarize_text(text):
    """
    Generate a summary of the text using NLTK.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    freq_table = defaultdict(int)

    for word in words:
        word = word.lower()
        if word not in stop_words and word.isalnum():
            freq_table[word] += 1

    sentences = sent_tokenize(text)
    sentence_scores = defaultdict(int)

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[sentence] += freq_table[word]

    summary = ' '.join(sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3])
    return summary