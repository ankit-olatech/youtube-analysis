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
# Sentiment Analysis
nltk.download('vader_lexicon')
from googleapiclient.discovery import build
from django.conf import settings
import re
import subprocess
import googleapiclient.discovery
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import io
from deepface import DeepFace

def fetch_youtube_video_details(video_id):
    """
    Fetch metadata and engagement metrics for a YouTube video using its video ID.
    """
    youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)
    print(youtube)
    # Fetch video details
    request = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    )
    try:
        response = request.execute()
    except Exception as e:
        print(f"Error fetching video details: {e}")
        return None

    if not response['items']:
        return None

    video_data = response['items'][0]
    snippet = video_data['snippet']
    statistics = video_data['statistics']

    # Extract relevant details
    metadata = {
        'title': snippet['title'],
        'description': snippet['description'],
        'tags': snippet.get('tags', []),
        'upload_date': snippet['publishedAt'],
        'category_id': snippet['categoryId'],
        'language': snippet.get('defaultAudioLanguage', ''),
        'views': statistics.get('viewCount', 0),
        'likes': statistics.get('likeCount', 0),
        'dislikes': statistics.get('dislikeCount', 0),
        'comments': statistics.get('commentCount', 0),
        'shares': 0,  # Shares are not available via the API
        'thumbnail_url': snippet['thumbnails']['high']['url'],  # Fetch thumbnail URL
    }

    return metadata
def fetch_competitor_videos(keyword, max_results=5):
    """
    Fetch competitor videos based on a keyword.
    """
    youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)

    # Search for videos based on the keyword
    request = youtube.search().list(
        q=keyword,
        part='snippet',
        type='video',
        maxResults=max_results,
        order='viewCount'  # Sort by most viewed
    )
    response = request.execute()

    competitor_videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_details = fetch_youtube_video_details(video_id)
        if video_details:
            competitor_videos.append(video_details)

    return competitor_videos

def fetch_youtube_video_details(video_id):
    """
    Fetch metadata, engagement metrics, and thumbnail URL for a YouTube video.
    """
    youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)

    # Fetch video details
    request = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    )
    response = request.execute()

    if not response['items']:
        return None

    video_data = response['items'][0]
    snippet = video_data['snippet']
    statistics = video_data['statistics']

    # Extract relevant details
    metadata = {
        'title': snippet['title'],
        'description': snippet['description'],
        'tags': snippet.get('tags', []),
        'upload_date': snippet['publishedAt'],
        'category_id': snippet['categoryId'],
        'language': snippet.get('defaultAudioLanguage', ''),
        'views': statistics.get('viewCount', 0),
        'likes': statistics.get('likeCount', 0),
        'dislikes': statistics.get('dislikeCount', 0),
        'comments': statistics.get('commentCount', 0),
        'shares': 0,  # Shares are not available via the API
        'thumbnail_url': snippet['thumbnails']['high']['url'],  # Fetch thumbnail URL
    }

    return metadata
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

import logging
def extract_frames(video_path, frame_interval=7):

    """

    Extract frames from the video at a specified interval.

    """

    frames = []


    if not os.path.exists(video_path):

        logging.error(f"Video file does not exist: {video_path}")

        return None, 0


    cap = cv2.VideoCapture(video_path)
    print(cap)


    if not cap.isOpened():

        logging.error(f"Could not open video file: {video_path}")

        return None, 0  # Return None for frames and 0 for frame rate


    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0


    logging.info(f"Total frames in video: {total_frames}, Frame rate: {frame_rate}")


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            logging.warning("Could not read frame or end of video reached.")

            break

        if frame_count % frame_interval == 0:

            frames.append(frame)

        frame_count += 1


    cap.release()

    logging.info(f"Extracted {len(frames)} frames from {video_path}")

    return frames, frame_rate  # Return the frames and frame rate
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
    Analyze a thumbnail for compliance with YouTube's guidelines and provide an optimization score (0-100).
    """
    try:
        # Open the image
        img = Image.open(thumbnail_path)
        width, height = img.size
        resolution = f"{width}x{height}"
        aspect_ratio = width / height

        # Check file size
        file_size = os.path.getsize(thumbnail_path) / (1024 * 1024)  # Size in MB
        image_format = img.format

        # Check for text overlay
        extracted_text = pytesseract.image_to_string(img)
        has_text = bool(extracted_text.strip())

        # Load image for OpenCV processing
        img_cv = cv2.imread(thumbnail_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        has_faces = len(faces) > 0

        # ---- Detect Facial Expressions (Exaggerated Emotions) ----
        exaggerated_expression = False
        if has_faces:
            analysis = DeepFace.analyze(img_path=thumbnail_path, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            exaggerated_emotions = ['surprise', 'fear', 'angry']
            if dominant_emotion in exaggerated_emotions:
                exaggerated_expression = True

        # ---- Detect Misleading Elements (Red/Yellow Overuse) ----
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        red_percentage = np.sum(red_mask) / (img_cv.shape[0] * img_cv.shape[1])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_percentage = np.sum(yellow_mask) / (img_cv.shape[0] * img_cv.shape[1])

        high_red_yellow = red_percentage > 0.15 or yellow_percentage > 0.15  # Threshold for excessive color use

        # ---- Detect High Contrast Elements (Fake Arrows, Outlines) ----
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        misleading_elements = edge_ratio > 0.1  # High contrast threshold

        # ---- Generate Compliance Score ----
        compliance_score = 100
        deductions = []
        if aspect_ratio != 16 / 9:
            compliance_score -= 10
            deductions.append("Aspect ratio should be 16:9.")
        if file_size > 2:
            compliance_score -= 10
            deductions.append("File size should not exceed 2MB.")
        if exaggerated_expression:
            compliance_score -= 20
            deductions.append("Exaggerated facial expressions detected (shock, fear, or anger).")
        if high_red_yellow:
            compliance_score -= 20
            deductions.append("Excessive use of red/yellow detected, which may be misleading.")
        if misleading_elements:
            compliance_score -= 20
            deductions.append("High contrast elements detected (possible arrows, outlines, or fake elements).")

        compliance_score = max(compliance_score, 0)  # Ensure it doesn’t go below 0

        # Return analysis results
        return {
            'resolution': resolution,
            'aspect_ratio': f"{aspect_ratio:.2f}:1",
            'file_size': f"{file_size:.2f} MB",
            'format': image_format,
            'has_text': has_text,
            'has_faces': has_faces,
            'exaggerated_expression': exaggerated_expression,
            'high_red_yellow': high_red_yellow,
            'misleading_elements': misleading_elements,
            'compliance_score': compliance_score,
            'deductions': deductions,
        }
    except Exception as e:
        return {'error': f'Error analyzing thumbnail: {str(e)}'}
    
# Comment Sentiment Analysis
def fetch_youtube_comments(video_id, max_comments=100):
    """
    Fetches YouTube comments for a given video ID and performs sentiment analysis.
    
    Args:
        video_id (str): The ID of the YouTube video.
        max_comments (int): The maximum number of comments to fetch.
    
    Returns:
        dict: Contains total comment count and sentiment analysis summary.
    """
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)

    # Get video statistics to fetch the total comment count
    video_response = youtube.videos().list(part="statistics", id=video_id).execute()
    total_comments = int(video_response["items"][0]["statistics"].get("commentCount", 0))

    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_comments)
    
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)
        request = youtube.commentThreads().list_next(request, response)

    # Perform Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

    for comment in comments:
        score = sia.polarity_scores(comment)["compound"]
        if score >= 0.05:
            sentiment_scores["positive"] += 1
        elif score <= -0.05:
            sentiment_scores["negative"] += 1
        else:
            sentiment_scores["neutral"] += 1

    return {
        "total_comments": total_comments,
        "sentiment_analysis": sentiment_scores
    }

# Clickbait Analysis
# # List of common clickbait words
CLICKBAIT_WORDS = [
    "shocking", "amazing", "you won’t believe", "mind-blowing", "insane", 
    "crazy", "must-watch", "top 10", "gone wrong", "the truth about", "biggest ever"
]

def calculate_clickbait_index(video_id):
    """
    Calculates the Clickbait Index (0-100%) for a YouTube video based on title, thumbnail, description, and engagement.
    
    Args:
        video_id (str): The YouTube video ID.
    
    Returns:
        float: Clickbait score percentage (0-100).
    """
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)

    # Fetch video details
    video_response = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    
    if not video_response["items"]:
        return {"error": "Invalid video ID or video not found"}

    video_info = video_response["items"][0]["snippet"]
    video_stats = video_response["items"][0]["statistics"]

    title = video_info["title"].lower()
    description = video_info["description"].lower()
    thumbnail_url = video_info["thumbnails"]["high"]["url"]
    
    views = int(video_stats.get("viewCount", 0))
    likes = int(video_stats.get("likeCount", 0)) if "likeCount" in video_stats else 0

    # ------ 1. Title Clickbait Score (40%) ------
    title_score = sum(1 for word in CLICKBAIT_WORDS if word in title) / len(CLICKBAIT_WORDS)
    title_clickbait = title_score * 40

    # ------ 2. Description Clickbait Score (20%) ------
    description_score = sum(1 for word in CLICKBAIT_WORDS if word in description) / len(CLICKBAIT_WORDS)
    description_clickbait = description_score * 20

    # ------ 3. Thumbnail Clickbait Score (30%) ------
    try:
        image = io.imread(thumbnail_url)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection for exaggerated outlines
        edges = cv2.Canny(gray, 100, 200)

        # Count white pixels (high contrast detection)
        edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        thumbnail_clickbait = min(edge_ratio * 150, 30)  # Cap at 30%
    except Exception as e:
        thumbnail_clickbait = 10  # Default value if thumbnail processing fails

    # ------ 4. Like-to-View Ratio Clickbait Score (10%) ------
    engagement_ratio = (likes / views) if views > 0 else 0
    if engagement_ratio < 0.01:  # If likes are very low compared to views
        engagement_score = 10
    elif engagement_ratio < 0.05:
        engagement_score = 5
    else:
        engagement_score = 0

    # ------ Final Clickbait Index ------
    clickbait_index = round(title_clickbait + description_clickbait + thumbnail_clickbait + engagement_score, 2)
    clickbait_index = min(clickbait_index, 100)  # Ensure it's capped at 100%

    return {
        "title": title,
        "clickbait_index": clickbait_index,
        "details": {
            "title_clickbait": round(title_clickbait, 2),
            "description_clickbait": round(description_clickbait, 2),
            "thumbnail_clickbait": round(thumbnail_clickbait, 2),
            "engagement_score": round(engagement_score, 2)
        }
    }