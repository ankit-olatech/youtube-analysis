import cv2
import numpy as np
from moviepy import VideoFileClip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from collections import defaultdict
import os
import pytesseract
from PIL import Image, ImageOps
# from deepface import DeepFace
# import dlib
from collections import defaultdict
import nltk
import string
import random
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build
from django.conf import settings
import re
import subprocess
import googleapiclient.discovery
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import io
# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
# Sentiment Analysis
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')


import moviepy as mp
import speech_recognition as sr
import yt_dlp
import os
from collections import Counter

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

def extract_keywords(text):
    """
    Extract meaningful keywords from text using NLTK.
    
    Args:
        text (str or list): A string or list of strings from which to extract keywords.
    
    Returns:
        list: A list of extracted keywords.
    """
    # If the input is a list, join it into a single string
    if isinstance(text, list):
        text = ' '.join(text)

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    tagged_words = pos_tag(words)  # Tag words with their part of speech

    # Extract nouns and adjectives as keywords
    keywords = [word for word, tag in tagged_words if word.isalnum() and word not in stop_words and tag.startswith(('NN', 'JJ'))]
    return keywords

def fetch_competitor_videos(keyword, max_results=10):
    """
    Fetch competitor videos with detailed metadata based on a keyword.
    """
    youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)

    try:
        # Search for videos based on the keyword
        request = youtube.search().list(
            q=keyword,
            part='snippet',
            type='video',
            maxResults=max_results,
            # order='viewCount'  # Sort by most viewed
        )
        response = request.execute()

        competitor_videos = []
        for item in response.get('items', []):
            video_id = item['id'].get('videoId')
            if video_id:
                video_details = fetch_youtube_video_details(video_id)
                if video_details:
                    # Add search snippet data (title, description, etc.)
                    video_details['snippet'] = item['snippet']
                    competitor_videos.append(video_details)

        return competitor_videos

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def compare_metadata_and_engagement(analyzed_video, competitor_videos):
    """
    Compare metadata and engagement metrics between the analyzed video and competitor videos.
    """
    comparison_results = []

    for competitor in competitor_videos:
        comparison = {
            'title': competitor['title'],
            'views': int(competitor['views']) - int(analyzed_video.get('views', 0)),
            'likes': int(competitor['likes']) - int(analyzed_video.get('likes', 0)),
            'comments': int(competitor['comments']) - int(analyzed_video.get('comments', 0)),
            'keywords': list(set(extract_keywords(competitor['description'])) - set(extract_keywords(analyzed_video.get('description', '')))),
        }
        comparison_results.append(comparison)

    return comparison_results

def analyze_content_strategy(analyzed_video, competitor_videos):
    """
    Analyze differences in content strategy (length, pacing, hooks) between the analyzed video and competitor videos.
    """
    strategy_comparison = []

    analyzed_duration = float(analyzed_video.get('duration', 0))  # Assuming duration is in seconds

    for competitor in competitor_videos:
        competitor_duration = float(competitor.get('duration', 0))
        duration_diff = competitor_duration - analyzed_duration

        strategy_comparison.append({
            'title': competitor['title'],
            'duration_diff': duration_diff,
            'hooks': "Hooks analysis placeholder",  # Add logic to analyze hooks
            'pacing': "Pacing analysis placeholder",  # Add logic to analyze pacing
        })

    return strategy_comparison

# def summarize_keywords(frame_keywords, metadata_keywords):
#     """
#     Summarizes a list of extracted keywords from video frames, metadata, and audio.

#     Args:
#         frame_keywords (list): A list of keywords extracted from video frames.
#         metadata_keywords (list): A list of keywords extracted from video metadata (title, description).
#         audio_keywords (list): A list of keywords extracted from audio transcription.

#     Returns:
#         str: A concise and readable summary of the key themes.
#     """

#     # Combine keywords from all sources
#     combined_keywords = frame_keywords + metadata_keywords 

#     if not combined_keywords:
#         return "No meaningful content detected."

#     # Step 1: Preprocess Keywords (Lowercase and Remove Stopwords)
#     stop_words = set(stopwords.words('english'))
#     cleaned_keywords = [word.lower() for word in combined_keywords if word.isalnum() and word.lower() not in stop_words]

#     if not cleaned_keywords:
#         return "No meaningful keywords found for summarization."

#     # Step 2: Calculate Word Frequency
#     word_frequencies = defaultdict(int)
#     for word in cleaned_keywords:
#         word_frequencies[word] += 1

#     # Step 3: Identify Top Keywords (Most Frequent)
#     sorted_keywords = sorted(word_frequencies, key=word_frequencies.get, reverse=True)
#     top_keywords = sorted_keywords[:5]  # Get the 5 most frequent keywords

#     # Step 4: Generate a Human-Like Summary
#     if len(top_keywords) < 3:
#         summary = f"The main focus appears to be on {', '.join(top_keywords)}."
#     else:
#         summary = (
#             f"This content primarily discusses {top_keywords[0]}, "
#             f"with significant emphasis on {top_keywords[1]} and {top_keywords[2]}. "
#             f"Additionally, it touches upon {', '.join(top_keywords[3:])}."
#         )

#     return summary
def summarize_keywords(frame_keywords, metadata_keywords):
    combined_keywords = frame_keywords + metadata_keywords
    if not combined_keywords:
        return "No meaningful content detected."

    stop_words = set(stopwords.words('english'))
    cleaned_keywords = [re.sub(r'\W+', '', word).lower() for word in combined_keywords if word.lower() not in stop_words]

    word_frequencies = Counter(cleaned_keywords)
    top_keywords = [word for word, _ in word_frequencies.most_common(5)]

    if len(top_keywords) < 3:
        return f"Main topics: {', '.join(top_keywords)}."
    else:
        return f"This video mainly discusses {top_keywords[0]}, with emphasis on {top_keywords[1]} and {top_keywords[2]}."

def extract_frames(video_path, frame_interval=1):
    """
    Extract frames from the video at a specified interval (1 per second).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS

    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (frame_rate * frame_interval) == 0:  # 1 frame per second
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames, frame_rate
def extract_text_from_frames(frames, video_metadata):
    texts = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (640, 360))  # Resize for speed

        # Optimize OCR performance
        text = pytesseract.image_to_string(resized_frame, config='--oem 3 --psm 6')
        texts.append(text.strip())
    

    cleaned_texts = {text.replace("\n", " ").strip() for text in texts if text.strip()}  # Remove empty lines
    print("TEXT EXTRACTED FROM FRAMES", cleaned_texts)
    # Extract metadata keywords
    metadata_keywords = extract_keywords(video_metadata.get('tags', '')) + \
                        extract_keywords(video_metadata.get('title', '')) + \
                        extract_keywords(video_metadata.get('description', '')) 

    return cleaned_texts, summarize_keywords(list(cleaned_texts), metadata_keywords)

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


# EXTRACTION FROM AUDIO AND SUBTITLES


def extract_audio(video_path, audio_output_path="audio.wav"):
    """
    Extract audio from a video file and save it as a WAV file.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    return audio_output_path

def transcribe_audio(audio_path):
    """
    Transcribe audio to text using Google Speech-to-Text API.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# def extract_subtitles(video_url, subtitle_output_path="subtitles.srt"):
#     """
#     Extract subtitles from a YouTube video using yt-dlp.
#     """
#     ydl_opts = {
#         'writesubtitles': True,
#         'subtitlesformat': 'srt',
#         'outtmpl': subtitle_output_path.replace('.srt', '.%(ext)s'),
#         'skip_download': True,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([video_url])
#     return subtitle_output_path

# def read_subtitles(subtitle_path):
#     """
#     Read subtitles from an SRT file and return as a single string.
#     """
#     with open(subtitle_path, 'r', encoding='utf-8') as file:
#         subtitles = file.read()
#     return subtitles

def summarize_text(text):
    """
    Summarize text using NLTK or any other summarization library.
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    summary = ' '.join(sentences[:3])  # Simple summarization: take the first 3 sentences
    return summary

def analyze_audio_and_subtitles(video_url, video_path):
    """
    Extract and summarize text from both audio and subtitles.
    """
    # Step 1: Extract audio from video
    audio_path = extract_audio(video_path)
    print("Extraction")
    
    # Step 2: Transcribe audio to text
    audio_text = transcribe_audio(audio_path)
    
    # Step 3: Extract subtitles
    # subtitle_path = extract_subtitles(video_url)
    # subtitle_text = read_subtitles(subtitle_path)
    
    # Step 4: Combine text from audio and subtitles
    # combined_text = f"{audio_text}\n{subtitle_text}"
    audio_keywords = extract_keywords(audio_text)
    print("EXTRACTED AUDIO", audio_keywords)
    combined_text = f"{audio_text}"

    
    # Step 5: Summarize the combined text
    summary = summarize_text(combined_text)
    
    return {
        "audio_text": audio_text,
        # "subtitle_text": subtitle_text,
        "combined_text": combined_text,
        "summary": summary,
        "audio_keywords": audio_keywords
    }



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

def calculate_clickbait_index(video_id, audio_keywords):
    """
    Calculates the Clickbait Index (0-100%) for a YouTube video based on title, thumbnail, description, and engagement.
    
    Args:
        video_id (str): The YouTube video ID.
        audio_keywords (list): A list of keywords extracted from the audio transcription.
    
    Returns:
        dict: Clickbait score percentage (0-100) and details.
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

    # ------ 1.1. Audio Keywords Similarity Score ------
    audio_similarity_score = sum(1 for word in audio_keywords if word in title) / len(audio_keywords) if audio_keywords else 0
    title_clickbait += audio_similarity_score * 10  # Adjust weight as needed

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
            "engagement_score": round(engagement_score, 2),
            "audio_similarity_score": round(audio_similarity_score, 2)

        }

    }