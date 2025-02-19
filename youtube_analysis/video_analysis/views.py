from django.shortcuts import render, redirect
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import re
import cv2
import os
import subprocess
from django.core.files.storage import FileSystemStorage
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from moviepy import VideoFileClip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from .utils.utils import *
from pytube import YouTube

from django.conf import settings
import requests
developerKey=settings.YOUTUBE_API_KEY



def home(request):


    return render(request, 'analysis/home.html')

# def analyze_url(request):
#     if request.method == 'POST':
#         youtube_url = request.POST.get('youtube_url')
#         print("TEST 1")
#         # Extract video ID from URL
#         youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
#         match = re.match(youtube_regex, youtube_url)
#         print("TEST 2")

#         if not match:
#             return render(request, 'analysis/home.html', {'error': 'Invalid YouTube URL. Please enter a valid URL.'})

#         video_id = match.group(6)
#         print("TEST 3")
#         print(video_id)
#         # Fetch video details
#         video_details = fetch_youtube_video_details(video_id)
#         if not video_details:
#             return render(request, 'analysis/home.html', {'error': 'Unable to fetch video details. Please check the URL.'})

#         # Download the video
#         yt = YouTube(youtube_url)
#         print(yt)
#         stream = yt.streams.filter(file_extension='mp4').first()
#         print("TEST A")

#         video_path = stream.download(output_path='media')
#         print("TEST 4")

#         # Analyze video content
#         frames, frame_rate = extract_frames(video_path)
#         key_moments = detect_key_moments(frames)
#         summary = summarize_keywords(video_details['description'])
#         print("TEST 5")

#         # Extract keywords for competitor search
#         keywords = summarize_keywords(video_details['title'] + ' ' + video_details['description'])
#         competitor_videos = fetch_competitor_videos(' '.join(keywords))
#         print("TEST 6")

#         # Add analysis results to video_details
#         video_details['key_moments'] = key_moments
#         video_details['summary'] = summary
#         video_details['competitor_videos'] = competitor_videos
#          # Analyze thumbnail
#         thumbnail_analysis = analyze_thumbnail(video_details['thumbnail_url'])
#         video_details['thumbnail_analysis'] = thumbnail_analysis
#         print("TEST 7")

#         # Pass details to the results template
#         return render(request, 'analysis/results.html', {'video_details': video_details})

#     return redirect('home')
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')
# def analyze_url(request):

#     if request.method == 'POST':

#         youtube_url = request.POST.get('youtube_url')

#         print("TEST 1")



#         # Extract video ID from URL

#         youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'

#         match = re.match(youtube_regex, youtube_url)

#         print("TEST 2")


#         if not match:

#             return render(request, 'analysis/home.html', {'error': 'Invalid YouTube URL. Please enter a valid URL.'})


#         video_id = match.group(6)

#         print("TEST 3")

#         print(video_id)


#         # Fetch video details

#         video_details = fetch_youtube_video_details(video_id)

#         if not video_details:

#             return render(request, 'analysis/home.html', {'error': 'Unable to fetch video details. Please check the URL.'})


#         # Download the video using yt-dlp

#         output_path = 'media/%(title)s.%(ext)s'  # Specify the output path and filename format

#         try:

#             subprocess.run(['yt-dlp', '-o', output_path, youtube_url], check=True)

#             print("Video downloaded successfully.")

#         except subprocess.CalledProcessError as e:

#             return render(request, 'analysis/home.html', {'error': f'Error downloading video: {str(e)}'})


#         # Assuming the video file is saved in the media directory

#         # You may need to adjust the filename based on the output format

#         video_filename = f"media/{video_details['title']}.mp4"  # Adjust this based on the actual downloaded filename


#         # Analyze video content

#         frames, frame_rate = extract_frames(video_filename)

#         key_moments = detect_key_moments(frames)

#         summary = summarize_keywords(video_details['description'])

#         print("TEST 5")


#         # Extract keywords for competitor search

#         keywords = summarize_keywords(video_details['title'] + ' ' + video_details['description'])

#         competitor_videos = fetch_competitor_videos(' '.join(keywords))

#         print("TEST 6")


#         # Add analysis results to video_details

#         video_details['key_moments'] = key_moments

#         video_details['summary'] = summary

#         video_details['competitor_videos'] = competitor_videos

#         # Convert frames to base64 for rendering
#         base64_frames = [frame_to_base64(frame) for frame in frames]


#         # Analyze thumbnail

#         thumbnail_analysis = analyze_thumbnail(video_details['thumbnail_url'])

#         video_details['thumbnail_analysis'] = thumbnail_analysis

#         print("TEST 7")


#         # Pass details to the results template


#         return render(request, 'analysis/results.html', {'video_details': video_details, 'frame_capture': base64_frames})


#     return redirect('home')

def analyze_url(request):
    if request.method == 'POST':
        youtube_url = request.POST.get('youtube_url')
        youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        match = re.match(youtube_regex, youtube_url)

        if not match:
            return render(request, 'analysis/home.html', {'error': 'Invalid YouTube URL. Please enter a valid URL.'})

        video_id = match.group(6)

        # Fetch video details
        video_details = fetch_youtube_video_details(video_id)
        if not video_details:
            return render(request, 'analysis/home.html', {'error': 'Unable to fetch video details. Please check the URL.'})

        # Fetch comment analysis
        comment_data = fetch_youtube_comments(video_id)
        video_details["comment_count"] = comment_data["total_comments"]
        video_details["comment_sentiment"] = comment_data["sentiment_analysis"]

        # Download the video using yt-dlp
        output_path = 'media/%(title)s.%(ext)s'
        try:
            subprocess.run(['yt-dlp', '-o', output_path, youtube_url], check=True)
        except subprocess.CalledProcessError as e:
            return render(request, 'analysis/home.html', {'error': f'Error downloading video: {str(e)}'})

        video_filename = f"media/{video_details['title']}.mp4"

        # Analyze video content
        frames, frame_rate = extract_frames(video_filename)
        key_moments = detect_key_moments(frames)
        summary = summarize_keywords(video_details['description'])
        keywords = summarize_keywords(video_details['title'] + ' ' + video_details['description'])
        competitor_videos = fetch_competitor_videos(' '.join(keywords))

        video_details.update({
            "key_moments": key_moments,
            "summary": summary,
            "competitor_videos": competitor_videos,
            "thumbnail_analysis": analyze_thumbnail(video_details['thumbnail_url'])
        })

        # Convert frames to base64 for rendering
        base64_frames = [frame_to_base64(frame) for frame in frames]

        return render(request, 'analysis/results.html', {'video_details': video_details, 'frame_capture': base64_frames})

    return redirect('home')



def analyze_file(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')

        # Validate file type
        if not video_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return render(request, 'analysis/home.html', {'error': 'Invalid file format. Please upload a video file (mp4, avi, mov, mkv).'})

        # Save the file temporarily
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        full_file_path = os.path.join(fs.location, filename)

        # THUMBNAIL ANALYSIS
        thumbnail_file = request.FILES.get('thumbnail_file')
        print("Thumbnail File:", thumbnail_file)
        thumbnail_path = None
        if thumbnail_file:
            thumbnail_path = os.path.join('media', thumbnail_file.name)
            with open(thumbnail_path, 'wb+') as destination:
                for chunk in thumbnail_file.chunks():
                    destination.write(chunk)

        # Extract basic metadata
        try:
            clip = VideoFileClip(full_file_path)
            metadata = {
                'file_name': video_file.name,
                'file_size': f"{video_file.size / (1024 * 1024):.2f} MB",
                'duration': f"{clip.duration:.2f} seconds",
                'resolution': f"{clip.size[0]}x{clip.size[1]}",
                                'thumbnail_path': thumbnail_path,  # Add thumbnail path to metadata
            }

            # Analyze video content
            frames, frame_rate = extract_frames(full_file_path)
            key_moments = detect_key_moments(frames)
            summary = summarize_keywords("Sample description for uploaded video.")  # Placeholder for actual description

            # Analyze thumbnail
            if thumbnail_path:
                thumbnail_analysis = analyze_thumbnail(thumbnail_path)
                metadata['thumbnail_analysis'] = thumbnail_analysis


        # Extract basic metadata



            # Add analysis results to metadata
            metadata['key_moments'] = key_moments
            metadata['summary'] = summary

            # Extract text from frames
            text_extract = extract_text_from_frames(frames)
            print(text_extract)
            print("TEXT EXTRACTED!")
            # Convert frames to base64 for rendering
            base64_frames = [frame_to_base64(frame) for frame in frames]

            # Clean up: remove the temporary file after processing
            os.remove(full_file_path)
            clip.close()
        except Exception as e:
            print("Error", e)

            # Ensure the temporary file is removed even if an error occurs
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            return render(request, 'analysis/home.html', {'error': f'Error processing video file: {str(e)}'})

        # Pass metadata and extracted text to the results template
        return render(request, 'analysis/results.html', {
            'video_details': metadata,
            'frame_capture': base64_frames,
            'text_extract': text_extract  # Pass the extracted text to the template
        })

    return redirect('home')
