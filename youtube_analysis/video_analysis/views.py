from django.shortcuts import render, redirect
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import re
import cv2
import os
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


def home(request):
    return render(request, 'analysis/home.html')

# def analyze_url(request):
#     if request.method == 'POST':
#         youtube_url = request.POST.get('youtube_url')

#         # Validate YouTube URL
#         youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
#         if not re.match(youtube_regex, youtube_url):
#             return render(request, 'analysis/home.html', {'error': 'Invalid YouTube URL. Please enter a valid URL.'})

#         # TODO: Add logic to fetch and analyze video data
#         return render(request, 'analysis/results.html', {'message': 'YouTube URL analysis will be implemented soon.'})

#     return redirect('home')

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def analyze_file(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')

        # Validate file type
        if not video_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return render(request, 'analysis/home.html', {'error': 'Invalid file format. Please upload a video file (mp4, avi, mov, mkv).'})

        # Save the file temporarily
        file_path = os.path.join('media', video_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        # Save the uploaded file to a temporary location
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        full_file_path = os.path.join(fs.location, filename)
        # Extract basic metadata
        try:
            clip = VideoFileClip(file_path)
            metadata = {
                'file_name': video_file.name,
                'file_size': f"{video_file.size / (1024 * 1024):.2f} MB",
                'duration': f"{clip.duration:.2f} seconds",
                'resolution': f"{clip.size[0]}x{clip.size[1]}",
            }

            # Analyze video content
            frames, frame_rate = extract_frames(file_path)
            key_moments = detect_key_moments(frames)
            summary = summarize_text("Sample description for uploaded video.")  # Placeholder for actual description

            # Add analysis results to metadata
            metadata['key_moments'] = key_moments
            metadata['summary'] = summary
                # FRAME CAPTURE
            cap = cv2.VideoCapture(full_file_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            # Convert frames to base64
            base64_frames = [frame_to_base64(frame) for frame in frames]

            # Clean up: remove the temporary file after processing
            os.remove(full_file_path)


            clip.close()
        except Exception as e:
            return render(request, 'analysis/home.html', {'error': f'Error processing video file: {str(e)}'})

        # Pass metadata to the results template
        return render(request, 'analysis/results.html', {'video_details': metadata, 'frame_capture': base64_frames})

    return redirect('home')