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

#PDF GEN

from django.template.loader import render_to_string
from django.http import HttpResponse
from weasyprint import HTML
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt



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



# Global variable to store progress
progress = {
    'percentage': 0,
    'message': 'Initializing...'
}

@csrf_exempt
def update_progress(request):
    global progress
    return JsonResponse(progress)

def analyze_url(request):
    global progress
    progress = {'percentage': 0, 'message': 'Initializing...'}  # Reset progress at the start

    if request.method == 'POST':
        youtube_url = request.POST.get('youtube_url')

        youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        match = re.match(youtube_regex, youtube_url)

        if not match:
            progress['message'] = 'Invalid YouTube URL. Please enter a valid URL.'
            return render(request, 'analysis/home.html', {'error': progress['message']})

        video_id = match.group(6)

        # Fetch video details
        progress['percentage'] = 10
        progress['message'] = 'Fetching video details...'
        video_details = fetch_youtube_video_details(video_id)
        if not video_details:
            progress['message'] = 'Unable to fetch video details. Please check the URL.'
            return render(request, 'analysis/home.html', {'error': progress['message']})
        print("Video Details Fetched")

        # Fetch comment analysis
        progress['percentage'] = 20
        progress['message'] = 'Fetching comments...'
        comment_data = fetch_youtube_comments(video_id)
        video_details["comment_count"] = comment_data["total_comments"]
        video_details["comment_sentiment"] = comment_data["sentiment_analysis"]
        print("Comments Fetched!")

        # Download the video using yt-dlp
        progress['percentage'] = 30
        progress['message'] = 'Downloading video...'
        output_path = 'media/%(title)s.%(ext)s'
        try:
            # subprocess.run(['yt-dlp', '-o', output_path, youtube_url], check=True)
                subprocess.run(['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', '-o', output_path, youtube_url], check=True)
        except subprocess.CalledProcessError as e:
            progress['message'] = f'Error downloading video: {str(e)}'
            return render(request, 'analysis/home.html', {'error': progress['message']})

        video_filename = f"media/{video_details['title']}.mp4"
        print("Video Downloaded")

        # Download the thumbnail
        progress['percentage'] = 40
        progress['message'] = 'Downloading thumbnail...'
        thumbnail_url = video_details['thumbnail_url']
        thumbnail_filename = f"media/{video_details['title']}_thumbnail.jpg"

        try:
            response = requests.get(thumbnail_url)
            response.raise_for_status()  # Raise an error for bad responses
            with open(thumbnail_filename, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            progress['message'] = f'Error downloading thumbnail: {str(e)}'
            return render(request, 'analysis/home.html', {'error': progress['message']})
        print("Thumbnail Analysis Done")

        # Analyze video content
        progress['percentage'] = 50
        progress['message'] = 'Analyzing video content...'
        frames, frame_rate = extract_frames(video_filename)
        key_moments = detect_key_moments(frames)
        print("Frame Extraction and Hook Done!")

        # Extract keywords from video metadata
        progress['percentage'] = 60
        progress['message'] = 'Extracting keywords...'
        metadata_keywords = extract_keywords(video_details['description']) + extract_keywords(video_details['title'])

        # Analyze audio and subtitles
        progress['percentage'] = 70
        progress['message'] = 'Analyzing audio and subtitles...'
        audio_analysis = analyze_audio_and_subtitles(youtube_url, video_filename)

        # Extract audio keywords
        audio_keywords = audio_analysis['audio_keywords']  # Assuming this is returned from the function
        print("Audio Analysis Done")

        # Calculate Clickbait Index with audio keywords
        progress['percentage'] = 80
        progress['message'] = 'Calculating Clickbait Index...'
        clickbait_analysis = calculate_clickbait_index(video_id, audio_keywords)
        video_details["clickbait_index"] = clickbait_analysis["clickbait_index"]
        video_details["clickbait_details"] = clickbait_analysis["details"]
        print("Clickbait Analysis Done")

        # Summarize keywords from description, tags, title, and metadata
        progress['percentage'] = 90
        progress['message'] = 'Summarizing keywords...'
        summary = summarize_keywords(
            description=video_details['description'],
            tags=video_details.get('tags', []),
            title=video_details['title'],
            metadata_keywords=metadata_keywords
        )
        print("Video Summary Done")

        # Fetch competitor videos
        competitor_videos = fetch_competitor_videos(' '.join(metadata_keywords))

        # Analyze content strategy and generate suggestions
        content_strategy_analysis = analyze_content_strategy(video_details, competitor_videos)
        video_details['strategy_comparison'] = content_strategy_analysis['strategy_comparison']
        video_details['suggestions'] = content_strategy_analysis['suggestions']
        print("Content Strategy Analysis Done")

        # Add results to video_details
        video_details['metadata_comparison'] = compare_metadata_and_engagement(video_details, competitor_videos)
        video_details['audio_text'] = audio_analysis['audio_text']  # Add extracted audio text to video details

        # Analyze music
        progress['percentage'] = 95
        progress['message'] = 'Analyzing music...'
        audio_path = extract_audio(video_filename)
        has_music = detect_music(audio_path)
        video_details["has_music"] = has_music

        # Analyze sound adequacy
        sound_analysis = analyze_sound_adequacy(audio_path)
        video_details["sound_analysis"] = sound_analysis

        # Suggest trending music
        music_suggestions = suggest_music(video_details["category_id"])
        video_details["music_suggestions"] = music_suggestions

        # Analyze competitor music
        competitor_music = analyze_competitor_music(competitor_videos)
        video_details["competitor_music"] = competitor_music

        # Suggest music improvements
        music_improvements = suggest_music_improvements(video_details, competitor_music)
        video_details["music_improvements"] = music_improvements

        # Convert frames to base64 for rendering
        base64_frames = [frame_to_base64(frame) for frame in frames]
        request.session['video_details'] = video_details
        request.session['frame_capture'] = base64_frames

        progress['percentage'] = 100
        progress['message'] = 'Analysis complete! Preparing results...'

        return render(request, 'analysis/results.html', {'video_details': video_details, 'frame_capture': base64_frames})

    return redirect('home')

def download_pdf(request):
    # Retrieve data from session or context
    video_details = request.session.get('video_details', {})
    frame_capture = request.session.get('frame_capture', [])

    # Render the HTML template
    html_string = render_to_string('analysis/pdf_template.html', {
        'video_details': video_details,
        'frame_capture': frame_capture,
    })

    # Convert HTML to PDF
    pdf_file = HTML(string=html_string).write_pdf()

    # Create HTTP response with PDF
    response = HttpResponse(pdf_file, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="video_analysis_report.pdf"'
    return response

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
