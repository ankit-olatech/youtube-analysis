from celery import shared_task
from .utils.utils import *
import subprocess

@shared_task
def process_video_analysis(video_id, youtube_url):
    # Fetch video details
    video_details = fetch_youtube_video_details(video_id)
    if not video_details:
        return {'error': 'Unable to fetch video details.'}

    # Fetch comment analysis
    comment_data = fetch_youtube_comments(video_id)
    video_details["comment_count"] = comment_data["total_comments"]
    video_details["comment_sentiment"] = comment_data["sentiment_analysis"]

    # Calculate Clickbait Index
    clickbait_analysis = calculate_clickbait_index(video_id)
    video_details["clickbait_index"] = clickbait_analysis["clickbait_index"]
    video_details["clickbait_details"] = clickbait_analysis["details"]

    # Download the video using yt-dlp
    output_path = 'media/%(title)s.%(ext)s'
    try:
        subprocess.run(['yt-dlp', '-o', output_path, youtube_url], check=True)
    except subprocess.CalledProcessError as e:
        return {'error': f'Error downloading video: {str(e)}'}

    video_filename = f"media/{video_details['title']}.mp4"

    # Analyze video content
    frames, frame_rate = extract_frames(video_filename)
    key_moments = detect_key_moments(frames)
    summary = summarize_keywords(video_details['description'])
    keywords = extract_keywords(video_details['title'] + ' ' + video_details['description'])
    competitor_videos = fetch_competitor_videos(' '.join(keywords))

    # Compare metadata and engagement
    metadata_comparison = compare_metadata_and_engagement(video_details, competitor_videos)

    # Analyze content strategy
    strategy_comparison = analyze_content_strategy(video_details, competitor_videos)

    # Add results to video_details
    video_details.update({
        'metadata_comparison': metadata_comparison,
        'strategy_comparison': strategy_comparison,
        'key_moments': key_moments,
        'summary': summary,
        'competitor_videos': competitor_videos,
        'thumbnail_analysis': analyze_thumbnail(video_details['thumbnail_url'])
    })

    return video_details