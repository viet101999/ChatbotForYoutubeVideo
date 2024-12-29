import os
import re

import yt_dlp


def download_video(link, save_folder):
    """Download YouTube video and return its path."""
    ydl_opts = {
        "outtmpl": os.path.join(save_folder, '%(title)s.%(ext)s'),
        "format": "best",
        'noplaylist': True,  # Ensure it's just one video (not a playlist)
        'quiet': True  # Reduce output from yt-dlp
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.download([link])
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', 'downloaded_video')

    # Replace spaces with underscores in the file name
    sanitized_title = re.sub(r'[^a-zA-Z0-9]', '', video_title).lower()
    video_extension = info_dict.get('ext', 'mp4')  # Default extension to mp4 if not available
    cleaned_path = os.path.join(save_folder, f"{sanitized_title}.{video_extension}")

    original_path = os.path.join(save_folder, f"{video_title}.{video_extension}")
    if os.path.exists(original_path):
        os.rename(original_path, cleaned_path)

    return cleaned_path


def download_and_convert_video(link, save_folder):
    # Define the output directory and ensure it exists
    os.makedirs(save_folder, exist_ok=True)

    # Create the yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download the best available audio format
        'outtmpl': os.path.join(save_folder, '%(title)s.%(ext)s'),  # Use the video title for the filename
        'postprocessors': [{  # Add postprocessing for audio conversion
            'key': 'FFmpegExtractAudio',  # Correct postprocessor key for audio conversion
            'preferredcodec': 'wav',  # Convert to .wav format
            'preferredquality': '192',  # Set preferred audio quality (optional)
        }],
        'postprocessor_args': [
            '-ar', '44100',  # Set audio sample rate to 44.1kHz (standard for WAV)
            '-ac', '2'  # Set audio to stereo (2 channels)
        ],
        'noplaylist': True,  # Ensure it's just one video (not a playlist)
        'quiet': True  # Reduce output from yt-dlp
    }

    # Download and convert the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract the video information first to get the title
        result = ydl.download([link])
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', 'downloaded_video')

        # # Update output filename with title and download the audio
        # ydl_opts['outtmpl'] = os.path.join(save_folder, f'{video_title}.%(ext)s')
        # with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #     ydl.download([link])

    # Replace spaces with underscores in the file name
    sanitized_title = re.sub(r'[^a-zA-Z0-9]', '', video_title).lower()
    video_extension = "wav"  # Default extension to mp4 if not available
    cleaned_path = os.path.join(save_folder, f"{sanitized_title}.{video_extension}")

    original_path = os.path.join(save_folder, f"{video_title}.{video_extension}")
    if os.path.exists(original_path):
        os.rename(original_path, cleaned_path)

    return cleaned_path

    # # Return the full path to the .wav file
    # return os.path.join(save_folder, f'{video_title}.wav')


def get_video_title(link):
    # Create the yt-dlp options
    ydl_opts = {
        "format": "best",
        'noplaylist': True,  # Ensure it's just one video (not a playlist)
        'quiet': True  # Reduce output from yt-dlp
    }

    # Download and convert the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract the video information first to get the title
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', 'downloaded_video')

    return video_title


if __name__ == "__main__":
    output_file_path = get_video_title("https://www.youtube.com/watch?v=9RhWXPcKBI8")
    print(f"The audio file is saved at: {output_file_path}")
