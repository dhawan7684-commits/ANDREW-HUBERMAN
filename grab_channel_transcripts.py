import os
import re
import glob
from yt_dlp import YoutubeDL

# --- BOT CHANNEL CONFIGURATION ---
CHANNEL_URL = "https://www.youtube.com/@hubermanlab/videos"
TARGET_DIR = "./dataset/youtube_history"

# Change this number to fetch more or fewer videos (None catches everything)
MAX_VIDEOS_TO_FETCH = None  

def clean_filename(title_string):
    """Removes special characters to make titles safe for Windows/Mac filenames."""
    clean = re.sub(r'[^a-zA-Z0-9\s-]', '', title_string)
    clean = re.sub(r'\s+', '-', clean).strip('-')
    return clean

def auto_harvest_entire_channel():
    print(f"🕵️‍♂️ Initializing pure yt-dlp transcription engine on: {CHANNEL_URL}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. First, flat-extract the video IDs and metadata entries
    meta_opts = {
        'extract_flat': True,
        'skip_download': True,
        'playlistend': MAX_VIDEOS_TO_FETCH,
        'quiet': True
    }
    
    video_list = []
    try:
        with YoutubeDL(meta_opts) as ydl:
            result = ydl.extract_info(CHANNEL_URL, download=False)
            if 'entries' in result:
                video_list = list(result['entries'])
    except Exception as e:
        print(f"❌ Failed to parse channel indices using yt-dlp: {str(e)}")
        return

    real_videos = [v for v in video_list if v.get('id') and v.get('title') != 'Unknown video']
    total_discovered = len(real_videos)
    
    if total_discovered == 0:
        print("❌ Zero actual videos returned. Check channel link or connection.")
        return
        
    print(f"🚀 Extracted {total_discovered} target videos! Commencing direct caption downloads...")
    success_count = 0
    
    for idx, video in enumerate(real_videos, start=1):
        video_id = video.get('id')
        raw_title = video.get('title', f"Unknown-Episode-{video_id}")
        safe_title = clean_filename(raw_title)
        
        # Pull release year context from string parameters
        year_match = re.search(r'\b(202[0-9]|201[0-9])\b', raw_title)
        video_year = int(year_match.group(1)) if year_match else 2026
        
        final_filename = f"{video_year}_{safe_title}_youtube.txt"
        final_file_path = os.path.join(TARGET_DIR, final_filename)
        
        if os.path.exists(final_file_path):
            print(f"⏭️ Skipping [{idx}/{total_discovered}] (Already Saved): {raw_title}")
            continue
            
        print(f"\n📥 [{idx}/{total_discovered}] Extracting Transcript: '{raw_title}'...")
        
        # Temporary unique base path for pulling subtitle artifacts
        temp_base = os.path.join(TARGET_DIR, f"temp_{video_id}")
        
        # 2. Tell yt-dlp to request only English auto-generated captions, 
        # skip saving the giant actual video, and output the text directly
        dl_opts = {
            'writeautomaticsub': True,  # Pull auto-generated subtitles
            'subtitleslangs': ['en'],    # Target English text tracks
            'skip_download': True,       # Do NOT download the video file
            'outtmpl': temp_base,        # Temp name for sub matching
            'quiet': True
        }
        
        try:
            with YoutubeDL(dl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
            # Find the caption file downloaded by yt-dlp (usually .vtt format)
            downloaded_subs = glob.glob(f"{temp_base}.*")
            if not downloaded_subs:
                print("❌ No English subtitles found/generated for this video track.")
                continue
                
            sub_file_path = downloaded_subs[0]
            
            # 3. Read the caption file, clean out VTT timestamps/formatting code
            with open(sub_file_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
                
            clean_blocks = []
            for line in raw_lines:
                line = line.strip()
                # Exclude VTT metadata headers, numeric lines, arrow pointers, or position styles
                if (not line or "WEBVTT" in line or "Kind:" in line or "Language:" in line or 
                    "-->" in line or line.isdigit() or line.startswith("<c>")):
                    continue
                # Strip out inner word timestamp tags like <00:00:01.000>
                cleaned_line = re.sub(r'<[^>]+>', '', line)
                if cleaned_line:
                    clean_blocks.append(cleaned_line)
            
            # Merge text components into clean paragraph blocks
            full_text = " ".join(clean_blocks)
            # Remove any repeated back-to-back words caused by scrolling captions
            words = full_text.split()
            deduped_words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i-1]]
            final_cleaned_prose = " ".join(deduped_words)
            
            # Save out to your clean, final database schema file location
            with open(final_file_path, "w", encoding="utf-8") as f:
                f.write(final_cleaned_prose)
                
            print(f"✅ Saved clean transcript format: {final_filename}")
            success_count += 1
            
            # Housekeeping: delete the messy temporary .vtt file
            if os.path.exists(sub_file_path):
                os.remove(sub_file_path)
                
        except Exception as e:
            print(f"❌ Subtitle conversion failed on this item: {str(e)}")
            # Cleanup temp artifacts if a crash happened halfway
            for f in glob.glob(f"{temp_base}.*"):
                try: os.remove(f)
                except: pass

    print(f"\n🎉 Task Completed! Successfully localized {success_count} structural transcripts via yt-dlp.")

if __name__ == "__main__":
    auto_harvest_entire_channel()