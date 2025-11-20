"""
Utility functions for TubeChat application.

This module provides helper functions for:
- Validating YouTube URLs
- Fetching video transcripts
- Processing and chunking text
"""

import re
from typing import List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    HTTPError,
)


def validate_youtube_url(url: str) -> Optional[str]:
    """
    Validate and extract video ID from a YouTube URL.
    
    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video ID if valid, None otherwise
        
    Example:
        >>> validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url or not isinstance(url, str):
        return None
    
    # YouTube URL patterns
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|m\.youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            # Validate video ID format (should be 11 characters)
            if len(video_id) == 11:
                return video_id
    
    return None


def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript for a YouTube video.
    
    Attempts to fetch English transcript first, then falls back to any available
    language. Handles various error cases with informative error messages.
    
    Args:
        video_id: YouTube video ID (11 characters)
        
    Returns:
        Combined transcript text as a single string
        
    Raises:
        ValueError: If transcript cannot be fetched for various reasons
            - Transcripts disabled for the video
            - No transcript found
            - Video unavailable
            - Rate limiting or HTTP errors
            - Other unexpected errors
            
    Example:
        >>> transcript = fetch_transcript("dQw4w9WgXcQ")
        >>> len(transcript) > 0
        True
    """
    if not video_id or len(video_id) != 11:
        raise ValueError(f"Invalid video ID: {video_id}. Video ID must be 11 characters.")
    
    try:
        # Create an instance of YouTubeTranscriptApi
        yt_api = YouTubeTranscriptApi()
        
        # Get list of available transcripts
        transcript_list = yt_api.list(video_id)
        
        transcript = None
        try:
            # Try to get English transcript (manual or auto-generated)
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            try:
                # If no English transcript, try to get auto-generated English
                transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                # If still no English, get the first available transcript
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    transcript = available_transcripts[0]
                else:
                    raise NoTranscriptFound(video_id, None, None)
        
        # Fetch the actual transcript data
        transcript_data = transcript.fetch()
        
        # Combine all text entries into a single string
        # Each entry is a FetchedTranscriptSnippet object with text, start, and duration attributes
        full_text = ' '.join([entry.text for entry in transcript_data])
        
        if not full_text.strip():
            raise ValueError(
                f"Transcript is empty for video (ID: {video_id}). "
                "The video may have captions but no actual text content."
            )
        
        return full_text
        
    except TranscriptsDisabled:
        raise ValueError(
            f"Transcripts are disabled for this video (ID: {video_id}). "
            "Please try a different video that has captions enabled."
        )
    except NoTranscriptFound:
        raise ValueError(
            f"No transcript found for this video (ID: {video_id}). "
            "The video may not have captions available. "
            "Try a different video with captions enabled."
        )
    except VideoUnavailable:
        raise ValueError(
            f"Video is unavailable (ID: {video_id}). "
            "Please check if the video exists and is accessible. "
            "The video may be private, deleted, or restricted."
        )
    except (RequestBlocked, HTTPError) as e:
        # Handle rate limiting and HTTP errors
        error_msg = str(e).lower()
        if 'too many' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
            raise ValueError(
                "Too many requests to YouTube. Please wait a moment and try again. "
                "YouTube may be rate-limiting requests."
            )
        raise ValueError(
            f"HTTP error occurred while fetching transcript: {str(e)}. "
            "Please try again later."
        )
    except Exception as e:
        raise ValueError(
            f"An unexpected error occurred while fetching the transcript: {str(e)}. "
            "Please verify the video ID and try again."
        )


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context retention.
    
    Attempts to break text at sentence boundaries when possible to maintain
    semantic coherence. Uses overlapping chunks to preserve context between
    adjacent chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters). Default is 1000.
        chunk_overlap: Number of characters to overlap between chunks. Default is 200.
        
    Returns:
        List of text chunks. If text is shorter than chunk_size, returns single-item list.
        
    Example:
        >>> text = "This is a long text. " * 100
        >>> chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        >>> len(chunks) > 1
        True
    """
    if not text or not isinstance(text, str):
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            # Check in reverse order (closest to end first)
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '.\t']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    # Found a sentence boundary, break there
                    end = last_punct + 2
                    break
        
        # Extract chunk and strip whitespace
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        
        # Prevent infinite loop or going backwards
        if start >= len(text):
            break
        if start <= 0:
            start = end
    
    return chunks if chunks else [text]
