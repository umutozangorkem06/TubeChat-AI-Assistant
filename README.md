# 🎥 TubeChat

A professional RAG (Retrieval-Augmented Generation) application that allows you to chat with YouTube video content. Simply provide a YouTube video URL, and TubeChat will process the transcript and enable you to ask questions about the video using AI-powered context-aware responses.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **YouTube Transcript Processing**: Automatically fetches and processes video transcripts using `youtube-transcript-api`
- **RAG-Powered Q&A**: Ask questions about video content with context-aware AI responses using GPT-3.5-turbo
- **Vector Search**: Efficient similarity search using FAISS vector database
- **Clean UI**: User-friendly Streamlit interface with embedded video player
- **Comprehensive Error Handling**: Handles various edge cases with informative error messages
- **Secure API Key Management**: Support for environment variables and manual input
- **Multiple URL Formats**: Supports various YouTube URL formats (youtube.com, youtu.be, embed links)

## 🛠️ Tech Stack

- **Python 3.10+**: Core programming language
- **Streamlit**: Web UI framework for rapid application development
- **LangChain**: AI orchestration and chain management for RAG
- **OpenAI API**: 
  - GPT-3.5-turbo for question answering
  - text-embedding-ada-002 for embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector storage
- **youtube-transcript-api**: Python library for fetching YouTube transcripts
- **python-dotenv**: Environment variable management

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** installed on your system
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **Internet connection** (for fetching YouTube transcripts and API calls)
- **pip** package manager (usually comes with Python)

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd TubeChat
```

Or download and extract the ZIP file to your desired location.

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies:

```bash
# Create virtual environment
python -m venv venv
```

**Activate the virtual environment:**

- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- streamlit
- langchain
- langchain-openai
- langchain-community
- openai
- faiss-cpu
- python-dotenv
- youtube-transcript-api
- tiktoken

### Step 4: Set Up Environment Variables (Optional)

Create a `.env` file in the project root directory:

```bash
# On Windows
type nul > .env

# On macOS/Linux
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** You can also enter the API key directly in the app's sidebar if you prefer not to use a `.env` file.

## 🎯 Usage

### Starting the Application

1. **Activate your virtual environment** (if not already activated)

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

### Using the Application

1. **Configure the API Key:**
   - In the sidebar, choose "Enter Manually" or "Load from .env"
   - Enter your OpenAI API key (or ensure it's in your `.env` file)

2. **Input a YouTube Video URL:**
   - Paste a YouTube video URL in the sidebar
   - Supported formats:
     - `https://www.youtube.com/watch?v=VIDEO_ID`
     - `https://youtu.be/VIDEO_ID`
     - `https://www.youtube.com/embed/VIDEO_ID`

3. **Process the Video:**
   - Click the "🚀 Process Video" button
   - Wait for the transcript to be fetched and processed
   - The app will create embeddings and build a vector store

4. **Start Chatting:**
   - Once processed, you can ask questions about the video
   - The AI will respond based on the video's transcript content
   - Example questions:
     - "What is the main topic of this video?"
     - "Summarize the key points discussed"
     - "What are the main takeaways?"
     - "Explain [specific concept] mentioned in the video"

## 📁 Project Structure

```
TubeChat/
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions (URL validation, transcript fetching, text chunking)
├── rag_engine.py       # RAG engine (embeddings, vector store, QA chain)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template (optional)
├── README.md          # This file
└── venv/              # Virtual environment (created during setup)
```

## 🔧 How It Works

### Architecture Overview

1. **URL Validation**: 
   - Validates the YouTube URL using regex patterns
   - Extracts the 11-character video ID

2. **Transcript Fetching**: 
   - Retrieves the video transcript using `youtube-transcript-api`
   - Attempts to get English transcript first, falls back to other languages
   - Handles various error cases (disabled transcripts, unavailable videos, etc.)

3. **Text Chunking**: 
   - Splits the transcript into manageable chunks (default: 1000 characters)
   - Uses overlapping chunks (default: 200 characters) for better context retention
   - Attempts to break at sentence boundaries when possible

4. **Embedding Creation**: 
   - Generates embeddings for each chunk using OpenAI's `text-embedding-ada-002` model
   - Converts text into high-dimensional vectors for similarity search

5. **Vector Store**: 
   - Stores embeddings in a FAISS vector database
   - Enables efficient similarity search for relevant context retrieval

6. **Question Answering**: 
   - Uses LangChain's RetrievalQA chain
   - Retrieves top 4 most relevant chunks based on the question
   - Generates context-aware responses using GPT-3.5-turbo
   - Returns answers based solely on the video transcript content

### RAG Pipeline

```
User Question
    ↓
Vector Store Search (FAISS)
    ↓
Retrieve Top 4 Relevant Chunks
    ↓
Combine with Question in Prompt
    ↓
GPT-3.5-turbo Processing
    ↓
Context-Aware Answer
```

## ⚠️ Limitations

- **Transcript Availability**: Videos must have transcripts/captions enabled
- **Processing Time**: Processing time depends on video transcript length
- **Internet Connection**: Requires an active internet connection for:
  - Fetching YouTube transcripts
  - Making OpenAI API calls
- **API Costs**: OpenAI API usage incurs costs (check [OpenAI pricing](https://openai.com/pricing))
- **Rate Limits**: Subject to OpenAI API rate limits
- **Language Support**: Best results with English transcripts (though other languages are supported)

## 🐛 Troubleshooting

### "No transcript found" Error

**Problem:** The video doesn't have captions available.

**Solutions:**
- Try a different video that has captions enabled
- Check if the video has auto-generated captions
- Some videos may have captions in languages other than English

### "Invalid YouTube URL" Error

**Problem:** The URL format is not recognized.

**Solutions:**
- Ensure the URL is a valid YouTube video URL
- Supported formats:
  - `https://www.youtube.com/watch?v=VIDEO_ID`
  - `https://youtu.be/VIDEO_ID`
  - `https://www.youtube.com/embed/VIDEO_ID`
- Make sure the video ID is 11 characters long

### API Key Errors

**Problem:** Issues with OpenAI API key.

**Solutions:**
- Verify your OpenAI API key is correct
- Ensure you have sufficient API credits
- Check that the API key has proper permissions
- Try regenerating the API key from OpenAI Platform

### Import Errors

**Problem:** Missing dependencies or Python version issues.

**Solutions:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.10 or higher: `python --version`
- Try reinstalling dependencies: `pip install -r requirements.txt --upgrade`
- Ensure your virtual environment is activated

### Rate Limiting Errors

**Problem:** Too many requests to OpenAI API.

**Solutions:**
- Wait a few moments and try again
- Check your OpenAI API usage limits
- Consider upgrading your OpenAI plan if needed

### Video Unavailable Error

**Problem:** The video cannot be accessed.

**Solutions:**
- Check if the video exists and is publicly accessible
- The video may be private, deleted, or restricted
- Try a different video

## 📝 Code Quality

This project follows Python best practices:

- **Comprehensive Docstrings**: All functions and classes have detailed docstrings
- **Type Hints**: Type annotations for better code clarity
- **Error Handling**: Comprehensive error handling with informative messages
- **Modular Design**: Clean separation of concerns across modules
- **Comments**: Inline comments for complex logic

## 🔒 Security Notes

- **API Key Security**: Never commit your `.env` file or API keys to version control
- **Environment Variables**: Use `.env` files for local development
- **API Key Storage**: Consider using secure secret management for production deployments

## 📊 Performance Considerations

- **Chunk Size**: Default chunk size (1000 characters) balances context and processing speed
- **Overlap**: Default overlap (200 characters) helps maintain context between chunks
- **Retrieval**: Top 4 chunks are retrieved for each question (configurable in `rag_engine.py`)
- **Embedding Model**: Uses `text-embedding-ada-002` for fast and cost-effective embeddings

## 🚀 Future Enhancements

Potential improvements for future versions:

- Support for multiple videos in a single session
- Conversation history persistence
- Export chat conversations
- Support for video playlists
- Custom chunk size and overlap configuration
- Support for other LLM providers
- Batch processing for multiple videos
- Transcript language detection and translation

## 📄 License

This project is open source and available for educational and portfolio purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 👨‍💻 Author

Built as a portfolio project demonstrating:
- RAG (Retrieval-Augmented Generation) implementation
- LangChain integration
- Streamlit web application development
- Vector database usage (FAISS)
- OpenAI API integration

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://openai.com/) for the AI models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for transcript fetching

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the error messages for specific guidance
3. Ensure all dependencies are correctly installed
4. Verify your API key and internet connection
5. Reach me out - www.linkedin.com/in/umutozangorkem/
---

**Note**: This application uses OpenAI's API, which requires an API key and may incur costs. Please review [OpenAI's pricing](https://openai.com/pricing) before extensive use. Always keep your API keys secure and never share them publicly.

**Happy Chatting! 🎉**
