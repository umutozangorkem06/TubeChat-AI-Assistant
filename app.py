import streamlit as st
import os
from dotenv import load_dotenv
from utils import validate_youtube_url, fetch_transcript
from rag_engine import RAGEngine


load_dotenv()


st.set_page_config(
    page_title="TubeChat - Chat with YouTube Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF0000;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #CC0000;
        border-color: #CC0000;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "transcript_length" not in st.session_state:
    st.session_state.transcript_length = 0


def reset_chat():
    """Reset the chat history and video processing state."""
    st.session_state.messages = []
    st.session_state.video_processed = False
    st.session_state.rag_engine = None
    st.session_state.video_id = None
    st.session_state.video_url = ""
    st.session_state.transcript_length = 0


def main():
    """Main application function."""
    

    st.markdown('<h1 class="main-header">🎥 TubeChat</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Chat with YouTube videos using AI-powered RAG (Retrieval-Augmented Generation)</p>',
        unsafe_allow_html=True
    )
    
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
       
        api_key_source = st.radio(
            "API Key Source",
            ["Enter Manually", "Load from .env"],
            help="Choose how to provide your OpenAI API key"
        )
        
        api_key = None
        if api_key_source == "Enter Manually":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys",
                value=st.session_state.get("api_key", "")
            )
            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ API key entered")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("✅ API key loaded from .env")
            else:
                st.error("❌ No API key found in .env file")
                st.info("💡 Create a `.env` file with `OPENAI_API_KEY=your_key_here`")
        
        st.divider()
        
       
        st.header("📹 Video Input")
        video_url = st.text_input(
            "YouTube Video URL",
            value=st.session_state.video_url,
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL here. The video must have captions enabled."
        )
        
        process_button = st.button(
            "🚀 Process Video", 
            type="primary", 
            disabled=not api_key or not video_url,
            use_container_width=True
        )
        
      
        if st.button("🔄 Reset Chat", use_container_width=True):
            reset_chat()
            st.rerun()
        
        st.divider()
        
     
        st.header("ℹ️ About")
        st.info("""
        **TubeChat** uses RAG (Retrieval-Augmented Generation) to let you chat with YouTube videos.
        
        **How it works:**
        1. Enter your OpenAI API key
        2. Paste a YouTube video URL
        3. Click "Process Video"
        4. Start chatting!
        
        **Note:** Videos must have captions/transcripts enabled.
        """)
        
       
        if st.session_state.video_processed:
            st.divider()
            st.header("📊 Status")
            st.success("✅ Video processed successfully!")
            if st.session_state.transcript_length > 0:
                st.metric("Transcript Length", f"{st.session_state.transcript_length:,} characters")
    
   
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
        st.info("💡 You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
        st.markdown("""
        ### Getting Started
        
        1. **Get an OpenAI API Key**
           - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
           - Sign up or log in
           - Create a new API key
           - Copy the key (you won't be able to see it again!)
        
        2. **Enter the API Key**
           - Choose "Enter Manually" in the sidebar
           - Paste your API key
           - Or create a `.env` file with `OPENAI_API_KEY=your_key_here`
        
        3. **Start Chatting**
           - Paste a YouTube video URL
           - Click "Process Video"
           - Ask questions about the video!
        """)
        return
    

    if process_button and video_url:
        with st.spinner("🔄 Processing video transcript..."):
            try:
                # Validate URL
                video_id = validate_youtube_url(video_url)
                if not video_id:
                    st.error("❌ Invalid YouTube URL. Please check the URL and try again.")
                    st.info("""
                    **Supported URL formats:**
                    - `https://www.youtube.com/watch?v=VIDEO_ID`
                    - `https://youtu.be/VIDEO_ID`
                    - `https://www.youtube.com/embed/VIDEO_ID`
                    """)
                    return
                
               
                transcript = fetch_transcript(video_id)
                
                if not transcript:
                    st.error("❌ Could not fetch transcript. The video may not have captions.")
                    return
                
              
                st.session_state.rag_engine = RAGEngine(api_key)
                
              
                with st.spinner("🔄 Creating embeddings and vector store..."):
                    st.session_state.rag_engine.create_vector_store(transcript)
                
              
                st.session_state.video_processed = True
                st.session_state.video_id = video_id
                st.session_state.video_url = video_url
                st.session_state.transcript_length = len(transcript)
                
              
                st.session_state.messages = []
                
                st.success(f"✅ Video processed successfully! Transcript length: {len(transcript):,} characters")
                st.info("💬 You can now start asking questions about the video!")
                st.balloons()  
                
            except ValueError as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                
              
                if "quota" in error_msg.lower() or "billing" in error_msg.lower():
                    st.warning("""
                    **API Quota Exceeded**
                    
                    Your OpenAI API account has exceeded its quota. To continue using TubeChat:
                    1. Check your account balance at [OpenAI Billing](https://platform.openai.com/account/billing)
                    2. Add credits or upgrade your plan
                    3. Wait for your quota to reset (if on a usage-based plan)
                    """)
                elif "rate limit" in error_msg.lower():
                    st.info("💡 Please wait a moment and try again. Rate limits reset periodically.")
                elif "invalid" in error_msg.lower() and "api key" in error_msg.lower():
                    st.info("💡 Please check your API key in the sidebar configuration.")
                elif "transcript" in error_msg.lower():
                    st.info("💡 Try a different video that has captions enabled.")
                else:
                    st.info("💡 Please check your API key and try again. If the problem persists, try a different video.")
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ An unexpected error occurred: {error_msg}")
                
              
                if "insufficient_quota" in error_msg.lower() or "429" in error_msg.lower():
                    st.warning("""
                    **OpenAI API Quota/Billing Issue**
                    
                    Your OpenAI API account has exceeded its quota or has billing issues:
                    - Check your account balance: https://platform.openai.com/account/billing
                    - Review your usage: https://platform.openai.com/account/usage
                    - Add payment method or credits if needed
                    """)
                else:
                    st.info("💡 Please check your API key and try again. If the problem persists, try a different video.")
    
    
    if st.session_state.video_processed and st.session_state.rag_engine:
        st.divider()
        st.header("💬 Chat with the Video")
        
        
        video_id = st.session_state.video_id
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.components.v1.iframe(embed_url, width=700, height=400)
        
        st.caption(f"Video ID: {video_id}")
        
       
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        
        if prompt := st.chat_input("Ask a question about the video..."):
      
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
           
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_engine.query(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_message = f"An error occurred: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    elif st.session_state.video_url and not st.session_state.video_processed:
        st.info("👆 Click 'Process Video' in the sidebar to start chatting!")
    
    else:
        st.info("👈 Enter a YouTube video URL in the sidebar to get started!")
        
      
        with st.expander("📚 Example Questions to Try"):
            st.markdown("""
            Once you process a video, you can ask questions like:
            
            - "What is the main topic of this video?"
            - "Summarize the key points discussed"
            - "What are the main takeaways?"
            - "Explain [specific concept] mentioned in the video"
            - "What did the speaker say about [topic]?"
            - "List the steps mentioned in the video"
            """)
        
     
        st.markdown("""
        ### ✨ Features
        
        - **Smart Transcript Processing**: Automatically fetches and processes YouTube video transcripts
        - **Context-Aware Responses**: Uses RAG to provide accurate answers based on video content
        - **Multiple URL Formats**: Supports various YouTube URL formats
        - **Error Handling**: Comprehensive error messages for common issues
        - **Clean UI**: User-friendly interface with embedded video player
        """)


if __name__ == "__main__":
    main()
