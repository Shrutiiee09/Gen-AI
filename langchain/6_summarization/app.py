import streamlit as st
from pytube import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# Set up the language model with API key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='llama-3.1-70b-versatile')
 
# List of top 10 most spoken languages
top_languages = [
    ("English", "en"),
    ("Mandarin Chinese", "zh"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Standard Arabic", "ar"),
    ("Bengali", "bn"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
]
 
# Main function for YouTube summarizer
def youtube_summarizer():
    st.title("YouTube Video Summarizer")
 
    # Input YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:")
 
    # Dropdown for language selection
    language = st.selectbox(
        "Select the language for summarization:",
        [lang_name for lang_name, _ in top_languages]
    )
 
    # Map the selected language name to its code
    language_code = dict(top_languages)[language]
 
    # Button to generate summary
    if st.button("Generate Summary"):
        if youtube_url:
            with st.spinner("Loading transcript, fetching thumbnail, and generating summary..."):
                try:
                    # Get video details using PyTube
                    yt = YouTube(youtube_url)
                    video_title = yt.title
                    thumbnail_url = yt.thumbnail_url
 
                    # Display video title
                    st.subheader(video_title)
 
                    # Load Transcript in the selected language
                    loader = YoutubeLoader.from_youtube_url(youtube_url, language=["hi", "en", "en-US"])
                    transcript = loader.load()
 
                    # Check if transcript is available
                    if not transcript:
                        st.warning("No transcript available in the selected language.")
                        return
 
                    # Split Transcript into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=10000, add_start_index=True)
                    chunks = text_splitter.split_documents(transcript)
 
                    # Prepare prompts
                    chunk_prompt = "Break down the following text into key points and highlight the most critical information for each section: Text: '{text}' Key Points:"
 
 
                    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunk_prompt)
                    
                    final_prompt = '''
                    Provide a comprehensive summary of the key points in the specified language ({language}). First, list the key points as a breakdown, then synthesize these points into a clear and concise summary. TEXT: {text}
                    '''
 
 
                    final_prompt_template = PromptTemplate(input_variables=['text', 'language'], template=final_prompt)
 
                    # Load summarization chain
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type='map_reduce',
                        map_prompt=map_prompt_template,
                        combine_prompt=final_prompt_template,
                        verbose=True
                    )
                  
                    
 
                    # Display video thumbnail
                    st.image(thumbnail_url, use_column_width=True)
 
                    # Generate and display the full summary
                    summary = chain.run({"input_documents": chunks, "language": language})
                    st.subheader(f"Summary (in {language}):")
                    st.success(summary)
 
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid YouTube URL.")
 
if __name__ == "__main__":
    youtube_summarizer()