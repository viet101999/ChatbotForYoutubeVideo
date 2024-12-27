import streamlit as st
import requests
import time
from streamlit_chat import message

if "my_text" not in st.session_state:
    st.session_state.my_text = ""
    
def submit_text():
    st.session_state.my_text = st.session_state.input_text
    st.session_state.input_text = ""
    
def query_text(prompt):
	response = requests.get(
        f"{API_URL}/query_chatbot_text/", 
        params={"question": prompt}
    )
	return response

def query_image(prompt, image_file):
	response = requests.get(
        f"{API_URL}/query_chatbot_image/", 
        params={"question": prompt}, 
        files={"image_file": image_file}
    )
	return response

def get_text():
    st.text_input("Textbox", key="input_text", on_change=submit_text, placeholder="Ask a question about the video")
    input_text = st.session_state.my_text
    return input_text 

def get_image():
    input_image = st.file_uploader("Upload an image:", type=["jpg", "png"], key="input_image")
    return input_image 

# Backend URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🎥 Video Q&A Chatbot",
    page_icon="✅",
    layout="wide",
)
st.title("🎥 Video Q&A Chatbot")

if 'start' not in st.session_state:
    st.session_state['start'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
st.sidebar.write("Upload a video or provide a YouTube link to ask questions about its content.")

input_type = st.sidebar.radio("Choose input type:", ["File Upload", "YouTube link"])

if input_type == "File Upload":
    path = st.sidebar.file_uploader("Upload a video file (mp4)", type=["mp4"])
else:
    path = st.sidebar.text_area("Paste your YouTube link here:")

video_name = ""
if input_type == "File Upload":
    if path:
        with st.spinner("Processing video..."):
            response = requests.post(f"{API_URL}/upload/", files={"file": path})
            if response.status_code == 200:
                st.success("Video processed successfully!")
                video_name = response.json()["name"]
            else:
                st.error("Error processing video.")
else:
    if path:
        with st.spinner("Processing video..."):
            response = requests.post(f"{API_URL}/process_youtube/", params={"link": path})
            if response.status_code == 200:
                st.success("Video processed successfully!")
                video_name = response.json()["name"]
            else:
                st.error("Error processing video.")
                
                
if path:
    if video_name:
        hello_sentence = f"Hello! Welcome to the Video Q&A Chatbot.\nIf you have any questions regarding '{video_name}' video, feel free to ask"
        st.session_state['start'].append(hello_sentence)
    
    image_file = get_image()
    user_input = get_text()

    # Ask question
    if image_file and user_input:
        response = query_image(user_input, image_file)
        print(response.json())
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response.json())
        
    elif user_input:
        response = query_text(user_input)
        print(response.json())
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response.json())

    if st.session_state['generated'] and st.session_state['start']:
        for i in range(len(st.session_state['generated'])-1, -1 , -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["start"][0], key="start")
    elif st.session_state['start']:
        message(st.session_state["start"][0], key="start")