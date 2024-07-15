import streamlit as st
import requests
import tempfile

st.title("Vehicle Detection and Tracking")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)

    if st.button("Process Video"):
        files = {'file': open(video_path, 'rb')}
        response = requests.post('http://3.81.212.99z:5000/process', files=files)

        if response.status_code == 200:
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            output_video_path.write(response.content)
            st.video(output_video_path.name)
            with open(output_video_path.name, 'rb') as f:
                st.download_button("Download Processed Video", f, file_name="processed_video.avi")
        else:
            st.error("Error processing video")