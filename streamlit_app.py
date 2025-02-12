import time
import streamlit as st
from PIL import Image
# import cv2
import tempfile
import numpy as np
# from ultralytics import YOLO 
# from ultralytics.yolo.engine.model import YOLO
import torch



# from io import BytesIO
# import base64
# from tracker import Tracker

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
st.title("Vehicle Tracking")

hex2rgb = lambda hex : tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# @st.cache_resource
# def get_yolo_model():
#     model = YOLO("model/best.pt")
#     model.fuse()
#     return model

# model = get_yolo_model()

# class_names = model.names

# tracker = Tracker()

line_length = 20

bbox_color = (255,0,0)

st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {   
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {   
            width: 350px;
            margin-left: -350px;
        }
        </style>

    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Settings")
# st.sidebar.subheader("parameters")

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

app_mode = st.sidebar.selectbox("Choose the app mode",["About app","Count","Check Running red light"])

if app_mode == "About App":
    st.markdown("## This app is used to track vehicles in a video using **YOLOv8** & **DeepSORT**")
        
elif app_mode == "Count Vehicles From Video":
        
    bbox_color = st.sidebar.color_picker("Bbox Color", "#ffffff")
    # st.sidebar.markdown("---")
    
    # detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # st.sidebar.markdown("---")
    
    # if torch.cuda.is_available():
    #     enable_gpu = st.sidebar.checkbox("Enable GPU", False)
    #     st.sidebar.markdown("---")
    # else:
    #     enable_gpu = False

    # video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    # tffile = tempfile.NamedTemporaryFile(delete=False)

    # in_ids = set()
    # out_ids = set()
    # in_line = ((500,700),(920,700))
    # out_line = ((950,700),(1300,700))



    # if video_file_buffer is not None:
    #     bbox_color = hex2rgb(bbox_color)[::-1]
    #     col1,col2,col3 = st.columns(3)
    #     with col1:
    #         st.markdown("## In Count: ")
    #         in_markdown = st.markdown(f"## {len(in_ids)}")
    #     with col2:
    #         st.markdown("## Out Count: ")
    #         out_markdown = st.markdown(f"## {len(out_ids)}")
    #     with col3:
    #         st.markdown("## FPS: ")
    #         fps_markdown = st.markdown(f"## 0")
    #     stframe = st.empty()

    #     tffile.write(video_file_buffer.read())

    #     vid = cv2.VideoCapture(tffile.name)


    #     st.sidebar.text("Input Video")
    #     st.sidebar.video(tffile.name)

    #     fps = 0
    #     frame_count = 0
    #     start_time = time.time()

    #     while vid.isOpened():
            
    #         ret,frame = vid.read()

    #         if not ret:
    #             break
    #         in_line_color = (0,255,0)
    #         out_line_color = (0,0,255)

    #         if enable_gpu:
    #             results = model(frame,device=0)
    #         else:
    #             results = model(frame,device='cpu')

    #         for result in results:
    #             detections = []
    #             for r in result.boxes.data.tolist():
    #                 x1, y1, x2, y2, score, class_id = r
    #                 x1 = int(x1)
    #                 x2 = int(x2)
    #                 y1 = int(y1)
    #                 y2 = int(y2)
    #                 class_id = int(class_id)
    #                 if score > detection_threshold:
    #                     detections.append([x1, y1, x2, y2, score])
    #         if len(detections) > 0:
    #             tracker.update(frame,detections)
            
    #         if tracker.tracks is not None:
    #             for track in tracker.tracks:
                    
    #                 bbox = track.bbox
    #                 x1, y1, x2, y2 = map(int,bbox)
    #                 center = (int((x1+x2)/2),int((y1+y2)/2))
    #                 track_id = track.track_id

    #                 cv2.line(frame,(x1,y1),(x1+line_length,y1),bbox_color,2)
    #                 cv2.line(frame, (x1, y1), (x1, y1+ line_length), bbox_color, 2)

    #                 cv2.line(frame, (x2, y1), (x2 - line_length, y1), bbox_color, 2)
    #                 cv2.line(frame, (x2, y1), (x2, y1 + line_length), bbox_color, 2)

    #                 cv2.line(frame, (x1, y2), (x1 + line_length, y2), bbox_color, 2)
    #                 cv2.line(frame, (x1, y2), (x1 , y2 - line_length), bbox_color, 2)

    #                 cv2.line(frame, (x2, y2), (x2-line_length, y2), bbox_color, 2)
    #                 cv2.line(frame, (x2, y2), (x2, y2-line_length), bbox_color, 2)
    #                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 0, .5, bbox_color, 2)
                    
    #                 if center[0] >= in_line[0][0] and center[1] >= in_line[0][1]-5 and center[0] <= in_line[1][0] and center[1] <= in_line[1][1]+5:
    #                     in_line_color = (255,255,255)
    #                     in_ids.add(track_id)
    #                     in_markdown.markdown(f"## {len(in_ids)}")
                    
    #                 if center[0] >= out_line[0][0] and center[1] >= out_line[0][1]-5 and center[0] <= out_line[1][0] and center[1] <= out_line[1][1]+5:
    #                     out_line_color = (255,255,255)
    #                     out_ids.add(track_id)
    #                     out_markdown.markdown(f"## {len(out_ids)}")
                    
                    
    
    #         cv2.line(frame,in_line[0],in_line[1],in_line_color,2)
            
    #         cv2.line(frame, out_line[0], out_line[1], out_line_color, 2)

    #         frame.flags.writeable = True
    #         stframe.image(frame, channels="BGR", use_column_width=True)
            
    #         frame_count+=1
    #         elapsed_time = time.time() - start_time
    #         if elapsed_time > 1:
    #             fps = frame_count/elapsed_time
    #             frame_count = 0
    #             start_time = time.time()
            
    #         fps_markdown.markdown(f"## {fps:.2f}")
