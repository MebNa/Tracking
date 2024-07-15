import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort # type: ignore
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tempfile import NamedTemporaryFile

# Cấu hình Streamlit
st.set_page_config(layout="wide")

# Tải mô hình YOLO và khởi tạo DeepSort
model = YOLO("./best.pt")
tracker = DeepSort(max_age=30)

# Tải video từ người dùng
uploaded_file = st.file_uploader("Tải lên video của bạn", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Lưu trữ tạm thời video tải lên
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Sử dụng moviepy để đọc video
    video = VideoFileClip(tfile.name)
    fps = video.fps

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo3.avi', fourcc, fps, (1280, 720))

    # Định nghĩa các đỉnh của các tứ giác
    vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
    vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

    # Định nghĩa phạm vi dọc cho việc cắt và ngưỡng làn đường
    xv1, xv2 = 325, 635
    lane_threshold = 609

    # Định nghĩa ngưỡng để xem xét giao thông
    traffic_flow = 2

    track_dict = {}
    vehicles_left_lane = 0
    vehicles_right_lane = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    org_in = (10, 50)
    org_out = (750, 50)
    left_line = (10, 100)
    right_line = (750, 100)
    fontScale = 1

    # Start the video processing
    stframe = st.empty()

    for frame in video.iter_frames():
        vleft, vright = 0, 0
        big_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = big_frame.shape[:2]
        detection_frame = big_frame.copy()

        # Làm đen các vùng ngoài phạm vi dọc đã chỉ định
        detection_frame[:xv1, :] = 0
        detection_frame[xv2:, :] = 0

        result = model.predict(big_frame, conf=0.5, verbose=False)
        processed_frame = result[0].plot(line_width=1)

        processed_frame[:xv1, :] = big_frame[:xv1, :].copy()
        processed_frame[xv2:, :] = big_frame[xv2:, :].copy()

        # Vẽ các tứ giác trên khung hình đã xử lý
        cv2.polylines(big_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(big_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

        if len(result):
            result = result[0]
            names = result.names
            detect = []
            for box in result.boxes:
                x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                xc = x1 + (x2 - x1) // 2
                yc = y1 + (y2 - y1) // 2
                conf = box.conf.item()
                cls = int(box.cls.item())
                detect.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

                if xc < lane_threshold and 350 < yc < 630:
                    vleft += 1
                elif xc > lane_threshold and 350 < yc < 630:
                    vright += 1

            tracks = tracker.update_tracks(detect, frame=big_frame)

            for i, track in enumerate(tracks):
                if track.is_confirmed() and track.det_conf:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = list(map(int, ltrb))

                    yc = y1 + abs(y2 - y1) // 2
                    xc = x1 + abs(x2 - x1) // 2
                    xy = (xc, yc)
                    cv2.circle(big_frame, (xc, yc), 2, (0, 255, 0), 2)

                    track_id = track.track_id
                    name = names[track.det_class]
                    confidence = track.det_conf

                    if track_id not in track_dict.keys():
                        track_dict[track_id] = 0

                    label = f"{str(track_id)} {name} {confidence:.2f}"

                    cv2.rectangle(big_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(big_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if 300 <= yc <= 700 and xc <= 640 and track_dict[track_id] == 0:
                        track_dict[track_id] = 1
                        vehicles_left_lane += 1
                    elif 300 < yc < 700 and xc > 640 and track_dict[track_id] == 0:
                        track_dict[track_id] = 1
                        vehicles_right_lane += 1

            traffic_intensity_left = "Heavy" if vleft > traffic_flow else "Smooth"
            traffic_intensity_right = "Heavy" if vright > traffic_flow else "Smooth"

            cv2.putText(big_frame, f'Vehicles Left Lane: {vehicles_left_lane}', org_in, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big_frame, f'Vehicles Right Lane: {vehicles_right_lane}', org_out, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big_frame, f'traffic flow Left Lane: {traffic_intensity_left}', left_line, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big_frame, f'traffic flow Right Lane: {traffic_intensity_right}', right_line, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big_frame, f'vehicles in box: {vleft}', (10, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big_frame, f'vehicles in box: {vright}', (750, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        stframe.image(big_frame, channels="BGR")
        out.write(big_frame)

    out.release()
    cv2.destroyAllWindows()