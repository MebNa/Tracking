import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit as st

# Load the model
model = YOLO("./model/best.pt")
tracker = DeepSort(max_age=30)

# Define the lanes
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for lane cutting and threshold
xv1, xv2 = 325, 635
lane_threshold = 609

# Define the threshold for traffic flow
traffic_flow = 2

# Initialize variables
track_dict = {}
vehicles_left_lane = 0
vehicles_right_lane = 0

# Define font for text
font = cv2.FONT_HERSHEY_SIMPLEX
org_in = (10, 50)
org_out = (750, 50)
left_line = (10, 100)
right_line = (750, 100)
fontScale = 1

# Streamlit app
st.title("Traffic Monitoring System")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # Read the video file
    cap = cv2.VideoCapture(uploaded_file)

    # Get video properties
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo3.avi', fourcc, 20.0, (width, height))

    # Start the video processing loop
    while cap.isOpened():
        ret, big_frame = cap.read()
        if not ret:
            break

        # Process the frame
        vleft, vright = 0, 0
        detection_frame = big_frame.copy()

        # Black out areas outside the specified vertical range
        detection_frame[:xv1, :] = 0
        detection_frame[xv2:, :] = 0

        # Perform object detection using YOLO
        result = model.predict(big_frame, conf=0.5, verbose=False)
        processed_frame = result[0].plot(line_width=1)

        # Restore original frame content for the blacked-out areas
        processed_frame[:xv1, :] = big_frame[:xv1, :].copy()
        processed_frame[xv2:, :] = big_frame[xv2:, :].copy()

        # Draw lanes on the frame
        cv2.polylines(big_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(big_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

        if len(result):
            result = result[0]
            names = result.names
            detect = []

            # Process detections
            for box in result.boxes:
                x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                xc = x1 + (x2 - x1) // 2
                yc = y1 + (y2 - y1) // 2
                conf = box.conf.item()
                cls = int(box.cls.item())
                detect.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

                # Count vehicles in each lane
                if xc < lane_threshold and 350 < yc < 630:
                    vleft += 1
                elif xc > lane_threshold and 350 < yc < 630:
                    vright += 1

            # Update the tracker with detected objects
            tracks = tracker.update_tracks(detect, frame=big_frame)

            # Draw bounding boxes and track IDs
            for i, track in enumerate(tracks):
                if track.is_confirmed() and track.det_conf:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = list(map(int, ltrb))

                    yc = y1 + abs(y2 - y1) // 2
                    xc = x1 + abs(x2 - x1) // 2
                    xy = (xc, yc)
                    x_y = cv2.circle(big_frame, (xc, yc), (2), (0, 255, 0), 2)

                    track_id = track.track_id
                    name = names[track.det_class]
                    confidence = track.det_conf

                    if track_id not in track_dict.keys():
                        track_dict[track_id] = 0

                    label = f"{str(track_id)} {name} {confidence:.2f}"

                    cv2.rectangle(big_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(big_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Count vehicles entering each lane
                    if 300 <= yc <= 700 and xc <= 640 and track_dict[track_id] == 0:
                        track_dict[track_id] = 1
                        vehicles_left_lane += 1
                    elif 300 < yc < 700 and xc > 640 and track_dict[track_id] == 0:
                        track_dict[track_id] = 1
                        vehicles_right_lane += 1

        # Determine traffic intensity for each lane
        traffic_intensity_left = "Heavy" if vleft > traffic_flow else "Smooth"
        traffic_intensity_right = "Heavy" if vright > traffic_flow else "Smooth"

        # Display traffic information on the frame
        cv2.putText(big_frame, f'Vehicles Left Lane: {vehicles_left_lane}', org_in, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Vehicles Right Lane: {vehicles_right_lane}', org_out, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Traffic Flow Left Lane: {traffic_intensity_left}', left_line, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Traffic Flow Right Lane: {traffic_intensity_right}', right_line, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Vehicles in Box: {vleft}', (10, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Vehicles in Box: {vright}', (750, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the processed frame
        st.image(big_frame, channels="BGR", use_column_width=True)

        # Write the frame to the output video file
        out.write(big_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(8) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()