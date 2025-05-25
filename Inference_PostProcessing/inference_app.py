import os
import cv2
import time
import math
import torch
import argparse
import pyttsx3 
import threading
import numpy as np
import tensorflow as tf

from PIL import Image
from queue import Queue, SimpleQueue 
from collections import defaultdict, deque
from shapely.geometry import box as shapely_box
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from lane_segmentation import UNet

# We are going to import the voice synthesis library and create a thread for it
engine = pyttsx3.init()
speak_queue = SimpleQueue()

def speak_worker():
    while True:
        msg = speak_queue.get()
        if msg is None:
            break
        engine.say(msg)
        engine.runAndWait()

tts_thread = threading.Thread(target = speak_worker, daemon = True)
tts_thread.start()

# We are going to use the following overlays to display on the video feed
compass_img = cv2.imread("C:/Users/tonyh_yxuq8za/Desktop/HACD/Inference_PostProcessing/compass.png", cv2.IMREAD_UNCHANGED)
warning_overlay = cv2.imread("C:/Users/tonyh_yxuq8za/Desktop/HACD/Inference_PostProcessing/warning.png", cv2.IMREAD_UNCHANGED)
if warning_overlay is not None and warning_overlay.shape[2] == 4:
    warning_overlay = cv2.resize(warning_overlay, (450, 300))

# We are going to use the following models for inference
# Load YOLOv8 model for vehicle detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO("C:/Users/tonyh_yxuq8za/Desktop/TONY_AI/runs/detect/vehicle_yolo2/weights/best.pt").to(device)
tracker = DeepSort(max_age=10, n_init=3, max_cosine_distance=0.3)

# We are loading the lane segmentation model
lane_model = UNet().to(device)
lane_model.load_state_dict(torch.load("C:/Users/tonyh_yxuq8za/Desktop/HACD/Lane_Best_Model/unet.pth", map_location=device)) 
lane_model.eval()

lane_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# We are loading the TensorFlow Lite models for brake and turn signal detection
def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

brake_interpreter = load_interpreter("C:/Users/tonyh_yxuq8za/Desktop/HACD/Brake_Best_Model/best_model_brake.tflite")
brake_input_index = brake_interpreter.get_input_details()[0]['index']
brake_output_index = brake_interpreter.get_output_details()[0]['index']

turn_interpreter = load_interpreter("C:/Users/tonyh_yxuq8za/Desktop/HACD/Indicator_Best_Model/best_model_indicators_mainn.tflite")
turn_input_index = turn_interpreter.get_input_details()[0]['index']
turn_output_index = turn_interpreter.get_output_details()[0]['index']

# We are going to create the queues for frame processing and results
frame_queue = Queue(maxsize = 5)
result_queue = Queue(maxsize = 5)
turn_signal_history = defaultdict(lambda: deque(maxlen = 20))
brake_status_history = defaultdict(lambda: deque(maxlen = 30))
indicator_crop_cache = {}
fps_deque = deque(maxlen = 10)

last_speak_time = 0
speak_interval = 5
warning_display_until = 0
lane_offset_history = deque(maxlen=10)
last_ldws_status = ""
last_lkas_status = ""
last_lane_status_time = 0

total_frames = 0
start_time_global = None

# We are going to define some utility functions for preprocessing and postprocessing
def preprocess_img(img, size=(64, 64)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

# This is to postprocess the turn signal and brake status based on the history of detections
def postprocess_turn_signal(history, fps=20):
    count = {'LEFT': 0, 'RIGHT': 0, 'OFF': 0}
    for state in history:
        count[state] += 1
    if count['LEFT'] >= fps * 0.2:
        return 'Turn: LEFT'
    elif count['RIGHT'] >= fps * 0.2:
        return 'Turn: RIGHT'
    elif count['OFF'] >= fps * 0.6:
        return 'Turn: OFF'
    return 'Turn: ?'

def postprocess_brake_status(history, fps=20):
    count = {'ON': 0, 'OFF': 0}
    for state in history:
        count[state] += 1
    if count['ON'] >= fps * 0.1:
        return 'Brake: ON'
    elif count['OFF'] >= fps * 0.5:
        return 'Brake: OFF'
    return 'Brake: ?'

# This is to calculate the IoU for two bounding boxes
def iou(box1, box2):
    return shapely_box(*box1).intersection(shapely_box(*box2)).area / shapely_box(*box1).union(shapely_box(*box2)).area

def is_indicator_flickering(prev_crop, curr_crop, threshold=20, flicker_ratio=0.02):
    if prev_crop is None or prev_crop.shape != curr_crop.shape:
        return True
    diff = cv2.absdiff(cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY), cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY))
    return (np.sum(diff > threshold) / diff.size) > flicker_ratio

def light_intensity_check(crop):
    h, w = crop.shape[:2]
    left_half = crop[:, :w//2]
    right_half = crop[:, w//2:]
    left_brightness = np.sum(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)) / (h * w / 2)
    right_brightness = np.sum(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)) / (h * w / 2)
    diff_ratio = abs(left_brightness - right_brightness) / max(left_brightness, right_brightness, 1)
    return 'OFF' if diff_ratio < 0.05 else None

def estimate_distance(bbox_width):
    focal_length = 500
    known_width = 2.0
    return (known_width * focal_length) / bbox_width if bbox_width > 0 else float('inf')

def get_color_by_distance(distance):
    if distance < 5:
        return (0, 0, 255)
    elif distance < 10:
        return (0, 165, 255)
    return (0, 255, 0)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        alpha_rotated = cv2.warpAffine(alpha, M, (w, h), flags = cv2.INTER_LINEAR)
        rotated = cv2.merge((rotated[:, :, :3], alpha_rotated))
    return rotated

def overlay_image_alpha(img, img_overlay, x, y):
    h, w = img_overlay.shape[:2]
    if img_overlay.shape[2] == 4:
        alpha_mask = img_overlay[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+h, x:x+w, c] = (
                img[y:y+h, x:x+w, c] * (1. - alpha_mask) +
                img_overlay[:, :, c] * alpha_mask
            )

# We are going to read frames from the video source
def read_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_queue.put(frame)

# We are going to process the inference in a separate thread
def process_inference():
    global last_ldws_status, last_lkas_status, last_lane_status_time, ldws_color, lkas_color
    global last_speak_time, warning_display_until
    lane_mask_cache = None
    lane_frame_counter = 0
    loop_start_time = time.time()

    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        
        # We are going to process the lane segmentation every 5 frames
        lane_frame_counter += 1
        if lane_frame_counter % 5 == 0: 
            lane_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lane_tensor = lane_transform(lane_input).unsqueeze(0).to(device)

            with torch.no_grad():
                lane_output = lane_model(lane_tensor)
                lane_mask = torch.sigmoid(lane_output)[0, 0].cpu().numpy()

            lane_mask = (lane_mask > 0.5).astype(np.uint8) * 255
            lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))

            kernel = np.ones((5, 5), np.uint8)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                lane_mask_cache = np.zeros_like(lane_mask)
                cv2.drawContours(lane_mask_cache, [largest], -1, 255, thickness=cv2.FILLED)
            else:
                lane_mask_cache = lane_mask.copy()

        if lane_mask_cache is not None:
            lane_overlay = cv2.cvtColor(lane_mask_cache, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0, lane_overlay, 0.3, 0)
        # We are going to process the vehicle detection using our YOLOv8m saved best model to detect the rear of the cars
        start_time = time.time()
        offset_y = int(frame.shape[0] * 0.3)
        frame_roi = frame[offset_y:, :]
        results = yolo_model.predict(frame_roi, imgsz = 640, conf = 0.5)[0]
        # We are going to process the results and filter out the detections
        detections = []
        for r in results.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            y1 += offset_y
            y2 += offset_y
            box_w, box_h = x2 - x1, y2 - y1
            aspect_ratio = box_w / (box_h + 1e-5)
            if int(r.cls[0]) == 0 and box_w >= 30 and box_h >= 30 and 0.2 <= aspect_ratio <= 4:
                detections.append(([x1, y1, box_w, box_h], float(r.conf[0]), 'vehicle'))
        # We are going to filter out overlapping detections using IoU
        filtered = []
        for det in detections:
            if all(iou([*det[0][:2], det[0][0]+det[0][2], det[0][1]+det[0][3]],
                       [*fd[0][:2], fd[0][0]+fd[0][2], fd[0][1]+fd[0][3]]) < 0.8 for fd in filtered):
                filtered.append(det)

        tracks = tracker.update_tracks(filtered, frame = frame)
        crops, track_info = [], []
        h_frame, w_frame = frame.shape[:2]
        # We are going to process each track and extract crops for brake and turn signal detection
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            pad = 5
            l, r = max(0, l - pad), min(r + pad, w_frame)
            t, b = max(0, t - pad), min(b + pad, h_frame)
            crop = frame[t:b, l:r]
            if crop.size and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                crops.append(preprocess_img(crop))
                track_info.append((track_id, l, t, r, b, crop))
        # We are going to process the crops for brake and turn signal detection
        if crops:
            batch = np.stack(crops)
            try:
                brake_interpreter.resize_tensor_input(brake_input_index, batch.shape)
                brake_interpreter.allocate_tensors()
                brake_interpreter.set_tensor(brake_input_index, batch)
                brake_interpreter.invoke()
                brake_results = brake_interpreter.get_tensor(brake_output_index)
            except:
                brake_results = [None] * len(batch)

            try:
                turn_interpreter.resize_tensor_input(turn_input_index, batch.shape)
                turn_interpreter.allocate_tensors()
                turn_interpreter.set_tensor(turn_input_index, batch)
                turn_interpreter.invoke()
                turn_results = turn_interpreter.get_tensor(turn_output_index)
            except:
                turn_results = [None] * len(batch)

            for i, (track_id, l, t, r, b, crop) in enumerate(track_info):
                current_time = time.time()
                bbox_w = r - l
                distance = estimate_distance(bbox_w)
                distance_text = f'Dist: {distance:.1f} m'
                box_color = get_color_by_distance(distance)

                # We are going to process the brake status detection
                brake_status = 'Brake: ?'
                if brake_results[i] is not None:
                    label = 'ON' if np.argmax(brake_results[i]) == 1 else 'OFF'
                    history = brake_status_history[track_id]
                    history.append(label)

                    on_ratio = history.count('ON') / len(history)
                    if on_ratio > 0.3:
                        brake_status = 'Brake: ON'
                    elif on_ratio < 0.2:
                        brake_status = 'Brake: OFF'

                # We are going to process the turn signal detection
                turn_status = 'Turn: ?'
                if turn_results[i] is not None:
                    probs = turn_results[i]
                    sorted_probs = sorted(zip(['LEFT', 'RIGHT', 'OFF'], probs), key=lambda x: x[1], reverse=True)
                    top_label, top_conf = sorted_probs[0]
                    second_label, second_conf = sorted_probs[1]

                    if top_conf >= 0.65 and (top_conf - second_conf) >= 0.15:
                        raw = top_label
                    else:
                        raw = 'OFF'

                    # We are going to check for flickering indicators
                    if raw in ['LEFT', 'RIGHT']:
                        prev_crop = indicator_crop_cache.get(track_id)
                        if prev_crop is not None and not is_indicator_flickering(prev_crop, crop):
                            raw = 'OFF'
                        indicator_crop_cache[track_id] = crop

                    # We are going to check for light intensity differences
                    if raw in ['LEFT', 'RIGHT']:
                        override = light_intensity_check(crop)
                        if override == 'OFF':
                            raw = 'OFF'

                    turn_signal_history[track_id].append(raw)

                    # We are going to determine the turn status based on the history
                    history = turn_signal_history[track_id]
                    scores = {'LEFT': 0, 'RIGHT': 0, 'OFF': 0}
                    for state in history:
                        scores[state] += 1
                    if scores['LEFT'] >= 5:
                        turn_status = 'Turn: LEFT'
                    elif scores['RIGHT'] >= 5:
                        turn_status = 'Turn: RIGHT'
                    elif scores['OFF'] >= 10:
                        turn_status = 'Turn: OFF'
                    else:
                        turn_status = 'Turn: ?'
                        
                show_warning = False
                if lane_mask_cache is not None and distance < 9 and brake_status == 'Brake: ON':
                    check_y = min(b - 1, lane_mask_cache.shape[0] - 1)
                    check_x = int((l + r) / 2)
                    if lane_mask_cache[check_y, check_x] > 0:
                        if current_time - last_speak_time > speak_interval:
                            speak_queue.put("Hey Boss, heads up! The car in front is hitting the brakes.")
                            last_speak_time = current_time
                            warning_display_until = current_time + 1
                            show_warning = True
                    
                # It is to check if the vehicle is turning into our lane
                vehicle_center_x = (l + r) / 2
                frame_center_x = frame.shape[1] / 2
                from_left_side = vehicle_center_x < frame_center_x
                from_right_side = vehicle_center_x > frame_center_x

                # The system will warn the driver if a vehicle is turning into their lane
                if distance < 8 and (current_time - last_speak_time) > speak_interval:
                    if turn_status == 'Turn: LEFT' and from_right_side:
                        speak_queue.put("BOSS, SLOW DOWN, THE CAR FROM THE RIGHT TURNING INTO YOUR LANE!")
                        last_speak_time = current_time
                        warning_display_until = current_time + 1
                        show_warning = True

                    elif turn_status == 'Turn: RIGHT' and from_left_side:
                        speak_queue.put("BOSS, SLOW DOWN, THE CAR FROM THE LEFT TURNING INTO YOUR LANE!")
                        last_speak_time = current_time
                        warning_display_until = current_time + 1
                        show_warning = True

                # This is to display the warning overlay
                if time.time() < warning_display_until and warning_overlay is not None:
                    overlay_image_alpha(frame, warning_overlay,
                                        x = frame.shape[1] - 460,
                                        y = frame.shape[0] - 320)


                cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
                cv2.putText(frame, f'ID {track_id}', (l, t - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, brake_status, (l, t - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, turn_status, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, distance_text, (l, t + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # We are going to display the FPS and time on the frame
        now = time.localtime()
        time_str = time.strftime("%Y-%m-%d  %H:%M:%S", now)
        text_size, _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        text_x, text_y = 30, 40
        cv2.putText(frame, time_str, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        direction_angle = 50 + 10 * math.sin(time.time())
        compass_size = 150
        rotated_compass = rotate_image(compass_img, direction_angle)
        rotated_compass = cv2.resize(rotated_compass, (compass_size, compass_size))
        overlay_image_alpha(frame, rotated_compass, x = 20, y = frame.shape[0] - compass_size - 20)
        
        frame_time = time.time() - start_time
        fps_deque.append(1.0 / frame_time)
        
        # This is to display the lane departure warning system and lane keeping assist system
        ldws_text, lkas_text = "Lane Departure Warning System: ", "Lane Keeping Assist System: "
        now_time = time.time()

        # This is to initialize the lane keeping assist system and lane departure warning system
        ldws_color, lkas_color = (0, 255, 0), (0, 255, 0)
        smoothed_offset = 0

        if lane_mask_cache is not None:
            lane_center_xs = np.where(np.any(lane_mask_cache > 0, axis=0))[0]
            if len(lane_center_xs) > 0:
                lane_center = (lane_center_xs[0] + lane_center_xs[-1]) // 2
                frame_center = frame.shape[1] // 2
                offset = lane_center - frame_center
                lane_offset_history.append(offset)
                smoothed_offset = sum(lane_offset_history) / len(lane_offset_history)

                # We are going to determine the lane departure warning system and lane keeping assist system status
                centered_thresh = 35
                drift_thresh = 50

                # This is to determine the lane departure warning system and lane keeping assist system status
                if abs(smoothed_offset) < centered_thresh:
                    current_ldws = "Good Lane Keeping"
                    current_lkas = "Keep Straight Ahead"
                    color_ldws = (0, 255, 0)
                    color_lkas = (0, 255, 0)
                elif smoothed_offset < -drift_thresh:
                    current_ldws = "Please Keep Left"
                    current_lkas = "To Be Determined ..."
                    color_ldws = (0, 0, 255)
                    color_lkas = (0, 255, 255)
                elif smoothed_offset > drift_thresh:
                    current_ldws = "Please Keep Right"
                    current_lkas = "To Be Determined ..."
                    color_ldws = (0, 0, 255)
                    color_lkas = (0, 255, 255)
                else:
                    current_ldws = last_ldws_status
                    current_lkas = last_lkas_status
                    color_ldws = ldws_color
                    color_lkas = lkas_color
            else:
                current_ldws = "Lane Undetected"
                current_lkas = "To Be Determined ..."
                color_ldws = (0, 0, 255)
                color_lkas = (0, 255, 255)
        else:
            current_ldws = "Lane Undetected"
            current_lkas = "To Be Determined ..."
            color_ldws = (0, 0, 255)
            color_lkas = (0, 255, 255)

        # This is to update the lane departure warning system and lane keeping assist system status
        if (current_ldws != last_ldws_status or current_lkas != last_lkas_status) or (now_time - last_lane_status_time > 0.5):
            last_ldws_status = current_ldws
            last_lkas_status = current_lkas
            ldws_color = color_ldws
            lkas_color = color_lkas
            last_lane_status_time = now_time

        ldws_text += last_ldws_status
        lkas_text += last_lkas_status

        # We are going to display the lane departure warning system and lane keeping assist system status on the frame
        cv2.rectangle(frame, (20, 120), (700, 200), (30, 30, 30), -1)
        cv2.putText(frame, ldws_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ldws_color, 2)
        cv2.putText(frame, lkas_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, lkas_color, 2)

        result_queue.put(frame)

# This is the main function to run the application
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type = str, default = "droidcam")
    args = parser.parse_args()

    cap = cv2.VideoCapture("http://192.168.1.103:4747/video" if args.source == "droidcam" else args.source) # This is to set the video source
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threads = [
        threading.Thread(target = read_frames, args = (cap,)),
        threading.Thread(target = process_inference)
    ]

    for t in threads:
        t.start()

    cv2.namedWindow("Inference Output", cv2.WINDOW_NORMAL)
    
    global total_frames, start_time_global
    total_frames = 0
    start_time_global = time.time()
    
    # We are going to display the frames and process the results
    while True:
        frame = result_queue.get()
        if frame is None:
            break
        
        total_frames += 1
        
        display_frame = cv2.resize(frame, (1280, 720))
        fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(display_frame, fps_text, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Inference Output", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # We are going to calculate the total time and average FPS
    end_time_global = time.time()
    elapsed_time = end_time_global - start_time_global
    avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
    print(f"Total Time: {elapsed_time:.2f}s | Total Frames: {total_frames} | Avg FPS: {avg_fps:.2f}")
    for t in threads:
        t.join()
    cap.release()
    cv2.destroyAllWindows()
    speak_queue.put(None)
    tts_thread.join()
    
if __name__ == "__main__":
    main()
