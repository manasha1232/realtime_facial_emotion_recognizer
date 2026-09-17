import os
import cv2
import numpy as np

def draw_synthetic_face(emotion="happy", width=400, height=400):
    """Generates a synthetic face image expressing a specified emotion."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    cx, cy = width // 2, height // 2
    face_r = 130
    
    # Draw Face Oval
    cv2.ellipse(img, (cx, cy), (face_r, int(face_r * 1.25)), 0, 0, 360, (210, 230, 250), -1)
    cv2.ellipse(img, (cx, cy), (face_r, int(face_r * 1.25)), 0, 0, 360, (140, 160, 180), 3)
    
    # Eyes
    eye_y = cy - 35
    eye_left_x = cx - 45
    eye_right_x = cx + 45
    
    if emotion == "surprised":
        # Wide open eyes
        cv2.circle(img, (eye_left_x, eye_y), 18, (255, 255, 255), -1)
        cv2.circle(img, (eye_left_x, eye_y), 8, (20, 20, 20), -1)
        cv2.circle(img, (eye_right_x, eye_y), 18, (255, 255, 255), -1)
        cv2.circle(img, (eye_right_x, eye_y), 8, (20, 20, 20), -1)
        # Raised eyebrows
        cv2.ellipse(img, (eye_left_x, eye_y - 28), (20, 10), 0, 180, 360, (40, 40, 40), 3)
        cv2.ellipse(img, (eye_right_x, eye_y - 28), (20, 10), 0, 180, 360, (40, 40, 40), 3)
        # O-shaped open mouth
        cv2.circle(img, (cx, cy + 60), 22, (30, 30, 30), -1)
        cv2.circle(img, (cx, cy + 60), 22, (200, 100, 100), 2)
    elif emotion == "sad":
        # Downward eyes
        cv2.circle(img, (eye_left_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_left_x, eye_y), 5, (20, 20, 20), -1)
        cv2.circle(img, (eye_right_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_right_x, eye_y), 5, (20, 20, 20), -1)
        # Sad eyebrows (angled up in center)
        cv2.line(img, (eye_left_x - 15, eye_y - 25), (eye_left_x + 15, eye_y - 15), (40, 40, 40), 3)
        cv2.line(img, (eye_right_x + 15, eye_y - 25), (eye_right_x - 15, eye_y - 15), (40, 40, 40), 3)
        # Downturned mouth frown
        cv2.ellipse(img, (cx, cy + 80), (40, 20), 0, 190, 350, (40, 40, 40), 4)
    elif emotion == "angry":
        # Narrowed eyes
        cv2.circle(img, (eye_left_x, eye_y), 10, (255, 255, 255), -1)
        cv2.circle(img, (eye_left_x, eye_y), 5, (20, 20, 20), -1)
        cv2.circle(img, (eye_right_x, eye_y), 10, (255, 255, 255), -1)
        cv2.circle(img, (eye_right_x, eye_y), 5, (20, 20, 20), -1)
        # V-shaped angry eyebrows
        cv2.line(img, (eye_left_x - 15, eye_y - 20), (eye_left_x + 15, eye_y - 10), (40, 40, 40), 4)
        cv2.line(img, (eye_right_x + 15, eye_y - 20), (eye_right_x - 15, eye_y - 10), (40, 40, 40), 4)
        # Tight straight mouth
        cv2.line(img, (cx - 30, cy + 60), (cx + 30, cy + 60), (40, 40, 40), 4)
    elif emotion == "neutral":
        # Normal eyes
        cv2.circle(img, (eye_left_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_left_x, eye_y), 6, (20, 20, 20), -1)
        cv2.circle(img, (eye_right_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_right_x, eye_y), 6, (20, 20, 20), -1)
        # Horizontal eyebrows & mouth
        cv2.line(img, (eye_left_x - 15, eye_y - 20), (eye_left_x + 15, eye_y - 20), (40, 40, 40), 3)
        cv2.line(img, (eye_right_x - 15, eye_y - 20), (eye_right_x + 15, eye_y - 20), (40, 40, 40), 3)
        cv2.line(img, (cx - 30, cy + 60), (cx + 30, cy + 60), (40, 40, 40), 3)
    else: # Happy (Default)
        # Smiling eyes
        cv2.circle(img, (eye_left_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_left_x, eye_y), 6, (20, 20, 20), -1)
        cv2.circle(img, (eye_right_x, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (eye_right_x, eye_y), 6, (20, 20, 20), -1)
        # Normal eyebrows
        cv2.ellipse(img, (eye_left_x, eye_y - 20), (18, 8), 0, 180, 360, (40, 40, 40), 3)
        cv2.ellipse(img, (eye_right_x, eye_y - 20), (18, 8), 0, 180, 360, (40, 40, 40), 3)
        # Big smile curve
        cv2.ellipse(img, (cx, cy + 45), (45, 30), 0, 0, 180, (40, 40, 40), 4)
        
    # Title Tag
    cv2.putText(img, f"EMOTION: {emotion.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
                
    return img

def generate_all_demo_faces(output_dir="input"):
    """Generates synthetic emotion images and a test video sequence in input/."""
    os.makedirs(output_dir, exist_ok=True)
    
    emotions = ["happy", "surprised", "sad", "angry", "neutral"]
    for emo in emotions:
        fimg = draw_synthetic_face(emotion=emo)
        cv2.imwrite(os.path.join(output_dir, f"sample_face_{emo}.jpg"), fimg)
        
    # Generate Synthetic Video (150 frames transitioning through emotions)
    video_path = os.path.join(output_dir, "sample_emotion_video.mp4")
    width, height, fps = 640, 480, 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        video_path = video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
    seq = [("happy", 30), ("surprised", 30), ("neutral", 30), ("sad", 30), ("angry", 30)]
    frame_idx = 0
    
    for emo, duration in seq:
        fimg = draw_synthetic_face(emotion=emo, width=width, height=height)
        for d in range(duration):
            frame_idx += 1
            frame = fimg.copy()
            cv2.putText(frame, f"EMOTION VIDEO SEQUENCE | FRAME: {frame_idx:03d}/150", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            out.write(frame)
            
    out.release()
    print(f"[OK] Synthetic faces & test video generated in '{output_dir}/'")

if __name__ == "__main__":
    generate_all_demo_faces()
