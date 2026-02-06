# this code is used to mark handpose keypoints on images and videos using MediaPipe


import numpy as np

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import os


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

Red = (0, 0, 255)
Green = (0, 255, 0)
Blue = (255, 0, 0)
DotSize = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'models', 'hand_landmarker.task')
DEFAULT_DEMO_IMG_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'imgs', 'hand.jpg')
DEFAULT_DEMO_VIDEO_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'videos', '1.mp4')

class HandposeMarker:
    def __init__(self,model_path = DEFAULT_MODEL_PATH,num_hands=2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=num_hands)
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def _draw_handpose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(input_image)

        h, w, _ = frame.shape

        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

                        pts = []
                        for lm in hand_landmarks:
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            pts.append((x, y))
                            cv2.circle(frame, (x, y), DotSize, Red, -1)

                        for start_idx, end_idx in HAND_CONNECTIONS:
                            cv2.line(
                                frame,
                                pts[start_idx],
                                pts[end_idx],
                                Green,
                                2
                            )
        return frame
    
    def annotate_image(self, image):
        return self._draw_handpose(image)
    
    def mark_pose(self, video_path, output_path = None):
        
        if output_path is None:
            output_path = video_path.replace(".mp4", "_handpose.mp4")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )

        frame_id = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self._draw_handpose(frame)

                cv2.putText(
                    annotated, f"Frame {frame_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                writer.write(annotated)
                frame_id += 1
                pbar.update(1)

        cap.release()
        writer.release()
    


# demos

def demo():
    handposeMarker = HandposeMarker()
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("IMG_PATH:", DEFAULT_DEMO_IMG_PATH)
    print("VIDEO_PATH:", DEFAULT_DEMO_VIDEO_PATH)
    img = cv2.imread(DEFAULT_DEMO_IMG_PATH)
    output_img = handposeMarker.annotate_image(img)
    output_img_path = DEFAULT_DEMO_IMG_PATH.replace(".jpg", "_handpose.jpg")
    cv2.imwrite(output_img_path, output_img)
    print("Annotated image saved")
    handposeMarker.mark_pose(DEFAULT_DEMO_VIDEO_PATH)
    print("Handpose video saved.")


if __name__ == "__main__":
    demo()
    pass
 