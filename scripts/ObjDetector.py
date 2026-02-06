# 基于YOLO实现了一个目标检测器
import os, torch, cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from PIL import ImageDraw
from PIL import Image
# import requests
# from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMO_IMG_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'imgs', 'hand.jpg')
DEFAULT_DEMO_VIDEO_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'videos', '1.mp4')

class ObjDetector:
    def __init__(self, save_dir = os.path.join(SCRIPT_DIR, '..', 'src','imgs')):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.detector = YOLO(SCRIPT_DIR + '/../src/models/yolo26n.pt')
        self.detector.model.eval()
    
    def annotate_image(self, image):
        if isinstance(image, str):
            img_path = image
            image_pil = Image.open(img_path)
            annotated_path = os.path.join(
                self.save_dir,
                os.path.basename(img_path).replace(".jpg", "_obj_detection.png")
            )
            frame = np.array(image_pil)
        else:
            frame = np.array(image)
            annotated_path = os.path.join(self.save_dir, "obj_detection_image.png")

        results = self.detector(frame)
        annotated = self.draw_boxes(frame, results[0])

        Image.fromarray(annotated).save(annotated_path)
        return annotated, results
    
    
    
    def annotate_video(self, video_path=DEFAULT_DEMO_VIDEO_PATH, batch_size=8):
        output_path = os.path.join(
            self.save_dir.replace('imgs', 'videos'),
            os.path.basename(video_path).replace(".mp4", "_obj_detection.mp4")
        )

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        buffer_frames = []
        buffer_raw = []

        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer_frames.append(Image.fromarray(rgb))
                buffer_raw.append(frame)

                if len(buffer_frames) == batch_size:
                    results = self.detector(buffer_frames,verbose=False)

                    for raw, dets in zip(buffer_raw, results):
                        annotated = self.draw_boxes(raw, dets)
                        writer.write(annotated)

                    buffer_frames.clear()
                    buffer_raw.clear()
                    pbar.update(batch_size)

            
            if buffer_frames:
                results = self.detector(buffer_frames,verbose=False)
                for raw, dets in zip(buffer_raw, results):
                    annotated = self.draw_boxes(raw, dets)
                    writer.write(annotated)
                pbar.update(len(buffer_frames))

        cap.release()
        writer.release()

    def draw_boxes(self, frame_bgr, result):
        img = frame_bgr.copy()

        boxes = result.boxes
        if boxes is None:
            return img

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
            label = f"{names[cl]} {c:.2f}"

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2
            )
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        return img

# 测试代码
def main():
    detector = ObjDetector()
    # Test image annotation
    annotated_img, outputs = detector.annotate_image(DEFAULT_DEMO_IMG_PATH)
    # annotated_img.show()
    # print("Detections:", outputs)
    # Test video annotation
    detector.annotate_video(DEFAULT_DEMO_VIDEO_PATH)
    print("Annotated video saved.")

if __name__ == "__main__":
    main()
    pass
