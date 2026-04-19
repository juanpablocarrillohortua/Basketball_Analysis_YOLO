"""Contains the BallTracker class, which is responsible for detecting and tracking the basketball in video frames using a YOLO model."""
from ultralytics import YOLO
import supervision as sv
import torch
import sys
sys.path.append("../")
from utils import save_stubs, read_stubs

class BallTracker:
    def __init__(self, model_path: str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect_frames(self, frames: list, batch_size: int = 16) -> list:
        """Detects objects in the given frames using the YOLO predict model."""
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections+= batch_detections
        return detections


    def get_object_tracks(
            self,
            frames: list,
            batch_size: int = 16,
            read_from_stub: bool = False,
            stub_path: str = None
            ) -> list:
        """
        Detects and tracks objects in the given frames. from the detections.

        Returns a list of tracks for each frame, only for the "Player" class.
        """
        # Try to skip detection and tracking if stubs are available
        tracks = read_stubs(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames): # Check if the number of tracks matches the number of frames (the same video)
                return tracks

        detections = self.detect_frames(frames, batch_size)
        tracks = []

        if detections:
            cls_names = detections[0].names
            cls_names_inv = {v: k for k, v in cls_names.items()}

        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for detection in detection_supervision:
                bbox = detection[0].tolist()
                cls_id = detection[3]
                confidence = detection[2]

                # get the bbox with the highest confidence for the "Ball" class
                if cls_id == cls_names_inv['Ball'] and confidence > max_confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence
            if chosen_bbox is not None:
                tracks[frame_num][1] = {"box": chosen_bbox,
                                        "confidence": max_confidence}

        save_stubs(stub_path, tracks)
        return tracks

