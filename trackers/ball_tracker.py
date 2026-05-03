"""Contains the BallTracker class, which is responsible for detecting and tracking the basketball in video frames using a YOLO model."""
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
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


    def remove_wrong_detections(self, ball_positions):

        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_bbox = ball_positions[i].get(1, {}).get('box', [])

            if len(current_bbox) == 0:
                continue

            if last_good_frame_index == -1:  # first detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('box', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # calculate distance between last good bbox and the currente one

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('box', []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing detections

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"box": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions



