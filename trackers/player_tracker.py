"""Creates a PlayerTracker class that detects and tracks players in video frames using a YOLO model and ByteTrack tracker."""
from ultralytics import YOLO
import supervision as sv
import torch
import sys
sys.path.append("../")
from utils import save_stubs, read_stubs


class PlayerTracker:
    def __init__(self, model_path: str) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker = sv.ByteTrack()

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

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["Player"]:
                    tracks[frame_num][track_id] = {"box": bbox}
        save_stubs(stub_path, tracks)

        return tracks
