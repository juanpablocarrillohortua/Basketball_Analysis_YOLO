
from typing import Any
from .utils import draw_ellipse

import torch

class PlayerTracksDrawer:
    def __init__(self, color=(0, 0, 255), thickness=2):
        self.color = color
        self.thickness = thickness

    def draw(self, video_frames: list, tracks: list[dict]) -> list:
        output_video_frames = []

        for frame_number, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_number]

            # Draw player tracks
            for track_id, player in player_dict.items():
                frame = draw_ellipse(frame, player["box"], color=self.color, track_id=track_id)


            output_video_frames.append(frame)
        return output_video_frames
