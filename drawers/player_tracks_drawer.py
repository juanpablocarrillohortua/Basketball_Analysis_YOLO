
from typing import Any
from .utils import draw_ellipse, draw_triangle

import torch

class PlayerTracksDrawer:
    def __init__(
            self,
            team_1_color: tuple[int, int, int],
            team_2_color: tuple[int, int, int],
            thickness=2
            ):
        self.default_player_team_id = 1,
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        self.thickness = thickness

    def draw(
            self,
            video_frames: list,
            tracks: list[dict],
            player_assignment: list,
            ball_aquisition: list
            ) -> list:
        output_video_frames = []

        for frame_number, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_number]

            player_assignment_for_frame = player_assignment[frame_number] # teams into the frame

            player_id_has_ball = ball_aquisition[frame_number]

            # Draw player tracks
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(
                    track_id,
                    self.default_player_team_id
                    )
                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color
                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, player['box'], color=(0, 0, 255))
                frame = draw_ellipse(frame, player["box"], color=color, track_id=track_id)


            output_video_frames.append(frame)
        return output_video_frames
