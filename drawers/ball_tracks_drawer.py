from .utils import draw_triangle


class BallTracksDrawer:
    def __init__(self, ball_pointer_color=(0, 255, 0), border_color=(0, 0, 0), thickness=2):
        self.ball_pointer_color = ball_pointer_color
        self.border_color = border_color
        self.thickness = thickness

    def draw(self, video_frames: list, tracks: list[dict]) -> list:
        output_video_frames = []

        for frame_number, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_number]

            # Draw ball pointer
            for _, track in ball_dict.items():
                bbox = track["box"]
                if bbox is None:
                    continue
                frame = draw_triangle(frame, bbox, self.ball_pointer_color, self.border_color, self.thickness)

            output_video_frames.append(frame)
        return output_video_frames