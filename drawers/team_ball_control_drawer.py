import numpy as np
import cv2


class TeamBallControlDrawer:

    def __init__(
            self,
            font_size=0.7,
            font_thickness = 2,
            alpha = 0.8
            ):
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.alpha = alpha

    def get_team_ball_control(
            self,
            player_assignment, # player teams
            ball_aquisition  # who have the ball
            ) -> np.array:
        team_ball_contol = []
        for player_assignment_frame, ball_aquisition_frame in zip(player_assignment, ball_aquisition):

            if ball_aquisition_frame == -1 or ball_aquisition_frame not in player_assignment_frame:
                team_ball_contol.append(-1) # no one has the ball
                continue

            elif player_assignment_frame[ball_aquisition_frame] == 1:
                team_ball_contol.append(1)
            else:
                team_ball_contol.append(2)

        return np.array(team_ball_contol)

    def draw_frame(self, frame, frame_num, team_ball_control):

        overlay = frame.copy()

        frame_height, frame_width, _ = overlay.shape

        # --- Layout constants ---
        padding   = int(frame_width * 0.012)   # inner padding
        margin    = int(frame_width * 0.012)   # margin from edge
        font      = cv2.FONT_HERSHEY_DUPLEX
        font_scale     = self.font_size
        font_thickness = self.font_thickness

        # --- Ball control logic (unchanged) ---
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        frames_with_possession = team_ball_control_till_frame[team_ball_control_till_frame != -1]
        total_possession_frames = frames_with_possession.shape[0]
        if total_possession_frames > 0:
            team_1_num_frames = frames_with_possession[frames_with_possession == 1].shape[0]
            team_2_num_frames = frames_with_possession[frames_with_possession == 2].shape[0]

            team_1 = (team_1_num_frames / total_possession_frames) * 100
            team_2 = (team_2_num_frames / total_possession_frames) * 100
        else:
            # Porcentajes por defecto si aún no ha iniciado la posesión de ningún equipo
            team_1 = 0.0
            team_2 = 0.0

        lines = [
            f"Team 1",
            f"{team_1:.1f}%",
            f"Team 2",
            f"{team_2:.1f}%",
        ]

        # --- Measure text to size the rectangle dynamically ---
        line_sizes = [
            cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            for line in lines
        ]
        line_heights = [
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][1]
            for line in lines
        ]

        line_spacing = int(frame_height * 0.012)
        block_width  = max(w for w, _ in line_sizes) + padding * 2
        block_height = sum(line_heights) + line_spacing * (len(lines) - 1) + padding * 2

        # --- Rectangle: top-right corner ---
        rect_x2 = frame_width - margin
        rect_x1 = rect_x2 - block_width
        rect_y1 = margin
        rect_y2 = rect_y1 + block_height

        # --- Draw semi-transparent background ---
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # --- Thin accent border ---
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 200, 200), 1)

        # --- Draw text lines, fully inside the rectangle ---
        cursor_y = rect_y1 + padding
        colors = [(255, 255, 255), (100, 220, 255), (255, 255, 255), (100, 255, 160)]

        for i, (line, h) in enumerate(zip(lines, line_heights)):
            cursor_y += h
            # Clamp so text never goes outside the rectangle
            safe_y = min(cursor_y, rect_y2 - padding)
            cv2.putText(
                frame,
                line,
                (rect_x1 + padding, safe_y),
                font,
                font_scale,
                colors[i],
                font_thickness,
                cv2.LINE_AA
            )
            cursor_y += line_spacing

        return frame



    def draw(
            self,
            video_frames,
            player_assignment, # player teams
            ball_aquisition  # who have the ball
            ):

        team_ball_control = self.get_team_ball_control(
            player_assignment,
            ball_aquisition
        )

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue

            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)

        return output_video_frames

