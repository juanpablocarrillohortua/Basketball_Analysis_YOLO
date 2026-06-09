class PassAndInterceptionDetector:
    def __init__(self):
        pass

    def detect_passes_interceptions(self, ball_aquisition, player_assignment):
        passes = [-1] * len(ball_aquisition)
        interceptions = [-1] * len(ball_aquisition)
        prev_holder = -1 # Initialize with no one having the ball
        previous_frame = -1

        for frame in range(len(ball_aquisition)):
            if frame - 1 >= 0 and ball_aquisition[frame - 1] != -1: # If someone has the ball
                prev_holder = ball_aquisition[frame - 1]
                previous_frame = frame - 1
            current_holder = ball_aquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)
                if prev_team != -1 and current_team != -1 and prev_team == current_team:
                    passes[frame] = prev_team
                elif prev_team != -1 and current_team != -1 and prev_team != current_team:
                    interceptions[frame] = current_team  # this team stole the ball

        return passes, interceptions
