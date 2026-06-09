from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector

def main():
    # read video
    video_frames = read_video("input_videos/video_3.mp4")

    # initialize trackers
    player_tracker = PlayerTracker("models/player_detector_2.pt")
    ball_tracker = BallTracker("models/player_detector.pt")

    # get player tracks
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,  # set it True for fast processing (without changes of logic, only draws)
        stub_path="stubs/player_track_stubs.pkl"
        )

    # get ball tracks
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub= True,  # set it True for fast processing (without changes of logic, only draws)
        stub_path="stubs/ball_track_stubs.pkl"
    )

    # remove wrong ball detections

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolate missing detections
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Team
    team_assigner = TeamAssigner(
        team_1_class_name='white shirt',
        # team_2_class_name='dark blue shirt',
        team_2_class_name='red shirt',
        )

    player_assignment = team_assigner.get_player_teams_accross_frames(
        video_frames,
        player_tracks,
        read_from_stub=True, # set it True for fast processing (without changes of logic, only draws)
        stub_path="stubs/player_assignment_stub.pkl",
        )

    #  Ball aquisition
    ball_aquisition_detector = BallAquisitionDetector(
        min_frames=13,
        possession_treshold=50,
        containment_treshold=0.8
        )
    ball_aquisition = ball_aquisition_detector.detect_ball_possesion(
        player_tracks,
        ball_tracks
    )

    # Pass and Interception Detection
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes, interceptions = pass_and_interception_detector.detect_passes_interceptions(
        ball_aquisition,
        player_assignment
    )

    # Draw player tracks
    player_tracks_drawer = PlayerTracksDrawer(
        team_1_color=(222, 250, 230),
        team_2_color=(134, 2, 250)
        )
    video_frames_with_tracks = player_tracks_drawer.draw(video_frames,
                                                         player_tracks,
                                                         player_assignment,
                                                         ball_aquisition
                                                         ) #track the players and draw the bounding boxes

    # Draw ball pointer
    ball_tracks_drawer = BallTracksDrawer()
    video_frames_with_tracks = ball_tracks_drawer.draw(video_frames_with_tracks,
                                                       ball_tracks) #track the ball and draw the pointer

    # Draw team control overlay
    team_ball_control_drawer = TeamBallControlDrawer()
    video_frames_with_tracks = team_ball_control_drawer.draw(video_frames_with_tracks,
                                                        player_assignment,
                                                        ball_aquisition)
    # Draw passes and interceptions
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    video_frames_with_tracks = pass_and_interceptions_drawer.draw(video_frames_with_tracks,
                                                        passes,
                                                        interceptions)


    # save video
    save_video(video_frames_with_tracks, "output_videos/pass_and_interception_test_video_3.mp4")
if __name__ == "__main__":
    main()
