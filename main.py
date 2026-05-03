from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner

def main():
    # read video
    video_frames = read_video("input_videos/video_1.mp4")

    # initialize trackers
    player_tracker = PlayerTracker("models/player_detector.pt")
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

    # Assign Player Teama
    team_assigner = TeamAssigner(
        team_1_class_name='white basketball jersey',
        team_2_class_name='dark navy blue basketball jersey',
        refresh_rate=15)

    player_assignment = team_assigner.get_player_teams_accross_frames(
        video_frames,
        player_tracks,
        read_from_stub=True, # set it True for fast processing (without changes of logic, only draws)
        stub_path="stubs/player_assignment_stub.pkl"
        )

    # Draw player tracks
    player_tracks_drawer = PlayerTracksDrawer(
        team_1_color=(222, 250, 230),
        team_2_color=(134, 2, 250)
        )
    video_frames_with_tracks = player_tracks_drawer.draw(video_frames,
                                                         player_tracks,
                                                         player_assignment) #track the players and draw the bounding boxes

    # Draw ball pointer
    ball_tracks_drawer = BallTracksDrawer()
    video_frames_with_tracks = ball_tracks_drawer.draw(video_frames_with_tracks,
                                                       ball_tracks) #track the ball and draw the pointer

    # save video
    save_video(video_frames_with_tracks, "output_videos/output_team_seg.mp4")
if __name__ == "__main__":
    main()
