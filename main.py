from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer

def main():
    # read video
    video_frames = read_video("input_videos/video_1.mp4")

    # initialize trackers
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/player_detector.pt")

    # get player tracks
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/player_track_stubs.pkl"
        )

    # get ball tracks
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub= True,
        stub_path="stubs/ball_track_stubs.pkl"
    )

    # Draw player tracks
    player_tracks_drawer = PlayerTracksDrawer()
    video_frames_with_tracks = player_tracks_drawer.draw(video_frames,
                                                         player_tracks) #track the players and draw the bounding boxes

    # Draw ball pointer
    ball_tracks_drawer = BallTracksDrawer()
    video_frames_with_tracks = ball_tracks_drawer.draw(video_frames_with_tracks,
                                                       ball_tracks) #track the ball and draw the pointer

    # save video
    save_video(video_frames_with_tracks, "output_videos/output_track_ball_test.mp4")
if __name__ == "__main__":
    main()
