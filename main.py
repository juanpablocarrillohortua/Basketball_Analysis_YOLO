from utils import read_video, save_video
from trackers import PlayerTracker
from drawers import PlayerTracksDrawer

def main():
    # read video
    video_frames = read_video("input_videos/video_1.mp4")

    # initialize player tracker
    player_tracker = PlayerTracker("models/player_detector.pt")

    # get player tracks
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/player_track_stubs.pkl"
        )

    # Draw player tracks
    player_tracks_drawer = PlayerTracksDrawer()
    video_frames_with_tracks = player_tracks_drawer.draw(video_frames, player_tracks)

    # save video
    save_video(video_frames_with_tracks, "output_videos/output_track_test.mp4")
if __name__ == "__main__":
    main()
