import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox


class BallAquisitionDetector:

    def __init__(
            self,
            possession_treshold = 50,
            min_frames = 11,
            containment_treshold = 0.8
            ):
        self.possession_treshold = possession_treshold
        self.min_frames = min_frames
        self.containment_treshold = containment_treshold

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox

        width = x2-x1
        height = y2-y1

        output_points = []

        if y1 < ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if x1 < ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
                    (x1, y1),               # top left corner
                    (x2, y1),               # top right corner
                    (x2, y2),               # bottom right corner
                    (x1, y2),               # bottom left corner
                    ((x1 + x2) // 2, y1),   # top center
                    ((x1 + x2) // 2, y2),   # bottom center
                    (x1, (y1 + y2) // 2),   # left center
                    (x2, (y1 + y2) // 2)    # right center
                ]

        return output_points

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_key_basketball_player_assignment_points(
            player_bbox,
            ball_center
        )

        return min(measure_distance(ball_center, key_point) for key_point in key_points)

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):

        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ball_area = (bx2-bx1)*(by2-by1)

        intersection_area = max(0, min(px2, bx2)-max(px1, bx1)) * \
                            max(0, min(py2, by2)-max(py1, by1))

        return intersection_area/ball_area

    def find_best_candidate_for_possesion(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Choose the player with the ball (the closest one).

        this methot gives priority to the overlapping.
        """
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('box', [])

            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(
                player_bbox,
                ball_bbox
            )

            min_distance = self.find_minimum_distance_to_ball(
                ball_center,
                player_bbox
            )

            if containment > self.containment_treshold:
                high_containment_players.append((player_id, containment))
            elif min_distance < self.possession_treshold:
                regular_distance_players.append((player_id, min_distance))

        # First priority high_containment_players

        if high_containment_players:
            return max(high_containment_players, key=lambda x: x[1])[0] # return player_id

        # Second regular_distance_players

        if regular_distance_players:
            return min(regular_distance_players, key = lambda x: x[1])[0]

        return -1 # no one has the ball

    def detect_ball_possesion(self, player_tracks, ball_tracks):
        num_frames = len(ball_tracks)
        possesion_list = [-1] * num_frames

        consecutive_possesion_count = {}

        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})

            if not ball_info:
                continue
            ball_bbox = ball_info.get('box', [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            best_player_id = self.find_best_candidate_for_possesion(
                    ball_center,
                    player_tracks[frame_num],
                    ball_bbox
                )
            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possesion_count.get(best_player_id, 0)+1
                consecutive_possesion_count = {
                    best_player_id: number_of_consecutive_frames
                    }

                if consecutive_possesion_count[best_player_id] >= self.min_frames:
                    possesion_list[frame_num] = int(best_player_id)
            else:
                consecutive_possesion_count = {}

        return possesion_list
