import supervision as sv
import numpy as np


class CourtKeyPointsDrawer:
    def __init__(self, color: str = "#FF7B00") -> None:
        self.keypoint_color = color

    def draw(self, frames, court_key_points):
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []
        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_key_points[index]
            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            # Draw labels
            # Convert PyTorch tensor to numpy array
            keypoints_numpy = keypoints.cpu().numpy()
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy)

            output_frames.append(annotated_frame)

        return output_frames