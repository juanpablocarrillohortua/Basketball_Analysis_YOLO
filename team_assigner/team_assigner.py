from PIL import Image
import cv2

from transformers import CLIPProcessor, CLIPModel
import torch

import sys

sys.path.append('../')

from utils import read_stubs, save_stubs

class TeamAssigner:
    def __init__(self,
                 team_1_class_name="white shirt",
                 team_2_class_name="dark blue shirt",
                 refresh_rate = 50):
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.refresh_rate = refresh_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.player_team_dict = {}

    def load_model(self,
                   model_url='patrickjohncyh/fashion-clip',
                   processor_url='patrickjohncyh/fashion-clip'):
        self.model = CLIPModel.from_pretrained(model_url)
        self.processor = CLIPProcessor.from_pretrained(processor_url)
        self.model.to(self.device)

    def get_player_color(self, frame: list, bbox: list):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        #  Preprocess image to the pytorch format

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(
            text=classes,
            images=pil_image,
            return_tensors = 'pt',
            padding=True
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        return classes[probs.argmax(dim=1)[0]]

    def get_player_team(self, frame, player_bbox, player_id):
        """helper function to get the team id, it's only is called once per player."""

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_player_teams_accross_frames(
            self,
            video_frames,
            player_tracks,
            read_from_stub=False,
            stub_path=None
            ):

        player_assignments = read_stubs(read_from_stub, stub_path)

        if player_assignments is not None:
            if len(player_assignments) == len(video_frames):
                return player_assignments

        self.load_model()

        player_assignments = []

        for frame_num, player_track in enumerate(player_tracks):
            player_assignments.append({})

            if frame_num % self.refresh_rate == 0:
                self.player_team_dict={} #  delete cahe in order to fix any misscalsification

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track['box'], player_id)
                player_assignments[frame_num][int(player_id)] = team

        save_stubs(stub_path, player_assignments)

        return player_assignments
