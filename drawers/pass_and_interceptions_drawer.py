import cv2
import numpy as np

class PassInterceptionDrawer:
    def __init__(
            self,
            font_size=0.7,
            font_thickness = 2,
            alpha = 0.8
            ):
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.alpha = alpha


    def draw_frame(self, frame, frame_num, passes, interceptions):

        overlay = frame.copy()

        frame_height, frame_width, _ = overlay.shape

        # --- Layout constants ---
        padding   = int(frame_width * 0.012)   # inner padding
        margin    = int(frame_width * 0.012)   # margin from edge
        font      = cv2.FONT_HERSHEY_DUPLEX
        font_scale     = self.font_size
        font_thickness = self.font_thickness

        # --- passes logic ---
        passes_till_frame = np.array(passes[:frame_num + 1])
        interceptions_till_frame = np.array(interceptions[:frame_num + 1])
        team_1_passes = np.sum(passes_till_frame == 1)
        team_2_passes = np.sum(passes_till_frame == 2)
        team_1_interceptions = np.sum(interceptions_till_frame == 1)
        team_2_interceptions = np.sum(interceptions_till_frame == 2)


        # ============ Logic for drawing the ball control overlay (unchanged) ============
        teams_data = [
                {"name": "Team 1", "passes": team_1_passes, "interceptions": team_1_interceptions, "color": (255, 255, 255)}, # Blanco
                {"name": "Team 2", "passes": team_2_passes, "interceptions": team_2_interceptions, "color": (80, 190, 245)}   # Amarillo/Dorado
            ]

        # --- Calcular tamaños exactos para el bloque ---
        # Medimos una sola línea plantilla para saber el ancho máximo del contenedor de forma dinámica
        sample_text = f"Team 2 - Passes: {max(team_1_passes, team_2_passes)} | Interceptions: {max(team_1_interceptions, team_2_interceptions)}"
        (text_w, text_h), _ = cv2.getTextSize(sample_text, font, font_scale, font_thickness)

        line_spacing = int(frame_height * 0.015)
        block_width = text_w + padding * 2
        block_height = (text_h * 2) + line_spacing + (padding * 2)

        # --- Posicionamiento del Rectángulo (SOLUCIÓN AL ESPACIO EXTRA) ---
        # Dejamos los 100 px a la derecha para el "ball pointer", pero calculamos x1 usando EXACTAMENTE el block_width
        rect_x2 = frame_width - margin - 150
        rect_x1 = rect_x2 - block_width
        rect_y1 = margin
        rect_y2 = rect_y1 + block_height

        # --- Dibujar fondo semi-transparente ---
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # --- Borde exterior fino (Gris claro) ---
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 200, 200), 1)

        # --- Dibujar texto segmentado por colores ---
        cursor_y = rect_y1 + padding + text_h

        for team in teams_data:
            cursor_x = rect_x1 + padding

            # Segmento 1: "Team X - Passes: Y" (Color del equipo)
            seg1 = f"{team['name']} - Passes: {team['passes']}"
            cv2.putText(frame, seg1, (cursor_x, cursor_y), font, font_scale, team['color'], font_thickness, cv2.LINE_AA)

            # Avanzar cursor horizontalmente usando el ancho del segmento anterior
            seg1_w, _ = cv2.getTextSize(seg1, font, font_scale, font_thickness)[0]
            cursor_x += seg1_w

            # Segmento 2: " | " (Separador en color del equipo o blanco)
            seg_sep = " | "
            cv2.putText(frame, seg_sep, (cursor_x, cursor_y), font, font_scale, team['color'], font_thickness, cv2.LINE_AA)
            sep_w, _ = cv2.getTextSize(seg_sep, font, font_scale, font_thickness)[0]
            cursor_x += sep_w

            # Segmento 3: "Interceptions: Z" (Color del equipo)
            seg2 = f"Interceptions: {team['interceptions']}"
            cv2.putText(frame, seg2, (cursor_x, cursor_y), font, font_scale, team['color'], font_thickness, cv2.LINE_AA)

            # Bajar a la siguiente línea
            cursor_y += text_h + line_spacing

        return frame


    def draw(self, video_frames, passes, interceptions):
        output_video_frames = []

        for frame_number, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(frame, frame_number, passes, interceptions)
            output_video_frames.append(frame_drawn)

        return output_video_frames