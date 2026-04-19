from ultralytics import YOLO

model = YOLO(r"models/player_detector.pt")
model.to('cuda')

results = model.predict(
    source="input_videos/video_1.mp4",
    save=True,
    exist_ok=True
)

print(results)
print("="*10)
for box in results[0].boxes:
    print(box)
