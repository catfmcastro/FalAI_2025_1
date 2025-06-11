import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time

# ---------------- MODEL DEFINITION ----------------
class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # bi-GRU
        return self.fc(h)

# ---------------- CONFIG ----------------
CKPT_PATH = "checkpoint_gru_h256_l3_best.pt"
SEQ_LEN   = 30
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DURATION  = 7  # seconds to run camera
# ---------------------------------------

# ---- Load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
labels = ckpt["label_encoder"]
model = GRUClassifier(
    input_size=ckpt["input_size"],
    hidden_size=ckpt["hidden_size"],
    num_layers=ckpt["num_layers"],
    num_classes=len(labels),
    dropout=ckpt["dropout"]
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---- Mediapipe ----
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---- Landmark extraction ----
def extract_vector(results):
    vec = []
    if results.pose_landmarks:
        vec.extend([c for lm in results.pose_landmarks.landmark for c in (lm.x, lm.y, lm.z)])
    else:
        vec.extend([0.0] * 33 * 3)
    if results.left_hand_landmarks:
        vec.extend([c for lm in results.left_hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)])
    else:
        vec.extend([0.0] * 21 * 3)
    if results.right_hand_landmarks:
        vec.extend([c for lm in results.right_hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)])
    else:
        vec.extend([0.0] * 21 * 3)
    return vec

# ---- Main logic ----
buffer = deque(maxlen=SEQ_LEN)
cap = cv2.VideoCapture(1)
print("📹 Starting camera for 7 seconds...")

predictions = []
start_time = time.time()

with torch.no_grad():
    while time.time() - start_time < DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        vec = extract_vector(results)
        buffer.append(vec)

        if len(buffer) == SEQ_LEN:
            seq = torch.tensor([buffer], dtype=torch.float32).to(DEVICE)
            lengths = torch.tensor([SEQ_LEN])
            logits = model(seq, lengths)
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()

# ---- Final prediction ----
if predictions:
    most_common = Counter(predictions).most_common(1)[0]
    label = labels[most_common[0]]
    print(f"\n✅ Most likely class: **{label}** (votes: {most_common[1]} out of {len(predictions)})")
else:
    print("\n⚠️ Not enough data captured to make a prediction.")
