import os
from playsound import playsound
import sys
from openai import OpenAI
import api
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
    
def highestConfidence():
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

    best_prediction = None  # (label_index, confidence)

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
                probs = torch.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()

                if (best_prediction is None) or (conf > best_prediction[1]):
                    best_prediction = (pred, conf)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    # ---- Final prediction ----
    if best_prediction:
        label_idx, conf = best_prediction
        label = labels[label_idx]
        print(f"\n✅ Most confident class: **{label}** with {conf:.2f} confidence")
        return label
    else:
        print("\n⚠️ Not enough data captured to make a prediction.")
        return ""




def mostFrequent():
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
        return label
    else:
        print("\n⚠️ Not enough data captured to make a prediction.")
        return ""





def setup_client():
    api_key =  f"{api.key}"
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    return OpenAI(api_key=api_key)


class ChatBot:
    def __init__(self, model="gpt-4.1-mini"):
        self.client = setup_client()
        self.model = model
        self.system_prompt = """Você receberá uma sequência de texto formada apenas por letras, sem espaços ou pontuação. Sua tarefa é:

1. Reescrever o texto como uma frase coerente, adicionando os espaços e a pontuação adequados.
2. Corrigir eventuais erros de digitação ou ortografia que possam estar presentes.
3. Garantir que a frase final fique clara, gramaticalmente correta e soe natural.

Exemplo de entrada:
esteeumtesteparasuashabilidades

Exemplo de saída:
Este é um teste para suas habilidades."""

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def send_message(self, message):
        self.conversation_history.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                max_tokens=1000,
                temperature=0.3
            )
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("Conversation history cleared (system prompt maintained).")

    def show_prompt(self):
        print("\nCurrent system prompt:")
        print("-" * 40)
        print(self.system_prompt)
        print("-" * 40)
        
def main():
    flag = 1
    text = ""
    i = 0
    captured_text = ""
    
    bot = ChatBot()
    print("Comandos: 'sair/q', 'clear', 'prompt', 'detectar'")
    while True:
        try:
            user_input = input("\nTexto para formatar: ").strip()
            if user_input.lower() in ['sair','q']:
                print("Goodbye! / Tchau!")
                break
            elif user_input.lower() in ['clear', 'limpar']:
                bot.clear_history()
                continue
            elif user_input.lower() in ['prompt', 'mostrar']:
                bot.show_prompt()
                continue
            elif user_input.lower() in ['detectar', 'detectar gesto']:
                captured_text = highestConfidence()
                continue
            
            print("Formatted: ", end="", flush=True)
            response = bot.send_message(captured_text)
            print(response)
            os.system(f"touch output{i}.wav")
            #os.system(f"echo \"{response}\" | ./piper.exe -m ./pt_BR-faber-medium.onnx -f ./output{i}.wav")
            os.system(f"echo \"{response}\" | piper --model pt_BR-faber-medium.onnx --output_file output{i}.wav")
            playsound(f'./output{i}.wav')
            i += 1
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! / Tchau!")
            break
        except EOFError:
            print("\n\nGoodbye! / Tchau!")
            break
                    
if __name__ == "__main__":
    main()
