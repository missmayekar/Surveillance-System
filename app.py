from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import os
from PIL import Image
import threading

app = Flask(__name__)

class MLPs(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, units=[4096, 4096], layernorm=False, dropout=None, last_activation=nn.Identity()):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.layernorm = layernorm

        def block(in_, out_):
            layers = [
                nn.Linear(in_, out_),
                nn.LayerNorm(out_) if self.layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout else nn.Identity()
            ]
            return nn.Sequential(*layers)

        for out_dim in units:
            layers.extend([block(in_dim, out_dim)])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(last_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ScoreOrLogDensityNetwork(nn.Module):
    def __init__(self, net, score_network=False):
        super().__init__()
        self.network = net
        self.is_score_network = score_network

    def forward(self, x):
        return self.network(x)

    def score(self, x, return_log_density=False):
        score, log_density = None, None
        if self.is_score_network:
            score = self.network(x)
            if return_log_density:
                log_density = torch.zeros_like(score[:, 0][:, None])
        else:
            x = x.requires_grad_()
            log_density = self.network(x)
            logp = -log_density.sum()
            score = torch.autograd.grad(logp, x, create_graph=True)[0]

        if return_log_density:
            return score, log_density
        else:
            return score

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_dim = 64*64*3 
output_dim = 1
hidden_units = [4096, 4096]

mlps = MLPs(input_dim=input_dim, output_dim=output_dim, units=hidden_units)
model = ScoreOrLogDensityNetwork(net=mlps).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  

model = torch.load('model/model2.pth')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

model_yolo = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=150, n_init=4, nms_max_overlap=0.7)

class_names = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting',
    'Shoplifting', 'Stealing', 'Vandalism'
]

stop_event = threading.Event()
live_feed_running = True
video_capture = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        input_path = 'uploaded_video.mp4'
        output_path = 'output_video.mp4'
        file.save(input_path)
        return redirect(url_for('process_video', input_path=input_path, output_path=output_path))
    return redirect(url_for('index'))

@app.route('/process_video')
def process_video():
    input_path = request.args.get('input_path')
    output_path = request.args.get('output_path')

    def process_video_file(input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        track_colors_dict = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            results = model_yolo(frame)
            bboxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                if cls == 0 and conf >= 0.5: 
                    width = x2 - x1
                    height = y2 - y1
                    bboxes.append(((x1.item(), y1.item(), width.item(), height.item()), conf.item(), cls))
            tracks = tracker.update_tracks(bboxes, frame=frame)

            if frame_count % 4 == 0:
                pil_frame = transform(pil_frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(pil_frame.view(pil_frame.size(0), -1))
                    _, predicted = torch.max(output, 1)
                    predicted_class = class_names[predicted.item()]
                cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2, cv2.LINE_AA)


            for track in tracks:
                if not track.is_confirmed():
                    continue
                box = track.to_tlbr()
                track_id = track.track_id
                

                if track_id not in track_colors_dict:
                    track_colors_dict[track_id] = get_fixed_color(track_id)
                color = track_colors_dict[track_id]
                

                if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                    color = (255, 0, 0) 
                
                if len(box) == 4:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    label = f'ID: {track_id}'
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    print(f"Invalid bounding box: {box}")

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    process_video_file(input_path, output_path)

    return redirect(url_for('download_video'))

@app.route('/start_video')
def start_video():
    return render_template('index.html')

def get_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def get_fixed_color(track_id):
    try:
        track_id = int(track_id)
    except ValueError:
        print(f"Invalid track_id: {track_id}. Defaulting to random color.")
        return tuple(np.random.randint(0, 255, 3).tolist())

    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global live_feed_running
        cap = cv2.VideoCapture(0)
        frame_count = 0
        track_colors_dict = {}

        while live_feed_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            results = model_yolo(frame)
            bboxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                if cls == 0 and conf >= 0.5:
                    width = x2 - x1
                    height = y2 - y1
                    bboxes.append(((x1.item(), y1.item(), width.item(), height.item()), conf.item(), cls))
            tracks = tracker.update_tracks(bboxes, frame=frame)

            if frame_count % 4 == 0:
                pil_frame = transform(pil_frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(pil_frame.view(pil_frame.size(0), -1))
                    _, predicted = torch.max(output, 1)
                    predicted_class = class_names[predicted.item()]

                cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            for track in tracks:
                if not track.is_confirmed():
                    continue
                box = track.to_tlbr()
                track_id = track.track_id

                if track_id not in track_colors_dict:
                    track_colors_dict[track_id] = get_fixed_color(track_id)
                color = track_colors_dict[track_id]

                if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                    color = (255, 0, 0) 
                
                if len(box) == 4:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    label = f'ID: {track_id}'
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    print(f"Invalid bounding box: {box}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            frame_count += 1

        cap.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_video')
def stop_video():
    global live_feed_running
    live_feed_running = False
    return redirect(url_for('start_video'))

@app.route('/download_video')
def download_video():
    return send_from_directory(directory='.', path='output_video.mp4', as_attachment=True)

def get_fixed_color(track_id):
    track_id = int(track_id) 
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 256, 3))

if __name__ == '__main__':
    app.run(port=5001, debug=True)