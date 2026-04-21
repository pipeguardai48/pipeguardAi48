from flask import Flask, render_template, request, jsonify, redirect, send_from_directory, url_for, session, Response
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
import sys
import threading
from queue import Queue, Empty
import json
import pandas as pd # Added for parsing results.csv
import time # For timestamps in logs
import re # For parsing lines
import glob # For finding latest results.csv
# Import custom modules
from shuffle import shuffle_dataset
from model import train_fault_detector, validate_and_predict # Note: If using custom, integrate callback below
# Import YOLO for callbacks
from ultralytics import YOLO
# NEW: Import prediction module
from predict import run_prediction
import base64
from io import BytesIO
import cv2
# ================== APP CONFIG ==================
app = Flask(__name__)
app.secret_key = "supersecretkey123" # 🔒 Required for session handling
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # 500MB max
# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# ================== DATASET FOLDERS ==================
base_dataset = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')
dataset_yaml_path = os.path.join(base_dataset, 'data.yaml') # Assume existing YAML
for dir_path in [
    os.path.join(base_dataset, 'train/images'), os.path.join(base_dataset, 'train/labels'),
    os.path.join(base_dataset, 'test/images'), os.path.join(base_dataset, 'test/labels'),
    os.path.join(base_dataset, 'valid/images'), os.path.join(base_dataset, 'valid/labels')
]:
    os.makedirs(dir_path, exist_ok=True)
# NEW: Temp predict folder
predict_temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'predict')
os.makedirs(predict_temp_dir, exist_ok=True)



# Folder for result videos (served directly by Flask)
RESULTS_DIR = os.path.join('static', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================== GLOBAL TRAINING STATE ==================
current_queue = None
current_thread = None
def log_to_queue(q, message, level='INFO'):
    """Helper to log with timestamp and level for VS Code-like output."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted = f"[{timestamp}] [{level}] {message}"
    q.put(('log', formatted))
    print(formatted) # Also print to console for debugging
def run_yolo_training(q, yaml_path, output_dir, epochs_per_iter, num_iterations, batch_size):
    class QueueingStdout:
        def __init__(self, q, iteration, epochs_per_iter, total_epochs):
            self.q = q
            self.current_epoch = 0
            self.iteration = iteration
            self.epochs_per_iter = epochs_per_iter
            self.total_epochs = total_epochs
        def write(self, text):
            for line in text.splitlines():
                line = line.rstrip()
                if line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_log = f"[{timestamp}] [YOLO] {line}"
                    self.q.put(('log', formatted_log))
                    # Parse for epoch training line - FIXED FOR REAL YOLO OUTPUT
                    train_match = re.search(r'Epoch\s+(\d+)/(\d+).*?box\s*=\s*([\d.]+).*?cls\s*=\s*([\d.]+).*?dfl\s*=\s*([\d.]+)', line)
                    if train_match:
                        epoch, total, box, cls, dfl = train_match.groups()
                        epoch = int(epoch)
                        self.current_epoch = epoch
                        global_epoch = (self.iteration - 1) * self.epochs_per_iter + epoch
                        data = {
                            'type': 'train',
                            'epoch': global_epoch,
                            'box_loss': float(box),
                            'cls_loss': float(cls),
                            'dfl_loss': float(dfl)
                        }
                        self.q.put(('train_update', json.dumps(data)))
                        # Additional summary log
                        summary = f"Train Global Epoch {global_epoch}/{self.total_epochs}: box={box:.4f}, cls={cls:.4f}, dfl={dfl:.4f}"
                        log_to_queue(self.q, summary, 'INFO')
                    # Parse for validation line - FIXED FOR REAL YOLO OUTPUT
                    val_match = re.search(r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                    if val_match:
                        p, r, map50, map5095 = val_match.groups()
                        global_epoch = (self.iteration - 1) * self.epochs_per_iter + self.current_epoch
                        progress = round((global_epoch / self.total_epochs) * 100, 1)
                        data = {
                            'type': 'val',
                            'epoch': global_epoch,
                            'map50': float(map50),
                            'progress': progress
                        }
                        self.q.put(('val_update', json.dumps(data)))
                        summary = f"Val Global Epoch {global_epoch}/{self.total_epochs}: mAP@0.5={map50:.4f} ({progress:.1f}%)"
                        log_to_queue(self.q, summary, 'SUCCESS')
        def flush(self):
            pass
    all_final_metrics = []
    model = YOLO('yolov8n.pt') # Or your custom model path
    total_start = datetime.now()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    total_epochs = num_iterations * epochs_per_iter
    try:
        for iteration in range(1, num_iterations + 1):
            log_to_queue(q, f"Starting YOLOv8 training Iteration {iteration}/{num_iterations} with {epochs_per_iter} epochs and batch size {batch_size}", 'INFO')
            iter_start = datetime.now()
            iter_output_name = f'fault_detector_iter{iteration}'
            # Create stdout for this iteration
            q_stdout = QueueingStdout(q, iteration, epochs_per_iter, total_epochs)
            sys.stdout = q_stdout
            sys.stderr = q_stdout
            try:
                results = model.train(
                    data=yaml_path,
                    epochs=epochs_per_iter,
                    batch=batch_size,
                    project=output_dir,
                    name=iter_output_name,
                    verbose=True # Ensure full logs are printed
                )
            except Exception as e:
                log_to_queue(q, f'Iteration {iteration} ERROR: {str(e)}', 'ERROR')
                q.put(('error', json.dumps({'error': str(e), 'iteration': iteration})))
                import traceback
                traceback.print_exc()
                continue
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            iter_end = datetime.now()
            iter_duration = (iter_end - iter_start).total_seconds() / 60
            # ==================== PARSE FULL CSV FOR THIS ITER ====================
            iter_metrics = {
                'iteration': iteration,
                'best_map': 0.0,
                'best_epoch_local': epochs_per_iter,
                'training_time': f"{iter_duration:.0f} min",
                # NEW: Full arrays for graph
                'epochs': [],
                'map50_array': [],
                'box_loss': [],
                'cls_loss': [],
                'dfl_loss': []
            }
            # Find the results.csv for this iteration
            csv_pattern = os.path.join(output_dir, iter_output_name, 'results.csv')
            csv_files = glob.glob(csv_pattern)
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_csv)
                    if not df.empty and 'metrics/mAP50(B)' in df.columns:
                        # Extract full arrays
                        iter_metrics['epochs'] = df['epoch'].astype(int).tolist()
                        iter_metrics['map50_array'] = (df['metrics/mAP50(B)'] * 100).round(2).tolist()
                        if 'train/box_loss' in df.columns:
                            iter_metrics['box_loss'] = df['train/box_loss'].round(4).tolist()
                        if 'train/cls_loss' in df.columns:
                            iter_metrics['cls_loss'] = df['train/cls_loss'].round(4).tolist()
                        if 'train/dfl_loss' in df.columns:
                            iter_metrics['dfl_loss'] = df['train/dfl_loss'].round(4).tolist()
                          
                        # Best metrics
                        best_idx = df['metrics/mAP50(B)'].idxmax()
                        iter_metrics['best_map'] = df.iloc[best_idx]['metrics/mAP50(B)'] * 100
                        iter_metrics['best_epoch_local'] = int(df.iloc[best_idx]['epoch'])
                      
                        # Avg loss at best map epoch
                        if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
                            box = df.iloc[best_idx]['train/box_loss']
                            cls = df.iloc[best_idx]['train/cls_loss']
                            dfl = df.iloc[best_idx]['train/dfl_loss']
                            iter_metrics['avg_loss'] = (box + cls + dfl) / 3.0
                        else:
                            iter_metrics['avg_loss'] = 0.0
                  
                        log_to_queue(q, f"Iter {iteration} CSV LOADED: {len(iter_metrics['epochs'])} epochs for graph", 'SUCCESS')
                    else:
                        log_to_queue(q, f'Warning: Empty CSV or missing mAP column for iter {iteration}', 'WARNING')
                except Exception as e:
                    log_to_queue(q, f'Warning: Error parsing results.csv for iter {iteration}: {e}', 'WARNING')
            else:
                log_to_queue(q, f'Warning: No results.csv found for iter {iteration}', 'WARNING')
            all_final_metrics.append(iter_metrics)
            # Send per iteration metrics
            q.put(('iter_done', json.dumps(iter_metrics)))
            log_to_queue(q, f"Iteration {iteration} completed. Best model saved. Best mAP: {iter_metrics['best_map']:.2f}%", 'SUCCESS')
        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds() / 60
        # Overall best
        if all_final_metrics:
            overall_best_map = max([m['best_map'] for m in all_final_metrics])
            overall_best_iter = next((m['iteration'] for m in all_final_metrics if m['best_map'] == overall_best_map), 1)
        else:
            overall_best_map = 0.0
            overall_best_iter = 1
        final_dict = {
            'all_iterations': all_final_metrics,
            'overall_best_map': overall_best_map,
            'overall_best_iter': overall_best_iter,
            'total_training_time': f"{total_duration:.0f} min"
        }
        q.put(('done', json.dumps(final_dict)))
        log_to_queue(q, 'All iterations completed. Overall best model saved.', 'SUCCESS')
      
    except Exception as e:
        error_msg = f'ERROR: {str(e)}'
        log_to_queue(q, error_msg, 'ERROR')
        q.put(('error', json.dumps({"error": str(e)})))
        import traceback
        traceback.print_exc()
    finally:
        # Restore streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
# ================== HELPER: LOGIN REQUIRED DECORATOR ==================
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
# ================== ROUTES ==================
@app.route('/')
def home():
    return redirect(url_for('login'))
# ---------- LOGIN ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple hardcoded login logic
        if username == 'admin' and password == 'admin':
            session['username'] = username
            session['role'] = 'admin'
            return jsonify({'success': True, 'redirect': '/admin'})
        elif username == 'employee' and password == 'admin':
            session['username'] = username
            session['role'] = 'employee'
            return jsonify({'success': True, 'redirect': '/index'})
        else:
            return jsonify({'error': 'Invalid username or password'})
    return render_template('login.html')
# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
#-----------videoBackground---------
@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory('static/video', filename)
# ---------- EMPLOYEE PAGE ----------
@app.route('/index')
@login_required(role='employee')
def employee_index():
    return render_template('index.html', username=session.get('username'))
# ---------- ADMIN PAGE ----------
@app.route('/admin')
@login_required(role='admin')
def admin():
    return render_template('admin.html', username=session.get('username'))
# ---------- PROCESS PAGE ----------
# ---------- PROCESS PAGE (Now supports GET for render + POST for predict) ----------
@app.route('/process', methods=['GET', 'POST'])
@login_required(role='employee')
def process():
    if request.method == 'GET':
        return render_template('processing.html', username=session.get('username'))
    
    # POST: Handle prediction
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, etc.'}), 400

        # Save temp file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"temp_{timestamp}_{file.filename.rsplit('.', 1)[0]}.{file.filename.rsplit('.', 1)[1]}")
        temp_path = os.path.join(predict_temp_dir, filename)
        file.save(temp_path)

        # Run prediction
        result = run_prediction(temp_path, app.config['UPLOAD_FOLDER'])

        # Cleanup
        os.remove(temp_path)

        return jsonify(result)

    except FileNotFoundError as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
#-------------------------------------Video-------------------------------------










@app.route('/video')
@login_required(role='employee')
def video_page():
    return render_template('video.html')




@app.route('/video_predict', methods=['POST'])
@login_required(role='employee')
def video_predict():
    input_path = output_path = None
    try:
        print("\nVIDEO ANALYSIS STARTED")

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.lower().endswith('.mp4'):
            return jsonify({'error': 'Only .mp4 allowed'}), 400

        # Save uploaded video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        input_path = os.path.join(predict_temp_dir, f"input_{timestamp}.mp4")
        output_filename = f"result_{timestamp}.mp4"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        file.save(input_path)
        print(f"Uploaded video → {input_path}")

        # Load latest trained model
        from predict import find_latest_model
        weights_path = find_latest_model(app.config['UPLOAD_FOLDER'])
        if not weights_path:
            os.remove(input_path)
            return jsonify({'error': 'No trained model found! Please train a model first in Admin → Train.'}), 404

        print(f"Using model → {weights_path}")
        model = YOLO(weights_path)

        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            os.remove(input_path)
            return jsonify({'error': 'Cannot read video file'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {w}x{h} @ {fps:.1f}fps")

        # mp4v works on Windows, macOS, Linux
        fourcc = cv2.VideoWriter_fourcc(*'avc1')   # use H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        detections = []
        frame_idx = 0
        max_frames = 600  # ~20 sec limit

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames:
                break
            frame_idx += 1

            if frame_idx % 3 == 0:  # Run YOLO every 3rd frame
                results = model(frame, conf=0.4, verbose=False)[0]
                annotated = results.plot()

                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0].item())
                        name = results.names.get(cls_id, f"class_{cls_id}")
                        conf = round(float(box.conf[0].item()) * 100, 1)
                        detections.append({
                            'class': name,
                            'confidence': conf,
                            'frame': frame_idx
                        })
            else:
                annotated = frame

            out.write(annotated)

        cap.release()
        out.release()
        os.remove(input_path)
        print(f"Result video saved → {output_path}")

        # Return direct URL (NO BASE64!)
        result_url = url_for('static', filename=f'results/{output_filename}') + f'?t={int(time.time())}'


        return jsonify({
            'detected': result_url,
            'detections': detections[:100]
        })

    except Exception as e:
        import traceback
        print("VIDEO PREDICTION FAILED:")
        traceback.print_exc()
        # Cleanup
        for p in [input_path, output_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
        return jsonify({'error': f'Server error: {str(e)}'}), 500























# ---------- IMAGES PAGE (Updated with login) ----------
@app.route('/images')
@login_required(role='employee')  # Added protection
def images():
    return render_template('images.html', username=session.get('username'))

# ---------- NEW: PREDICT ENDPOINT ----------
@app.route('/predict', methods=['POST'])
@login_required(role='employee')
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, etc.'}), 400

        # Save temp file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"temp_{timestamp}_{file.filename.rsplit('.', 1)[0]}.{file.filename.rsplit('.', 1)[1]}")
        temp_path = os.path.join(predict_temp_dir, filename)
        file.save(temp_path)

        # Run prediction
        result = run_prediction(temp_path, app.config['UPLOAD_FOLDER'])

        # Cleanup
        os.remove(temp_path)

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


#---------------------camera ---------------------


@app.route('/camera')
@login_required(role='employee')
def camera():
    return render_template('camera.html', username=session.get('username'))



    
        

# ---------- TRAINING ENDPOINT ----------
@app.route('/train', methods=['GET', 'POST'])
@login_required(role='admin')
def train_endpoint():
    if request.method == 'GET':
        return jsonify({'message': 'POST to upload and train. Use admin dashboard.'}), 200
    global current_queue, current_thread
    if current_thread and current_thread.is_alive():
        return jsonify({'error': 'Training already in progress'}), 400
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        files = request.files.getlist('files')
        class_name = request.form.get('class_name', '').strip().lower()
        epochs_per_iter = int(request.form.get('epochs_per_iter', 4))
        num_iterations = int(request.form.get('num_iterations', 6))
        batch_size = int(request.form.get('batch_size', 16))
        if not files:
            return jsonify({'error': 'No files received'}), 400
        # Start queue early to capture upload logs
        current_queue = Queue()
        # Log upload start
        log_to_queue(current_queue, f"Received {len(files)} files for class '{class_name}', {num_iterations} iterations x {epochs_per_iter} epochs, batch={batch_size}", 'INFO')
        train_images_dir = os.path.join(base_dataset, 'train/images')
        train_labels_dir = os.path.join(base_dataset, 'train/labels')
        uploaded_count = 0
        class_map = {}
        # Detect folder upload (webkitdirectory)
        is_folder_upload = any("/" in f.filename or "\\" in f.filename for f in files)
        log_to_queue(current_queue, f"Folder upload detected: {is_folder_upload}", 'INFO')
        for file in files:
            if not file or not allowed_file(file.filename):
                continue
            # Extract just the filename
            filename = secure_filename(os.path.basename(file.filename))
            image_path = os.path.join(train_images_dir, filename)
            file.save(image_path)
            uploaded_count += 1
            log_to_queue(current_queue, f"Uploaded image: {filename}", 'INFO')
            # Handle normal (manual class) upload
            if not is_folder_upload and class_name:
                label_filename = filename.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(train_labels_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 1.0 1.0\n")
                log_to_queue(current_queue, f"Generated label for {filename} (class: {class_name})", 'INFO')
            # Handle folder upload (auto infer class)
            elif is_folder_upload:
                subfolder = os.path.dirname(file.filename).replace("\\", "/")
                inferred_class = subfolder.split("/")[-1] if "/" in subfolder else subfolder
                inferred_class = inferred_class.strip().lower()
                # Assign class ID dynamically
                if inferred_class not in class_map:
                    class_map[inferred_class] = len(class_map)
                class_id = class_map[inferred_class]
                log_to_queue(current_queue, f"Inferred class '{inferred_class}' → ID {class_id} for {filename}", 'INFO')
                # Write YOLO label file
                label_filename = filename.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(train_labels_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        # Handle single class upload
        if not is_folder_upload and class_name:
            class_map[class_name] = 0
        if uploaded_count == 0:
            return jsonify({'error': 'No valid images uploaded'}), 400
        log_to_queue(current_queue, f"Uploaded {uploaded_count} images successfully. Class map: {class_map}", 'SUCCESS')
        if not class_map:
            return jsonify({'error': 'No classes defined'}), 400
        # Shuffle dataset
        train_count, test_count = shuffle_dataset(base_dataset)
        log_to_queue(current_queue, f"Shuffled dataset: {train_count} train, {test_count} test", 'INFO')
        # Generate data.yaml dynamically
        nc = len(class_map)
        names = {str(class_map[name]): name for name in class_map}
        yaml_content = f"""path: {os.path.abspath(base_dataset)} # dataset root dir
train: train/images
val: valid/images
test: test/images
nc: {nc}
names: {names}
"""
        with open(dataset_yaml_path, 'w') as f:
            f.write(yaml_content)
        log_to_queue(current_queue, f"Generated data.yaml with {nc} classes", 'INFO')
        # Start training thread
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'runs/detect')
        os.makedirs(output_dir, exist_ok=True)
        current_thread = threading.Thread(target=run_yolo_training, args=(current_queue, dataset_yaml_path, output_dir, epochs_per_iter, num_iterations, batch_size))
        current_thread.start()
        return jsonify({
            'success': True,
            'started': True,
            'uploaded': uploaded_count,
            'train_count': train_count,
            'test_count': test_count,
            'iterations': num_iterations,
            'epochs_per_iter': epochs_per_iter
        })
    except Exception as e:
        log_to_queue(current_queue, f'Train error: {str(e)}', 'ERROR')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
# ---------- STREAM ENDPOINT ----------
@app.route('/stream')
def stream():
    def event_stream():
        global current_queue, current_thread
        if current_thread is None or not current_thread.is_alive():
            yield 'data: No active training session.\n\n'
            return
        while True:
            try:
                msg = current_queue.get(timeout=1)
                if isinstance(msg, tuple):
                    if msg[0] == 'log':
                        yield f'data: {msg[1]}\n\n'
                    elif msg[0] == 'train_update':
                        yield f'event: train_update\ndata: {msg[1]}\n\n'
                    elif msg[0] == 'val_update':
                        yield f'event: val_update\ndata: {msg[1]}\n\n'
                    elif msg[0] == 'iter_done':
                        yield f'event: iter_done\ndata: {msg[1]}\n\n'
                    elif msg[0] == 'done':
                        yield f'event: done\ndata: {msg[1]}\n\n'
                        current_queue = None
                        current_thread = None
                        break
                    elif msg[0] == 'error':
                        yield f'event: error\ndata: {msg[1]}\n\n'
                        current_queue = None
                        current_thread = None
                        break
            except Empty:
                if current_thread and not current_thread.is_alive():
                    current_queue = None
                    current_thread = None
                    break
    return Response(event_stream(), mimetype='text/event-stream')
# ================== RUN APP ==================
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)