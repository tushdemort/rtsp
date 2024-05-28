from flask import Flask,render_template, Response, request
import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import threading
# Load the YOLO model
model = YOLO("yolov9c.pt")

def __init__(id,password,link):
    global rtsp_url
    #rtsp_url = 'rtsp://admin:tech%232024@3.109.22.72:554/dac/realplay/EEA5F6FB-DD49-4227-B720-8A3C475A3AC31/MAIN/TCP'
    rtsp_url = f'rtsp://{id}:{password}@{link}'
# FFmpeg command to capture the RTSP stream
    ffmpeg_command = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-i', rtsp_url,
    '-r', '15',
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo', '-'
]
    global process
# Run FFmpeg command
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

# Function to read a frame from the FFmpeg process
def read_frame(process, width, height):
    frame_size = width * height * 3
    raw_frame = process.stdout.read(frame_size)
    
    if len(raw_frame) != frame_size:
        return None

    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    return frame

app = Flask(__name__)

def stream():
    while True:
        frame = read_frame(process, 1920, 1080)
        if frame is None:
            break

    # Get YOLO predictions
        predictions = model.predict(frame, conf=0.5, stream=False)
    
    # Draw the predictions on the frame
        frame_with_detections = frame.copy()  # Make a copy of the frame
        for prediction in predictions:
            for box in prediction.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_name = model.model.names[int(box.cls[0])]

            # Draw bounding box
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label and confidence score
                cv2.putText(frame_with_detections, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes
        ret, buffer = cv2.imencode('.jpg', frame_with_detections)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        global id
        id = request.form['id']
        global password
        password = request.form['password']
        global link
        link = request.form['link']
        __init__(id,password,link)
        # Do something with the retrieved data (e.g., print, store in database)
        return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('home.html')


   

if __name__ == '__main__':
    thread = threading.Thread(target=stream)
    thread.daemon = True
    thread.start()
    app.run(host='0.0.0.0', debug=True)  # Change host to '0.0.0.0' for external access