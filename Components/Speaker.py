import cv2
import numpy as np
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
import os

# Update paths to the model files
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
temp_audio_path = "temp_audio.wav"

# Load DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

def voice_activity_detection(audio_frame, sample_rate=16000):
    return vad.is_speech(audio_frame, sample_rate)

def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

global Frames
Frames = [] # [x,y,w,h]

'''
def detect_faces_and_speakers(input_video_path, output_video_path):
    # Return Frams:
    global Frames
    # Extract audio from the video
    print("Extracting audio from video...")
    extract_audio_from_video(input_video_path, temp_audio_path)

    print("Read the extracted audio")
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    print("Creating video capture")
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_duration_ms = 30  # 30ms frames
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)


    while cap.isOpened():
        print("Cap opened")
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
        is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)
        MaxDif = 0
        Add = []
        for i in range(detections.shape[2]):
            print("Looping through detections 1")
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_width = x1 - x
                face_height = y1 - y

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((y + 2 * face_height // 3) - (y1))
                Add.append([[x, y, x1, y1], lip_distance])

                MaxDif = max(lip_distance, MaxDif)
        for i in range(detections.shape[2]):
            print("Looping through detections 2")
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_width = x1 - x
                face_height = y1 - y

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((y + 2 * face_height // 3) - (y1))
                print(lip_distance)

                # Combine visual and audio cues
                if lip_distance >= MaxDif and is_speaking_audio:  # Adjust the threshold as needed
                    cv2.putText(frame, "Active Speaker", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if lip_distance >= MaxDif:
                    break
                Frames.append([x, y, x1, y1])


        print("Writing frame")
        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.remove(temp_audio_path)
    print("Finished")
'''
def detect_faces_and_speakers(input_video_path, output_video_path):
    # Return Frames:
    global Frames
    # Extract audio from the video
    print("Extracting audio from video...")
    extract_audio_from_video(input_video_path, temp_audio_path)

    print("Reading the extracted audio...")
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    print("Creating video capture...")
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # If this still hangs, try 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_duration_ms = 30
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            print("Ran out of audio frames. Stopping.")
            break
        
        is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # --- START OF THE CORRECTED LOGIC ---

        detected_faces_in_frame = []
        # 1. Loop ONCE to find all faces and their metrics
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                face_height = y1 - y
                # Your lip distance metric
                lip_distance = abs((y + 2 * face_height // 3) - (y1))

                # Store face info in a dictionary for clarity
                face_info = {
                    "box": (x, y, x1, y1),
                    "lip_distance": lip_distance
                }
                detected_faces_in_frame.append(face_info)
                
                # Append to the global list for every detected face
                Frames.append([x, y, x1, y1])

        # 2. Now, determine the speaker from the faces found IN THIS FRAME
        speaker_box = None
        if is_speaking_audio and detected_faces_in_frame:
            # Find the face with the maximum lip_distance
            speaker_face = max(detected_faces_in_frame, key=lambda f: f['lip_distance'])
            speaker_box = speaker_face['box']

        # 3. Draw all boxes and label the speaker
        for face in detected_faces_in_frame:
            (x, y, x1, y1) = face['box']
            color = (0, 255, 0) # Green for non-speaker
            
            if speaker_box and face['box'] == speaker_box:
                color = (0, 0, 255) # Red for speaker
                cv2.putText(frame, "Active Speaker", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)

        # --- END OF THE CORRECTED LOGIC ---

        print("Writing frame...")
        out.write(frame)
        #cv2.imshow('Frame', frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    print("Releasing resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.remove(temp_audio_path)
    print("Finished")


if __name__ == "__main__":
    detect_faces_and_speakers()
    print(Frames)
    print(len(Frames))
    print(Frames[1:5])
