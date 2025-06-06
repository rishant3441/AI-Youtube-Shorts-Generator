import cv2
import numpy as np
from moviepy.editor import *
from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical(input_video_path, output_video_path):
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    print("Done detecting faces and speakers")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Creating video capture 2")
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(vertical_height, vertical_width)


    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"start and end - {x_start} , {x_end}")
    print(x_end-x_start)
    half_width = vertical_width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps
    print(fps)
    count = 0
    for _ in range(total_frames):
        print("Going through frames")
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # New logic starts here
        speaker_X, speaker_Y, speaker_W, speaker_H = None, None, None, None
        if count < len(Frames) and Frames[count] is not None:
            speaker_coords = Frames[count]  # [x_min, y_min, x_max, y_max]
            speaker_X = speaker_coords[0]
            speaker_Y = speaker_coords[1]
            speaker_W = speaker_coords[2] - speaker_coords[0]
            speaker_H = speaker_coords[3] - speaker_coords[1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_face_x, current_face_y, current_face_w, current_face_h = None, None, None, None

        if len(faces) > 0:
            if speaker_X is not None:
                for f_coords in faces:
                    fx, fy, fw, fh = f_coords
                    face_center_x = fx + fw // 2
                    if speaker_X <= face_center_x <= (speaker_X + speaker_W):
                        current_face_x, current_face_y, current_face_w, current_face_h = fx, fy, fw, fh
                        break
            
            if current_face_x is None: # Speaker not found among faces or no speaker data
                current_face_x, current_face_y, current_face_w, current_face_h = faces[0]

        elif speaker_X is not None: # No faces by Haar, but Frames has speaker data
            current_face_x, current_face_y, current_face_w, current_face_h = speaker_X, speaker_Y, speaker_W, speaker_H

        if current_face_x is not None:
            x, y, w, h = current_face_x, current_face_y, current_face_w, current_face_h
            centerX = x + (w // 2)

            # Preserve original x_start adjustment logic, but only if a face is found
            if count == 0 or abs(x_start - (centerX - half_width)) < 1 :
                pass
            else:
                new_x_start = centerX - half_width
                new_x_end = centerX + half_width

                # Boundary checks for new_x_start and new_x_end against original_width
                if new_x_start >= 0 and new_x_end <= original_width:
                    x_start = new_x_start
                    x_end = new_x_end
                elif new_x_start < 0:
                    x_start = 0
                    x_end = vertical_width
                elif new_x_end > original_width:
                    x_start = original_width - vertical_width
                    x_end = original_width
        # If current_face_x is None, x_start and x_end remain unchanged from the previous frame.

        # End of new logic

        count += 1
        cropped_frame = frame[:, x_start:x_end]
        if cropped_frame.shape[1] == 0:
            x_start = (original_width - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]
        
        print(cropped_frame.shape)

        out.write(cropped_frame)

    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path, count)



def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        # Load video clips
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio

        combined_clip = clip_without_audio.set_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=Fps, preset='medium', bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")
    
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)



