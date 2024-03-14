from flask import Flask, request, jsonify
import base64
import os
import cv2
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
#from IPython.display import display
import face_recognition
import csv


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to track attendance marking
attendance_marked = {}

def initialize_face_recognition():
    face_1 = face_recognition.load_image_file("train_images/harish.jpeg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]

    face_2 = face_recognition.load_image_file("train_images/shreya.jpeg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]

    face_3 = face_recognition.load_image_file("train_images/bipin.jpeg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]

    face_4 = face_recognition.load_image_file("train_images/siddu.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]

    known_face_encodings = [
        face_1_encoding,
        face_2_encoding,
        face_3_encoding,
        face_4_encoding
    ]
    known_face_names = [
        "Harish Thangaraj",
        "Shreya Chaurasia",
        "Bipin",
        "Siddhanth Sridhar"
    ]
    print("Done learning and creating profiles")
    return known_face_encodings, known_face_names







def makeAttendanceEntry(name):
    csv_file_path = 'attendance_list.csv'
    headers = ['Name', 'Date', 'Time']
    
    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a') as f:
        # Write headers only if the file doesn't exist or is empty
        if not file_exists or os.path.getsize(csv_file_path) == 0:
            f.write(','.join(headers) + '\n')

        now = datetime.now()
        date_string = now.strftime('%d/%b/%Y')
        time_string = now.strftime('%H:%M:%S')
        f.write(f'{name},{date_string},{time_string}\n')

def recognize_faces(frame, known_face_encodings, known_face_names):
    global attendance_marked

    unknown_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Mark attendance only if not already marked in this session
            #if name not in attendance_marked or not attendance_marked[name]:
            makeAttendanceEntry(name)
            attendance_marked[name] = True  # Mark attendance for this person

        # Draw a box around the face using OpenCV
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return frame
   
       
    
def process_video_and_display_one_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    known_face_encodings, known_face_names = initialize_face_recognition()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count == 10:  # Display the 10th frame, you can change this number as needed
            frame_with_recognized_faces = recognize_faces(frame, known_face_encodings, known_face_names)
            #cv2_imshow(frame_with_recognized_faces)
            break

    cap.release()
    cv2.destroyAllWindows() 


@app.route('/upload', methods=['POST'])
def upload_video():
    # Get base64 encoded video data from the request
    data = request.get_json()
    base64_video = data.get('base64_video')

    if base64_video:
        try:
            # Decode base64 to obtain the video data
            video_data = base64.b64decode(base64_video)

            # Save the video to a local directory
            filename = "uploaded_video.mp4"  # You can generate a unique filename if needed
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(video_data)

            # Process the video and display one frame
            process_video_and_display_one_frame(filepath)

            # Return success response with the file path
            return jsonify({'success': True, 'file_path': filepath})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'No video data found in the request.'})
    

import csv
from flask import jsonify

@app.route('/return_csv', methods=['GET'])
def return_csv_as_json():
    csv_file_path = 'attendance_list.csv'  # Path to your CSV file
    json_data = []

    # Read the CSV file with comma delimiter and convert each row to a dictionary
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # No need to specify delimiter for comma
        for row_number, row in enumerate(csv_reader, start=1):
            try:
                # Remove None values from the row dictionary
                row = {key: value for key, value in row.items() if value is not None}
                
                # Check if any value in the row is empty
                if not any(row.values()):
                    continue  # Skip this row if any value is empty
                
                json_data.append(row)
            except Exception as e:
                print(f"Error processing row {row_number}: {e}")
                print(f"Row data: {row}")

    # Return the list of dictionaries as JSON
    return jsonify(json_data)

@app.route('/clear_csv', methods=['POST'])
def clear_csv():
    csv_file_path = 'attendance_list.csv'  # Path to your CSV file

    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Open the file in write mode to clear its contents
        with open(csv_file_path, 'w') as f:
            f.truncate(0)  # Truncate the file to clear its contents
        return jsonify({'message': 'CSV file cleared successfully'}), 200
    else:
        return jsonify({'error': 'CSV file not found'}), 404


if __name__ == "__main__":
    app.run(debug=True)


    