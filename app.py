from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import mysql.connector
from datetime import datetime
import base64
import os
import io
import json
from PIL import Image
import shutil
from flask_cors import CORS
import gzip


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = None



# Making MySQL connection
db_connection = mysql.connector.connect(
    user='harish',
    password='harish',
    host='49.206.252.212',
    port='63306',
    database='LMS'
)

# Initializing a cursor
cursor = db_connection.cursor(buffered=True)

global marked_data_response
global marked_data_image
global last_unrecognized_index
global ug_id
marked_data_response=[]
marked_data_image=[]
last_unrecognized_index=-1
ug_id=419

def increment_ugid():
    global ug_id
    try:
        # Increment ug_id
        ug_id += 1
        print(f"ug_id incremented to {ug_id}")

    except TypeError as e:
        print(f"TypeError: {e}. Ensure ug_id is an integer.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def reset_ugid():
    global ug_id
    ug_id=419


def initialize_face_recognition():
    marked_data_image.clear()
    marked_data_response.clear()
    cursor.execute("SELECT ID, IMAGE FROM ATTENDENCE_MASTER WHERE STATUS_CODE='3';")
    rows = cursor.fetchall()
    if not rows:
        print("No faces with status code '3' found.")
        return []

    ids_to_update = []
    for (id, image_blob) in rows:
        try:
            image_data = base64.b64decode(image_blob)
            image = Image.open(io.BytesIO(image_data))
            face_encoding = face_recognition.face_encodings(np.array(image))[0]
            face_embeddings_json = json.dumps(face_encoding.tolist())
            update_query = """
            UPDATE ATTENDENCE_MASTER
            SET face_embeddings = %s, STATUS_CODE = '12'
            WHERE ID = %s;
            """
            cursor.execute(update_query, (face_embeddings_json, id))
            db_connection.commit()
            ids_to_update.append(id)
        except Exception as e:
            print(f"Error processing image for ID {id}: {e}")

    return ids_to_update

def mark_attendance(id, status, frame):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    _, buffer = cv2.imencode('.jpg', frame)

    #Base64 Image data
    image_blob = base64.b64encode(buffer).decode('utf-8')  
    
    #Convert the base64 string to bytes (gzip can compress only bytes)
    s_in_bytes = image_blob.encode('utf-8')

    #Compressing it
    s_compressed = gzip.compress(s_in_bytes, mtime=0)

    #Covnverting compressed bytes to base64 (I can't return bytes direcly in json objects)
    compressed_base64_encoded = base64.b64encode(s_compressed).decode('utf-8')



    if status == "Unknown":
        sql = "INSERT INTO Attendence_register (Date, Time, Status, Image, ID) VALUES (%s, %s, %s, %s, %s)"
        val = (date_string, time_string, status, image_blob, id)
        inserted_data = {
            "Date": date_string,
            "Time": time_string,
            "Status": status,
            "ID": str(id),
            "Image": image_blob,
            "Name": '-',
        }
        
    else:
        cursor.execute("SELECT NAME FROM ATTENDENCE_MASTER WHERE ID = %s", (id,))
        name_result = cursor.fetchone()
        name = name_result[0] if name_result else "Unknown"
        sql = "INSERT INTO Attendence_register (Date, Time, ID, NAME, Status, Image) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (date_string, time_string, id, name, status, image_blob)

        inserted_data = {
            "Date": date_string,
            "Time": time_string,
            "ID": id,
            "Name": name,
            "Status": status,
            "Image": image_blob,
        }

        

    cursor.execute(sql, val)
    db_connection.commit()
    #print(f"Attendance marked for {('ID ' + str(id)) if id != 420 else 'Unknown face'} on {date_string}")


    return inserted_data


def save_unrecognized_face(frame, face_encoding, face_location):
    unrecognized_folder = "Unrecognized"
    if not os.path.exists(unrecognized_folder):
        os.makedirs(unrecognized_folder)

    for filename in os.listdir(unrecognized_folder):
        if filename.endswith(".jpg"):
            stored_image_path = os.path.join(unrecognized_folder, filename)
            stored_image = face_recognition.load_image_file(stored_image_path)
            stored_face_encodings = face_recognition.face_encodings(stored_image)

            if stored_face_encodings:  # Check if any face encodings are found
                stored_face_encoding = stored_face_encodings[0]

                if face_recognition.compare_faces([stored_face_encoding], face_encoding, tolerance=0.7)[0]:
                    print("Similar face already exists in Unrecognized folder.")
                    return

    # Save new unrecognized face
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H-%M-%S')
    file_path = os.path.join(unrecognized_folder, f"unrecognized_{date_string}.jpg")
    cv2.imwrite(file_path, face_image)
    
    increment_ugid()
    # Mark attendance
    marked_unknown=mark_attendance(ug_id, "Unknown", face_image)
    marked_data_image.append({
                        "Image": marked_unknown['Image'],
                        "ID": marked_unknown['ID']
                    })
    marked_data_response.append(marked_unknown)
    

    

def recognize_faces(frame, marked_faces):
    unknown_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(unknown_image, model="hog")
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    cursor.execute("SELECT ID, face_embeddings FROM ATTENDENCE_MASTER WHERE face_embeddings IS NOT NULL")
    known_faces = cursor.fetchall()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        status = 'Unrecognized'
        id = "Unknown"
        face_image = frame[top:bottom, left:right]

        if known_faces:
            known_face_encodings = [np.array(json.loads(face_embedding)) for _, face_embedding in known_faces]
            known_face_ids = [face_id for face_id, _ in known_faces]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            
            if True in matches:
                match_index = matches.index(True)
                id = known_face_ids[match_index]
                status = 'Recognized'
                if id not in marked_faces:
                    marked_faces.add(id)
                    marked_faces_image = mark_attendance(id, status, face_image)
                    marked_data_image.append(marked_faces_image)
                    marked_data_response.append({
                        "Date": marked_faces_image['Date'],
                        "Time": marked_faces_image['Time'],
                        "ID": id,
                        "Name": marked_faces_image['Name'],
                        "Status": status,
                        "Image": marked_faces_image['Image']
                        
                    })
            else:
                save_unrecognized_face(frame, face_encoding, (top, right, bottom, left))
        else:
            save_unrecognized_face(frame, face_encoding, (top, right, bottom, left))    
    return frame

def process_video(video_path):
    initialize_face_recognition()
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    marked_faces = set()  # Keep track of marked faces for the current video

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            recognize_faces(frame, marked_faces)

    video_capture.release()
    #delete_unrecognized_folder() 

def delete_unrecognized_folder():
    unrecognized_folder = "Unrecognized"
    if os.path.exists(unrecognized_folder):
        shutil.rmtree(unrecognized_folder)
        print(f"{unrecognized_folder} directory deleted.")

def process_photo(photo_path):
    initialize_face_recognition()
    frame = cv2.imread(photo_path)
    recognize_faces(frame, set())
    #delete_unrecognized_folder() 

def base64_to_video(base64_data, output_path):
    video_data = base64.b64decode(base64_data)
    with open(output_path, "wb") as output_file:
        output_file.write(video_data)

def base64_to_photo(base64_data, output_path):
    photo_data = base64.b64decode(base64_data)
    with open(output_path, "wb") as output_file:
        output_file.write(photo_data)


global video_file_path , photo_file_path
video_file_path="vid.mp4"
photo_file_path="img.jpg"

@app.route('/upload-video', methods=['POST'])
def upload_video():
    reset_ugid()
    delete_unrecognized_folder()
    # Get base64 encoded video data from the request
    data = request.get_json()
    base64_video = data.get('base64_video')
    
    if base64_video:
        try:
            base64_to_video(base64_video, video_file_path)
            process_video(video_file_path)

            # Print successful response
            print('Successful response sent.')
            #print(marked_data_response)

            
            # Return success response with the file path
            return jsonify({'success': True, 'marked_data': marked_data_response}),200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}),500

    return jsonify({'success': False, 'error': 'No video data found in the request.'})
    

@app.route('/upload-photo', methods=['POST'])
def upload_photo():

    reset_ugid()
    delete_unrecognized_folder()
    # Get base64 encoded photo data from the request
    data = request.get_json()
    base64_photo = data.get('base64_photo')
    
    if base64_photo:
        try:
            base64_to_photo(base64_photo, photo_file_path)
            process_photo(photo_file_path)
            
            # Return success response with the file path
            return jsonify({'success': True, 'marked_data': marked_data_response})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'No video data found in the request.'})

@app.route('/get-image', methods=['POST'])
def get_image():

    try:
        data = request.get_json()
        id = data.get('ID')

        profile_list=marked_data_image
        

        for item in profile_list:
            if item['ID'] == id:
                image_profile = item['Image']
                
                image_bytes=image_profile.encode('utf-8') # To bytes
                response_size_mb = len(image_bytes) / (1024 * 1024)  # Convert bytes to MB
                print(f"Response size: {response_size_mb:.2f} MB")

                return jsonify({'success': True ,'Image': image_profile})
        # If no matching ID was found
        return jsonify({'success': False, 'error': 'ID not found'})

    except Exception as e:
        print('bye')
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)



    