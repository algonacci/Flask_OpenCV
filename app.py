import os
from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/model.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

person = {
    1: [
        'Eric Julianto',
        'Lantai 7',
        'R707',
        'Twin Bed',
        'VIP Guest'
    ]
}

# Initialize and start realtime video capture
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # set video widht
camera.set(4, 480)  # set video height

minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)


def gen_frames_dataset():
    while True:
        count = 0
        face_id = 1
        img, frame = camera.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('image', img)

        if count >= 30:  # Take 30 face sample and stop video
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   


def gen_frames_recognition():
    while True:
        succes, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for(x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 65):  # known pasenger and match > 35%
                confidence = "  {0}%".format(round(100 - confidence))
                try:
                    datainfo = person[id]
                    cv2.rectangle(frame, (x, y), (x+w, y+h),
                                  (0, 255, 0), 2)  # add green box

                    # add flight info
                    tinggi = 5
                    for i in datainfo:
                        cv2.putText(frame, i, (x+w+5, y+tinggi),
                                    font, 1, (255, 255, 255), 2)
                        tinggi += 40
                    # confidence info
                    cv2.putText(frame, str(confidence),
                                (x+5, y+h-5), font, 1, (255, 255, 0), 1)
                except:
                    # error load pasenger
                    pass
            else:
                confidence = "  {0}%".format(round(100 - confidence))
                datainfo = ['unkown']
                # unkown passenger
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 0, 255), 2)  # add red box
                cv2.putText(
                    frame, datainfo[0], (x+w+5, y+5), font, 1, (255, 255, 255), 2)

                # confidence info
                cv2.putText(frame, str(confidence), (x+5, y+h-5),
                            font, 1, (255, 255, 0), 1)
        if not succes:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/input_id')
def inout_id():
    return render_template('input_id.html')

@app.route('/dataset')
def take_image():
    return render_template('dataset.html')

@app.route('/video_feed_dataset')
def video_feed_dataset():
	return Response(gen_frames_dataset(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_recognition')
def video_feed_recognition():
	return Response(gen_frames_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
	app.run(debug=True)