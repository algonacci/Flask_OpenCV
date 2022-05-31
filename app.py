import os
from flask import Flask, render_template, Response
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


def gen_frames():
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

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
	app.run(debug=True)