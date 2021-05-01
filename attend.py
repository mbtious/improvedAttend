import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time


path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    classNames = [i.split('.')[0] for i in classNames]

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList







def markAttendance(name):
    with open('attend1.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%D')
            f.writelines(f'\n{name},{dtString}')
            cv2.putText(img, 'Attendance Registered', (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)



encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)



while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    time.sleep(1)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    cv2.rectangle(img, (200, 60,
                  250,350), (255, 255, 255), 8)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace,0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        cv2.putText(img, f'{round(faceDis[0], 2)} ',
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),
                    3)
        if round(faceDis[0], 2) > 0.6:
            tip = ('Unknown Face, Try keeping your face straight')
            cv2.putText(img, f'{tip} ',
                        (0, 250), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 0, 255),
                        2)

        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            print(name)



    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
