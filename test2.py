import numpy as np
import cv2
from gtts import gTTS
import os
import pygame
from pygame import mixer
import time
import requests
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
##cap = cv2.VideoCapture(1)
##j=0
url="http://100.123.57.247:8080/shot.jpg"
while True:
##        j=j+1
##        _,image=cap.read()
        img_resp=requests.get(url)
        img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
        image=cv2.imdecode(img_arr,-1)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        print ()
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        label=''
        for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = "{}".format(CLASSES[idx])
                        print("[INFO] {}".format(label))
                        print("startX=",startX)
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
##                        labels=labels+str(' ')+label
                        cv2.imshow("img",image)
                        cv2.waitKey(10)
                        name=label+".mp3"
                        if not os.path.isfile(name):
                                tts=gTTS(text=label,lang='en')
                                tts.save(name)
                        mixer.init()
                        mixer.music.load(name)
                        mixer.music.play()
                        time.sleep(2)
                        if cv2.waitKey(1)&0xFF ==ord('q'):
                            break
            

        
        
##        try:
##                tts = gTTS(text=labels, lang='en')
##                print(j)
##                tts.save("good"+str(j)+".mp3")
##                os.system("mpg321 "+"good"+str(j)+".mp3")
##                mixer.init()
##                mixer.music.load("good"+str(j)+".mp3")
##                mixer.music.play()
##                
##                time.sleep(5)
##                mixer.music.stop()
##        except:
##                pass		
##cv2.imshow("Output", image)
##cv2.waitKey(0)
        


