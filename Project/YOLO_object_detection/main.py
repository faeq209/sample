import cv2
import cvzone
import math
import time
from ultralytics import YOLO


# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("cars_4.mp4")  # For Video
#cap.set(3, 480)
#cap.set(4, 480)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
myColor = (0, 128, 0)
limits = [150, 220, 350, 220]
start_timer=0
end_timer=0
fps=0.0
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    start_timer=time.time()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.5 and (currentClass =='bicycle' or currentClass =='car' or currentClass == "motorbike" or currentClass =='bus' or currentClass =='train' or currentClass =='truck'):
                       if(y2>=220):
                           myColor=(0,0,255)
                           cv2.putText(img,"warning", (150,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 1, cv2.LINE_AA)

                       else:
                           myColor=(0,128,0)
                           
                       cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1.3, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                       cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                       cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)


                
    if cv2.waitKey(1)==ord('q'):
       break
    end_timer=time.time()
    fps=1/(end_timer-start_timer)
    cv2.putText(img,str(round(fps)), (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

