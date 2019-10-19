import sys
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tqdm

directory = sys.argv[1]
imagecount = int(sys.argv[2])

os.makedirs(directory, exist_ok=True)

video = cv2.VideoCapture(0)

filename = len(os.listdir(directory))
count = 0
pbar = tqdm(total = imagecount+1)
while True and count < imagecount:

    filename += 1
    count += 1
    _, frame = video.read()
    kernel = np.ones((3,3),np.uint8)
     
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


     
    # define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    #extract skin colur image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)



    #extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)

    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)
    path = directory+"//"+str(filename)+".jpg"
    cv2.imwrite(path, mask)
    cv2.imshow("Capturing", mask)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
pbar.update(1)
pbar.close()
video.release()
cv2.destroyAllWindows()
