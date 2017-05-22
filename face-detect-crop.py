import numpy as np
import cv2
import os, os.path

#multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

DIR = './GENKI4K/files'
OUTPUT_DIR = './cropped'
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# i = 0

for pic in sorted(os.listdir(DIR)):
    # i += 1
    # if i != 30 and i != 31:
    #     continue

    image_file = os.path.join(DIR, pic)
    try:
      # img = cv2.imread('./datasets/file'+str(pic)+'.jpg')
      img = cv2.imread(image_file);
      

      height = img.shape[0]
      width = img.shape[1]
      size = height * width

      if size > (500^2):
          r = 500.0 / img.shape[1]
          dim = (500, int(img.shape[0] * r))
          img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
          img = img2

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 5)
      # print faces
      eyesn = 0

      # cv2.imshow(pic,img)

      for (x,y,w,h) in faces:
          imgCrop = img[y:y+h,x:x+w]

          #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]

          eyes = eye_cascade.detectMultiScale(roi_gray)
          for (ex,ey,ew,eh) in eyes:
              #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
              eyesn = eyesn +1
          if eyesn >= 2:
              cv2.imwrite(os.path.join(OUTPUT_DIR, pic), imgCrop)

      imgCrop = cv2.resize(imgCrop, (64,64))
      cv2.imwrite(os.path.join(OUTPUT_DIR, pic), imgCrop)
      # if i == 30 or i == 31:
        # print image_file
        # cv2.imshow(pic,imgCrop)
        # cv2.waitKey(3000)
      # if i == 31:
        # break

      
      print("Image"+image_file+" has been processed and cropped")
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
    except IOError as e:
      print('Could not read: ', image_file, ' : ', e)

#cap.release()
print("All images have been processed!!!")
cv2.destroyAllWindows()
cv2.destroyAllWindows()