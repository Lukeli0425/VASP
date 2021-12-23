import cv2,os
from numpy.lib import utils
import tensorflow as tf
import utils

def detect_face(image,w0,h0,l0=3):
    """从图片中检测人脸，返回人脸数量n和人脸图片列表faces_resized"""
    h,w,_ = image.shape
    if l0 == 1:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # 转化为灰度图
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
            image,
            scaleFactor = 1.05,
            minNeighbors = 5,
            minSize = (100,100)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) > 0: # 识别到了人脸
        [x,y,w_face,h_face] = faces[0]
        # print(x,y,w_face,h_face)
        # 确定人脸x坐标范围
        if x + int(w_face/2) - int(w0/2) < 0: 
            x_low = 0
            x_high = w0
        elif x + int(w_face/2) + int(w0/2) >= w:
            x_low = w - w0
            x_high = w
        else:
            x_low = x + int(w_face/2) - int(w0/2)
            x_high = x + int(w_face/2) + int(w0/2)
        # 确定人脸y坐标范围
        if y + int(h_face/2) - int(h0/2) < 0: 
            y_low = 0
            y_high = h0
        elif y + int(h_face/2) + int(h0/2) >= h:
            y_low = h - h0
            y_high = h
        else:
            y_low = y + int(h_face/2) - int(h0/2)
            y_high = y + int(h_face/2) + int(h0/2)
    else: # 如果未识别出人脸，就返回一个靠近图片上部的截图
        x_low = int(w/2) - int(w0/2)
        x_high = int(w/2) + int(w0/2)
        y_low = 5
        y_high = 5 + h0
    # print(x_low,x_high,y_low,y_high)
    if l0 == 1:
        face = image[y_low:y_high,x_low:x_high]
    else:
        face = image[y_low:y_high,x_low:x_high,:]
    # cv2.rectangle(image, (x_low, y_low), (x_high, y_high), (0,0,255), thickness=2)
    # cv2.imshow("face",face)
    # cv2.waitKey(0)
    # print(face.shape)
    return face

if __name__ == "__main__":
    data_path = "./train/"
    e = 0
    for i in range(0,20,1):
        path = data_path + "ID" + str(i+1) + "/"
        files = os.listdir(path)
        temp = 0 # 计数
        face_num = 0
        for file in files:
            video,_ = utils.read_video(path + file)
            l,h,w,_ = video.shape
            face = detect_face(video[10],120,120)
            temp += 1
            face_num += 1
        print("Finished input data: ID" + str(i+1) + "  Total Videos: " + str(temp) + "  Total Faces: " + str(face_num))
    print("Videos without face detected:" + str(e))