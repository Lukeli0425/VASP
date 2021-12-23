import numpy as np
import face_recognition
import os
import utils

def read_faces(data_path):
    """读入data_path中的视频，记录其中每个标签的人脸信息用于后续判断"""
    known_faces = []
    min_l = np.Inf # 最短视频长度
    for i in range(0,20,1):
        faces = []
        path = data_path + "ID" + str(i+1) + "/"
        files = os.listdir(path)
        for file in files:
            if not file[-4:] == '.mp4':
                continue
            video,fps = utils.read_video(path + file)
            l,_,_,_ = video.shape
            if l < min_l:
                min_l = l
            step = max(1,int(len(files)*l/150)) # 采样步长，尽可能使每个label下数据量大致相等
            for j in range(0,l,step):
                face = video[j]
                try:
                    face_encoding = face_recognition.face_encodings(face)[0]
                except IndexError:
                    # print("I wasn't able to locate any faces in " + path + file + " !")
                    continue
                faces.append(face_encoding)
        known_faces.append(faces)
        print("\nReading Face done: ID " + str(i+1))
        # print("\nMinimu length: " + str(min_l/25) + "\n")
    return known_faces

if __name__ == "__main__":
    # Load the jpg files into numpy arrays
    video_1,fps = utils.read_video(os.path.join("./train/ID1","015.mp4"))
    video_2,fps = utils.read_video(os.path.join("./train/ID2","013.mp4"))
    video_3,fps = utils.read_video(os.path.join("./train/ID3","017.mp4"))
    face_1 = video_1[10]
    face_2 = video_2[10] 
    face_3 = video_3[10] 

    # Get the face encodings for each face in each image file
    # Since there could be more than one face in each image, it returns a list of encodings.
    # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
    try:
        face_1_encoding = face_recognition.face_encodings(face_1)[0]
        face_2_encoding = face_recognition.face_encodings(face_2)[0]
        unknown_face_encoding = face_recognition.face_encodings(face_3)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    known_faces = [
        face_1_encoding,
        face_2_encoding
    ]

    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
    print(results)
    print("Is the unknown face a picture of face1? {}".format(results[0]))
    print("Is the unknown face a picture of face2? {}".format(results[1]))
    print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))