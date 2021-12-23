import os
import matplotlib.pyplot  as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import seaborn as sns
import utils
import detect_face

def train_CNN(data_path):
    """根据data_path中的数据训练人脸检测CNN"""
    with tf.device('/gpu:0'):
        # 创建训练数据集
        dataset = []
        train_dataset = []
        total_images = 0 # 总图片数
        for i in range(0,20,1):
            path = data_path + "ID" + str(i+1) + "/"
            files = os.listdir(path)
            # np.random.shuffle(files)
            temp = 0 # 计数
            label_data = []
            for k in range(0,len(files)-2,1): # generate train data
                video,fps = utils.read_video(path + files[k])
                l,h,w,_ = video.shape
                step = max(1,int((len(files)-2)*l/500)) # 采样步长，尽可能使每个label下数据量大致相等
                for j in range(0,l,step):
                    label_data.append((detect_face.detect_face(video[j],120,120,3),i))
                    temp += 1    
            train_dataset += label_data 
            for k in range(len(files)-2,len(files),1): # generate valid & test date
                video,fps = utils.read_video(path + files[k])
                l,h,w,_ = video.shape
                step = int(l/50) # 采样步长，尽可能使每个label下数据量大致相等
                for j in range(0,l,step):
                    dataset.append((detect_face.detect_face(video[j],120,120,3),i))
            # np.random.shuffle(label_data) # 随机打乱顺序
            total_images += temp
            print("Finished input data: ID" + str(i+1) + " \t Total Images:\t " + str(temp))

        np.random.shuffle(dataset) # 随机打乱顺序
        np.random.shuffle(train_dataset) # 随机打乱顺序
        # # train_dataset = dataset[0:int(0.7*len(dataset))]
        # valid_dataset = dataset[0:int(0.6*len(dataset))]
        # test_dataset = dataset[int(0.6*len(dataset)):len(dataset)]
        valid_dataset = dataset

        # 预处理训练集
        print("\nCreating Train Dataset...")
        train_images = []
        train_labels = []
        for train_data in train_dataset:
            train_images.append(train_data[0])
            train_labels.append(train_data[1])
        train_images = tf.convert_to_tensor(train_images ,dtype=tf.float32)
        train_labels = tf.convert_to_tensor(train_labels)
        # train_labels = tf.one_hot(train_labels,depth=20)
    # with tf.device('/cpu:0'):
    #     data_augmentation = models.Sequential()
        # data_augmentation.add(layers.RandomFlip("horizontal",input_shape=(120,120,3),seed=0.15))
        # data_augmentation.add(layers.RandomRotation(0.2))
        # data_augmentation.add(layers.RandomZoom(height_factor=(0.05,0.95),width_factor=(0.05,0.95),fill_mode='constant',
        #     interpolation='bilinear', seed=0.2, fill_value=0.0,))
        # data_augmentation.add(layers.RandomContrast(0.15,0.15))
        # data_augmentation.summary()
        # train_images = data_augmentation(train_images)
        print("\nTrain Dataset Completed!\n")
    
    with tf.device('/gpu:0'):
        # 预处理训练集
        print("Creating Valid Dataset...")
        valid_images = []
        valid_labels = []
        for valid_data in valid_dataset:
            valid_images.append(valid_data[0])
            valid_labels.append(valid_data[1])
        valid_images = tf.convert_to_tensor(valid_images ,dtype=tf.float32)
        valid_labels = tf.convert_to_tensor(valid_labels)
        # valid_labels = tf.one_hot(valid_labels,depth=20)
        print("\nValid Data Completed!\n")

        # # 预处理训练集
        # print("Creating Test Dataset...")
        # test_images = []
        # test_labels = []
        # for test_data in test_dataset:
        #     test_images.append(test_data[0])
        #     test_labels.append(test_data[1])
        # test_images = tf.convert_to_tensor(test_images ,dtype=tf.float32)
        # test_labels = tf.convert_to_tensor(test_labels)
        # # test_labels = tf.one_hot(test_labels,depth=20)
        # print("\nTest Data Completed!\n")

        # 定义模型
        model = models.Sequential()
        model.add(layers.Rescaling(1./255, input_shape=(120,120,3)))
        model.add(layers.Conv2D(32, (5, 5),
                kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (5, 5),
                    kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(128, (3, 3),
                    kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(128, (3, 3),
                    kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024,
                    kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(512,
                    kernel_regularizer=keras.regularizers.l2(0.002), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(20, activation = 'softmax'))

        model.summary()  # 显示完整模型结构

        # 编译模型
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # 训练模型
        history = model.fit(train_images, train_labels, epochs=20, shuffle=True,
                    validation_data=(valid_images, valid_labels))
        
        # 保存模型
        model.save(filepath='./train/')

        # 评估模型
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        # 混淆矩阵
        # Confusion Matrix of Train data
        predictions_train = model.predict(train_images)
        pred_train = []
        for prediction in predictions_train:
            pred_train.append(np.argmax(prediction))
        con_mat_train = tf.math.confusion_matrix(train_labels, pred_train, num_classes=20, dtype=tf.dtypes.float32)
        con_mat_train = np.array(con_mat_train).T
        con_mat_train = con_mat_train/con_mat_train.sum(axis=0) # 每行归一化
        con_mat_train = con_mat_train.T
        plt.figure()
        sns.heatmap(con_mat_train, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title("Confusion Matrix of Train Data")

        # Confusion Matrix of Valid data
        predictions_valid = model.predict(valid_images)
        pred_valid = []
        for prediction in predictions_valid:
            pred_valid.append(np.argmax(prediction))
        con_mat_valid = tf.math.confusion_matrix(valid_labels, pred_valid, num_classes=20, dtype=tf.dtypes.float32)
        con_mat_valid = np.array(con_mat_valid).T
        con_mat_valid = con_mat_valid/con_mat_valid.sum(axis=0) # 每行归一化
        con_mat_valid = con_mat_valid.T
        plt.figure()
        sns.heatmap(con_mat_valid, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title("Confusion Matrix of Valid Data")

        # # Confusion Matrix of Test data
        # predictions_test = model.predict(test_images)
        # pred_test = []
        # for prediction in predictions_test:
        #     pred_test.append(np.argmax(prediction))
        # con_mat_test = tf.math.confusion_matrix(test_labels, pred_test, num_classes=20, dtype=tf.dtypes.float32)
        # con_mat_test = np.array(con_mat_test).T
        # con_mat_test = con_mat_test/con_mat_test.sum(axis=0) # 每行归一化
        # con_mat_test = con_mat_test.T
        # plt.figure()
        # sns.heatmap(con_mat_test, annot=True, cmap='Blues')
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title("Confusion Matrix of Test Data")

        plt.show()

    return model

if __name__ == "__main__":
    probability_model = train_CNN("./train/")
