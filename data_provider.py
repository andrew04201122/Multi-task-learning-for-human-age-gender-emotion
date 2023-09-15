import numpy as np
import cv2
import data_utils
import config as cf
import utils
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Datasets(object):
    def __init__(self, trainable, test_data_type='public_test'):
        self.all_data = []
        self.trainable = trainable
        self.age_train, self.age_test = data_utils.getAgeGenderImage()
        self.gender_train, self.gender_test = data_utils.getAgeGenderImage()
        self.emotion_train, self.emotion_test = data_utils.getAffectNet()

        if not trainable:
            self.test_data_type = test_data_type

        self.convert_data_format()

    def gen(self):
        np.random.shuffle(self.all_data)
        batch_images = []
        batch_labels = []
        batch_indexes = []

        for i in range(len(self.all_data)):
            image, label, index = self.all_data[i]
            batch_images.append(image)
            batch_labels.append(label)
            batch_indexes.append(index)

            if len(batch_images) == cf.BATCH_SIZE:
                yield batch_images, batch_labels, batch_indexes
                batch_images = []
                batch_labels = []
                batch_indexes = []

        if len(batch_images) > 0:
            yield batch_images, batch_labels, batch_indexes


    def convert_data_format(self):
        if self.trainable:
            # Age dataset
            for i, (images,gender,age) in enumerate(self.age_train):
                print("age")
                x = age.size()
                x = list(x)
                num = x[0]   
                for j in range (num):
                    index = 3.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(age[j]))
                    self.all_data.append((img, label, index))
            # emotion dataset
            for i, (images,labels) in enumerate(self.emotion_train):
                print("emotion")
                x = labels.size()
                x = list(x)
                num = x[0]      
                for j in range (num):
                    index = 5.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(labels[j]))
                    self.all_data.append((img, label, index))

             # gender dataset
            for i, (images,gender,age) in enumerate(self.gender_train):
                print(len(images))  
                x = gender.size()
                x = list(x)
                num = x[0]    
                for j in range (num):
                    index = 4.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(gender[j]))
                    self.all_data.append((img,label,index)) 
        
        else:
            for i, (images,gender,age) in enumerate(self.age_test): 
                print("age")  
                x = age.size()
                x = list(x)
                num = x[0]   
                for j in range (num):
                    index = 3.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(labels[j]))
                    self.all_data.append((img,label,index))

            # emotion dataset
            for i, (images,labels) in enumerate(self.emotion_test):   
                print("emotion")
                x = labels.size()
                x = list(x)
                num = x[0]
                for j in range (num):
                    index = 5.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(labels[j]))
                    self.all_data.append((img,label,index))

             # gender dataset
            for i, (images,gender,age) in enumerate(self.gender_test):   
                print("gender")
                x = gender.size()
                x = list(x)
                num = x[0]
                for j in range (num):
                    index = 4.0
                    img = images[j].permute(1, 2, 0).numpy()
                    label = utils.get_one_hot_vector(8, int(labels[j]))
                    self.all_data.append((img,label,index))
