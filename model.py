import tensorflow as tf
import os
import config
from data_provider import Datasets
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import random
from torchvision.transforms import transforms
from PIL import Image
#df = pd.read_csv('C:/Users/User/Desktop/Adience_affectnet_28800.csv')
df = pd.read_csv('C:/Users/User/Desktop/final/newdata/mix/train_set.csv')
df_test = pd.read_csv('C:/Users/User/Desktop/final/newdata/mix/test_set.csv')
from transforms import (RandomErasing, get_color_distortion,get_gaussian_blur)
tf.compat.v1.disable_eager_execution()
test_age_acc = []
test_emotion_acc = []
test_gender_acc = []

class Model(object):
    def __init__(self, session, trainable=True, prediction=False):
        self.global_step = tf.compat.v1.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.batch_size = config.BATCH_SIZE
        self.sess = session
        self.model_dir = config.MODEL_DIR 
        self.trainable = trainable
        self.prediction = prediction
        self.num_epochs = config.EPOCHS

        # Building model
        self._define_input()
        self._build_model()

        if not prediction:
            self._define_loss()
            # Learning rate and train op
            learningRate = tf.compat.v1.train.exponential_decay(learning_rate=config.INIT_LR, global_step=self.global_step,decay_steps=config.DECAY_STEP, decay_rate=config.DECAY_LR, staircase=True)
            #self.train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = config.INIT_LR).minimize(self.total_loss,global_step=self.global_step)
            #self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(self.total_loss,global_step=self.global_step)
            self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learningRate,beta1=0.5, beta2=0.999).minimize(self.total_loss,global_step=self.global_step)
            # Input data
            #self.data = Datasets(trainable=self.trainable, test_data_type='public_test')

        # Init checkpoints
        self.saver_all = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5) 
        #self.checkpoint_path = os.path.join(self.model_dir, 'model.ckpt')
        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        if ckpt:
            print('Reading model parameters from %s', ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh parameters.')
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def _define_input(self):
        self.input_images = tf.compat.v1.placeholder(tf.float32, [None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)
        self.phase_train = tf.compat.v1.placeholder(tf.bool)
        if not self.prediction:
            #self.input_labels = tf.compat.v1.placeholder(tf.float32, [None, 8])
            #self.input_indexes = tf.compat.v1.placeholder(tf.float32, [None])
            self.input_age_label = tf.compat.v1.placeholder(tf.float32, [None, 8])
            self.input_gender_label = tf.compat.v1.placeholder(tf.float32, [None, 8])
            self.input_emotion_label = tf.compat.v1.placeholder(tf.float32, [None, 8])

    def _build_model(self):
        # Extract features
        based_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
        x = based_model(self.input_images,training = True)
        x = utils.max_pool(x, 2, 2)
        # Gender branch
        gender_fc0 = utils.FC('gender_fc0', x, 512, self.keep_prob)
        gender_fc1 = utils.FC('gender_fc1', gender_fc0, 256, self.keep_prob)
        gender_fc2 = utils.FC('gender_fc2', gender_fc1, 128, self.keep_prob)
        gender_fc3 = utils.FC('gender_fc3', gender_fc2, 64, self.keep_prob)
        gender_fc4 = utils.FC('gender_fc4', gender_fc3, 32, self.keep_prob)
        self.y_gender_conv = utils.FC('gender_softmax', gender_fc4, 2, self.keep_prob, 'softmax')


        # Age branch
        age_fc0 = utils.FC('age_fc0', x, 512, self.keep_prob)
        age_fc1 = utils.FC('age_fc1', age_fc0, 256, self.keep_prob)
        age_fc2 = utils.FC('age_fc2', age_fc1, 128, self.keep_prob)
        self.y_age_conv = utils.FC('age_softmax', age_fc2, 8, self.keep_prob, 'softmax')


        #emotion branch 
        emotion_fc0 = utils.FC('emotion_fc0', x, 512, self.keep_prob)
        emotion_fc1 = utils.FC('emotion_fc1', emotion_fc0, 256, self.keep_prob)
        emotion_fc2 = utils.FC('emotion_fc2', emotion_fc1, 128, self.keep_prob)
        self.y_emotion_conv = utils.FC('emotion_softmax', emotion_fc2, 8, self.keep_prob, 'softmax')


#---------------------------------------------------------------------------------------------------

    def _define_loss(self):
        self.y_age = self.input_age_label[:, :8]
        self.y_gender = self.input_gender_label[:, :2]
        self.y_emotion = self.input_emotion_label[:, :8]
#================================================================
        
        # Extra variables 
        age_correct_prediction = tf.equal(tf.argmax(self.y_age_conv, 1), tf.argmax(self.y_age, 1))
        gender_correct_prediction = tf.equal(tf.argmax(self.y_gender_conv, 1), tf.argmax(self.y_gender, 1))
        emotion_correct_prediction = tf.equal(tf.argmax(self.y_emotion_conv, 1), tf.argmax(self.y_emotion, 1))

        self.age_label = tf.argmax(self.y_age, 1)
        self.age_pred = tf.argmax(self.y_age_conv, 1)

        self.gender_label = tf.argmax(self.y_gender, 1)
        self.gender_pred = tf.argmax(self.y_gender_conv, 1)

        self.emotion_label = tf.argmax(self.y_emotion, 1)
        self.emotion_pred = tf.argmax(self.y_emotion_conv, 1)

        self.age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32))
        self.gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32))
        self.emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32))

        self.age_cross_entropy = tf.reduce_mean(-
            tf.reduce_sum(self.y_age * tf.math.log(tf.clip_by_value(self.y_age_conv, 1e-10, 1.0)),axis=1))

        self.gender_cross_entropy = tf.reduce_mean(-
            tf.reduce_sum(self.y_gender * tf.math.log(tf.clip_by_value(self.y_gender_conv, 1e-10, 1.0)),axis=1))

        self.emotion_cross_entropy = tf.reduce_mean(-
            tf.reduce_sum(self.y_emotion * tf.math.log(tf.clip_by_value(self.y_emotion_conv, 1e-10, 1.0)),axis=1))
        
        """self.age_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_age * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_age_conv), 1e-10, 1.0)),
                          axis=1) * self.age_mask)

        self.gender_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_gender * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_gender_conv), 1e-10, 1.0)),
                          axis=1) * self.gender_mask)

        self.emotion_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_emotion * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_emotion_conv), 1e-10, 1.0)),
                          axis=1) * self.emotion_mask)"""
        

        # Add l2 regularizer
        l2_loss = []
        for var in tf.compat.v1.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                l2_loss.append(tf.nn.l2_loss(var))

        self.l2_loss = config.WEIGHT_DECAY * tf.add_n(l2_loss)

        self.total_loss = 0.25*self.age_cross_entropy + 0.5*self.gender_cross_entropy + self.l2_loss + 0.25*self.emotion_cross_entropy


    def plot_loss_graph(self,total):
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("total_loss")
        label='total loss'
        total=plt.plot(total,label=label)
        plt.legend(loc='best')
        plt.show()
    
    def plot_acc_graph(self,age,gender,emotion,acc):
        if acc=="loss":
            plt.ylabel("loss")
        elif acc=="acc":
            plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.title("multitask_acc")
        label=acc+'_age'
        age=plt.plot(age,label=label)
        label=acc+'_gender'
        gender=plt.plot(gender,label=label)
        label=acc+'_emotion'
        emotion=plt.plot(emotion,label=label)
        plt.legend(loc='best')
        plt.show()

    @staticmethod
    def count_trainable_params():
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def train(self):
        age_train_re = []
        emotion_train_re = []
        gender_train_re = []
        age_loss_re = []
        gender_loss_re = []
        emotion_loss_re = []
        total_loss_re = []
        x = df.to_numpy()
        data_num = int(x.size/4)
        for epoch in range(self.num_epochs):
            age_nb_true_pred = 0
            gender_nb_true_pred = 0
            emotion_nb_true_pred = 0 

            transformer=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(),
                transforms.RandomApply([transforms.Lambda(get_gaussian_blur)], p=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomErasing(probability=0.3, sh=0.4, r1=0.3)
            ])


            L = [i for i in range(data_num)]

            random.shuffle(L)
            num = 0
            print("=======================================================================")
            print('Epoch %d/%d: ' % (epoch + 1, config.EPOCHS))
            for i in range(int(data_num/config.BATCH_SIZE)):
                age_label = []
                gender_label = []
                emotion_label = []
                batch_img = []
                for j in range(config.BATCH_SIZE):
                    trans_img = transformer(Image.open(x[L[num]][0]))
                    trans_img = trans_img.permute(1, 2, 0).numpy()
                    batch_img.append(trans_img)
                    age_label_onehot = utils.get_one_hot_vector(8,x[L[num]][1])
                    age_label.append(age_label_onehot)

                    gender_label_onehot = utils.get_one_hot_vector(8,x[L[num]][2])
                    gender_label.append(gender_label_onehot)

                    emotion_label_onehot = utils.get_one_hot_vector(8,x[L[num]][3])
                    emotion_label.append(emotion_label_onehot)
                    
                    num = num + 1
                
                feed_dict = {self.input_images: batch_img,
                             self.input_age_label: age_label,
                             self.input_gender_label: gender_label,
                             self.input_emotion_label: emotion_label,
                             self.keep_prob: 0.3,
                             self.phase_train: self.trainable
                             }
                              
                ttl, agl, eml, gel, l2l, _ = self.sess.run([self.total_loss,self.age_cross_entropy,self.emotion_cross_entropy,
                                                            self.gender_cross_entropy,self.l2_loss,self.train_step], feed_dict=feed_dict)
                

                age_nb_true_pred += self.sess.run(self.age_true_pred, feed_dict=feed_dict)
                gender_nb_true_pred += self.sess.run(self.gender_true_pred, feed_dict=feed_dict)
                emotion_nb_true_pred += self.sess.run(self.emotion_true_pred, feed_dict=feed_dict)


                print('age_loss: %.2f, emotion_loss: %.2f, gender_loss: %.2f, l2_loss: %.2f, total_loss: %.2f\r' % (agl, eml, gel, l2l, ttl), end="")

            age_loss_re.append(agl)
            gender_loss_re.append(gel)
            emotion_loss_re.append(eml)
            total_loss_re.append(ttl)

            age_train_acc = age_nb_true_pred * 1.0 / data_num
            gender_train_acc = gender_nb_true_pred * 1.0 / data_num
            emotion_train_acc = emotion_nb_true_pred * 1.0 / data_num 

            age_train_re.append(age_train_acc)
            emotion_train_re.append(emotion_train_acc)
            gender_train_re.append(gender_train_acc)

            print('\n')
            print('Age task train accuracy: ', str(age_train_acc * 100))
            print('Gender task train accuracy: ', str(gender_train_acc * 100))
            print('emotion task train accuracy: ', str(emotion_train_acc * 100))
            self.saver_all.save(self.sess, self.model_dir + '/model.ckpt')
            self.valid()

        self.plot_acc_graph(age_train_re,gender_train_re,emotion_train_re,"acc")
        self.plot_acc_graph(age_loss_re,gender_loss_re,emotion_loss_re,"loss")
        self.plot_acc_graph(test_age_acc,test_gender_acc,test_emotion_acc,"acc")    
            
        

    def test(self):
        # Evaluate model on the test data
        age_nb_true_pred_test = 0
        gender_nb_true_pred_test = 0
        emotion_nb_true_pred_test = 0
        age_pred = []
        age_true= []
        gender_pred = []
        gender_true= []
        emotion_pred = []
        emotion_true= []
        transformer=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        test = df_test.to_numpy()
        test_num = int(test.size/4)
        num_test = 0
        L_test = [i for i in range(test_num)]
        random.shuffle(L_test)
        
        for k in range(int(test_num/config.BATCH_SIZE)):
            age_label = []
            gender_label = []
            emotion_label = []
            batch_img = []
            for j in range(config.BATCH_SIZE):
                trans_img = transformer(Image.open(test[L_test[num_test]][0]))
                trans_img = trans_img.permute(1, 2, 0).numpy()
                batch_img.append(trans_img)
                age_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][1])
                age_label.append(age_label_onehot)

                gender_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][2])
                gender_label.append(gender_label_onehot)

                emotion_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][3])
                emotion_label.append(emotion_label_onehot)

                num_test = num_test + 1

            feed_dict = {self.input_images: batch_img,
                        self.input_age_label: age_label,
                        self.input_gender_label: gender_label,
                        self.input_emotion_label: emotion_label,
                        self.keep_prob: 0,
                        self.phase_train: self.trainable
                        }

            age_nb_true_pred_test += self.sess.run(self.age_true_pred, feed_dict)
            gender_nb_true_pred_test += self.sess.run(self.gender_true_pred, feed_dict)
            emotion_nb_true_pred_test += self.sess.run(self.emotion_true_pred, feed_dict)

            #x = self.sess.run(self.age_mask, feed_dict)
            y_age = self.sess.run(self.age_pred, feed_dict)
            z_age = self.sess.run(self.age_label,feed_dict)

            #x_gen = self.sess.run(self.gender_mask, feed_dict)
            y_gen = self.sess.run(self.gender_pred, feed_dict)
            z_gen = self.sess.run(self.gender_label,feed_dict)
            
            
            #x_emo = self.sess.run(self.emotion_mask, feed_dict)
            y_emo = self.sess.run(self.emotion_pred, feed_dict)
            z_emo = self.sess.run(self.emotion_label,feed_dict)

            agep = []
            agel = []
            genp = []
            genl = []
            emop = []
            emol = []
            for i in range (32):
                agep.append(y_age[i])
                agel.append(z_age[i])
                genp.append(y_gen[i])
                genl.append(z_gen[i])
                emop.append(y_emo[i])
                emol.append(z_emo[i])

            age_pred.extend(agep)
            age_true.extend(agel)
            gender_pred.extend(genp)
            gender_true.extend(genl)
            emotion_pred.extend(emop)
            emotion_true.extend(emol)
        print(num_test)
        gender_test_accuracy = gender_nb_true_pred_test * 1.0 / num_test
        age_test_accuracy = age_nb_true_pred_test * 1.0 / num_test
        emotion_test_accuracy = emotion_nb_true_pred_test * 1.0 / num_test
        
        cf_matrix = confusion_matrix(age_true, age_pred, normalize="true")
        cf_matrix_gen = confusion_matrix(gender_true, gender_pred, normalize="true")
        cf_matrix_emo = confusion_matrix(emotion_true, emotion_pred, normalize="true")
        
        print('\nResult: ')
        print('Age task test accuracy: ' + str(age_test_accuracy * 100))
        print('Gender task test accuracy: ' + str(gender_test_accuracy * 100))
        print('Emotion task test accuracy: ' + str(emotion_test_accuracy * 100))
        #draw confusion matrix
        
        class_names = ["0-2","4-6","8-13","15-20","25-32","38-43","48-53",">=60"]
        class_names_gen = ["female","male"]
        class_names_emo = ["angry","disgust","fear","happy","neutral","sad","surprised","Contempt"]
        df_cm = pd.DataFrame(cf_matrix, class_names, class_names) 
        df_cm_gen = pd.DataFrame(cf_matrix_gen, class_names_gen, class_names_gen)
        df_cm_emo = pd.DataFrame(cf_matrix_emo, class_names_emo, class_names_emo)
        plt.figure(figsize = (9,6))

        """sns.heatmap(df_cm, annot=True, fmt=".2f", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig('C:/Users/User/Desktop/result/1116/sudo/leaky_mul_age_confusion_matrix.png')"""
        
        """sns.heatmap(df_cm_gen, annot=True, fmt=".2f", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig("C:/Users/User/Desktop/result/1116/sudo/leaky_mul_gender_confusion_matrix.png")"""

        sns.heatmap(df_cm_emo, annot=True, fmt=".2f", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig("C:/Users/User/Desktop/result/1116/sudo/leaky_mul_emotion_confusion_matrix.png")
        

    def valid(self):
        age_nb_true_pred_test = 0
        gender_nb_true_pred_test = 0
        emotion_nb_true_pred_test = 0

        transformer=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        test = df_test.to_numpy()
        test_num = int(test.size/4)
        num_test = 0
        L_test = [i for i in range(test_num)]
        random.shuffle(L_test)
        
        for k in range(int(test_num/config.BATCH_SIZE)):
            age_label = []
            gender_label = []
            emotion_label = []
            batch_img = []
            for j in range(config.BATCH_SIZE):
                trans_img = transformer(Image.open(test[L_test[num_test]][0]))
                trans_img = trans_img.permute(1, 2, 0).numpy()
                batch_img.append(trans_img)
                age_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][1])
                age_label.append(age_label_onehot)

                gender_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][2])
                gender_label.append(gender_label_onehot)

                emotion_label_onehot = utils.get_one_hot_vector(8,test[L_test[num_test]][3])
                emotion_label.append(emotion_label_onehot)

            num_test = num_test + 32
            #print(num_test)
            
            feed_dict = {self.input_images: batch_img,
                        self.input_age_label: age_label,
                        self.input_gender_label: gender_label,
                        self.input_emotion_label: emotion_label,
                        self.keep_prob: 0,
                        self.phase_train: self.trainable
                        }

            
            age_nb_true_pred_test += self.sess.run(self.age_true_pred, feed_dict)
            gender_nb_true_pred_test += self.sess.run(self.gender_true_pred, feed_dict)
            emotion_nb_true_pred_test += self.sess.run(self.emotion_true_pred, feed_dict)  

        #print(f"num_test {num_test} != test_num {test_num}")

        gender_test_accuracy = gender_nb_true_pred_test * 1.0 / num_test
        age_test_accuracy = age_nb_true_pred_test * 1.0 / num_test
        emotion_test_accuracy = emotion_nb_true_pred_test * 1.0 / num_test

        test_age_acc.append(age_test_accuracy)
        test_gender_acc.append(gender_test_accuracy)
        test_emotion_acc.append(emotion_test_accuracy)
        
        print('\nResult: ')
        print('Age task test accuracy: ' + str(age_test_accuracy * 100))
        print('Gender task test accuracy: ' + str(gender_test_accuracy * 100))
        print('Emotion task test accuracy: ' + str(emotion_test_accuracy * 100))



    def predict(self, image):
        GENDER_DICT = {0: 'Female', 1: 'Male'}
        AGE_DICT = {0: '10-25', 1: '26-40', 2: '41-55', 3: '56-70', 4: '71-85'}
        EMOTION_DICT = {0:'Angry', 1:'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        labels = []

        feed_dict = {self.input_images: image,
                     self.keep_prob: 0,
                     self.phase_train: self.trainable}

        age_prediction_idx = self.sess.run(tf.argmax(self.y_age_conv, axis=1), feed_dict=feed_dict)
        gender_prediction_idx = self.sess.run(tf.argmax(self.y_gender_conv, axis=1), feed_dict=feed_dict)
        emotion_prediction_idx = self.sess.run(tf.argmax(self.y_emotion_conv, axis=1), feed_dict=feed_dict)


        for i in range(len(gender_prediction_idx)):
            age_label = AGE_DICT[age_prediction_idx[i]] 
            emotion_label = EMOTION_DICT[emotion_prediction_idx[i]]
            gender_label = GENDER_DICT[gender_prediction_idx[i]]

            labels.append((age_label, emotion_label, gender_label))

        return labels
