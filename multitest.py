import model
import tensorflow as tf
sess = tf.compat.v1.Session()
multitask_model = model.Model(session=sess, trainable=False)
multitask_model.test()
#multitask_model.valid()
#multitask_model.count_trainable_params()