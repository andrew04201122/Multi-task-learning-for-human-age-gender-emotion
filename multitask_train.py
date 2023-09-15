import model
import tensorflow as tf
def main():
    session = tf.compat.v1.Session()
    multitask_model = model.Model(session, trainable=True)
    print("yes")
    multitask_model.train()

if __name__ == '__main__':
    main()
