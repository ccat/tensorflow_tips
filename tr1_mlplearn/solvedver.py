import numpy as np
import tensorflow as tf


inputs = [
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.]
]

winning_hands = [
  [0., 1., 0.],
  [0., 0., 1.],
  [1., 0., 0.]
]

def main():
    supervisor_labels_placeholder = tf.placeholder("float", [None, 3],name="teacher")
    input_placeholder = tf.placeholder("float", [None, 3],name="input")

    feed_dict={input_placeholder: inputs, supervisor_labels_placeholder: winning_hands}

    with tf.Session() as sess:
        Wf = tf.Variable(tf.random_normal([3, 3]))
        bf = tf.Variable(tf.random_normal([3]))
        iLayerY=tf.add(tf.matmul(input_placeholder, Wf) , bf)

        Wl = tf.Variable(tf.random_normal([3,3]))
        bl = tf.Variable(tf.random_normal([3]))
        y = tf.add(tf.matmul(iLayerY, Wl) , bl)

        losss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=supervisor_labels_placeholder))
        compute_op = tf.train.GradientDescentOptimizer(0.01).compute_gradients(losss)
        training_op = tf.train.GradientDescentOptimizer(0.01).minimize(losss)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(supervisor_labels_placeholder,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(100):
            sess.run(training_op, feed_dict=feed_dict)
            print unicode(sess.run(losss, feed_dict=feed_dict))+","+unicode(accuracy.eval( feed_dict=feed_dict))
        print sess.run(compute_op, feed_dict=feed_dict)

if __name__ == '__main__':
    import sys
    args = sys.argv
    main()
