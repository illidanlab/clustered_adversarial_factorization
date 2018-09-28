import tensorflow as tf
import numpy as np


def frob(z):
    vec_i = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec_i, vec_i))

def lrelu(x, alpha=0.01):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# activation function
def f(x):
    return tf.tanh(x)



def create_sample_ID(n, samplesize):
    idperm_missing = np.random.permutation(n)
    idperm_notmissing = np.random.permutation(n)
    notmissing_sample_ID = np.expand_dims(idperm_notmissing[0: samplesize],0)
    missing_sample_ID = np.expand_dims(idperm_missing[0: samplesize],0)
    return notmissing_sample_ID, missing_sample_ID

class TrainModel:
    def __init__(self):


def train(X_missing_data, missing_ID,  X_notmissing_data,samplesize, weight, index1, index2, iter,  D_1, D_2,  D_3, du1, du2, lambdad, lambdar, lambdax, lambdau, lambdav, maxiter, step, pretrainstep):
    tf.set_random_seed(1)
    d, n_missing = X_missing_data.shape
    d, n_notmissing = X_notmissing_data.shape
    NotMissing_ID = ~missing_ID
    missing_ID = tf.cast(missing_ID.astype(float), tf.float32)
    missing_ID_all = tf.concat((missing_ID, tf.zeros([d, n_notmissing], tf.float32)), 1)
    NotMissing_ID = NotMissing_ID.astype(float)
    NotMissing_ID = tf.cast(NotMissing_ID, tf.float32)
    NotMissing_ID_all = tf.concat((NotMissing_ID, tf.ones([d, n_notmissing], tf.float32)), 1)
    index1 = tf.cast(index1 - 1, tf.int32)
    index2 = tf.cast(index2 - 1, tf.int32)

    X_all_data = np.concatenate((X_missing_data, X_notmissing_data), 1)

    sess = tf.InteractiveSession()
    X_missing = tf.placeholder(tf.float32, shape=(d, n_missing))
    X_notmissing = tf.placeholder(tf.float32, shape=(d, n_notmissing))
    X_all = tf.placeholder(tf.float32, shape=(d, n_missing + n_notmissing))
    real_sample_ID = tf.placeholder(tf.int32, shape=(1, samplesize))
    fake_sample_ID = tf.placeholder(tf.int32, shape=(1, samplesize))

    # initialization
    pre_train_model = scipy.io.loadmat('pretrain_model.mat')
    u1 = pre_train_model['u1']
    v = pre_train_model['v']
    u1 = tf.Variable(tf.cast(u1, tf.float32))
    v = tf.Variable(tf.cast(v, tf.float32))

    theta_R = [u1, v]

    D_W1 = tf.get_variable("D_w1" + str(lambdad) + str(pretrainstep), shape=(d, D_1), initializer=tf.contrib.layers.xavier_initializer())
    D_b1 = tf.Variable(tf.zeros(shape=[D_1]), name='D_b1')

    D_W2 = tf.get_variable("D_w2" + str(lambdad) + str(pretrainstep), shape=(D_1, D_2), initializer=tf.contrib.layers.xavier_initializer())
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

    D_W3 = tf.get_variable("D_w3" + str(lambdad) + str(pretrainstep), shape=(D_2, D_3), initializer=tf.contrib.layers.xavier_initializer())
    D_b3 = tf.Variable(tf.zeros(shape=[1]), name='D_b3')

    D_W4 = tf.get_variable("D_w4" + str(lambdad) + str(pretrainstep), shape=(D_3, 1), initializer=tf.contrib.layers.xavier_initializer())
    D_b4 = tf.Variable(tf.zeros(shape=[1]), name='D_b4')

    theta_D = [D_W1, D_W2, D_b2, D_b1, D_W3, D_b3, D_W4, D_b4]

    def discriminator(z):
        D_h1 = tf.nn.relu(tf.matmul(z, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
        D_logit = tf.matmul(D_h3, D_W4) + D_b4
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    output = (tf.matmul(u1, (v)))
    R_sample = tf.multiply(missing_ID, output[:, 0: n_missing]) + tf.multiply(NotMissing_ID, X_missing)

    Real_sampled = tf.squeeze(tf.gather(tf.transpose(X_notmissing), real_sample_ID))
    fake_sampled = tf.squeeze(tf.gather(tf.transpose(R_sample), fake_sample_ID))
    D_real, D_logit_real = discriminator(Real_sampled)
    D_fake, D_logit_fake = discriminator(fake_sampled)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))

    Vmin_1 = tf.squeeze(tf.gather(tf.transpose(v), index1))
    Vmin_2 = tf.squeeze(tf.gather(tf.transpose(v), index2))
    Vmin12 = tf.transpose(Vmin_1 - Vmin_2)
    Vmin = tf.multiply(tf.cast(weight[0: du2, :], tf.float32), Vmin12)
    fuse_norm = tf.reduce_sum(tf.reduce_sum(tf.multiply(Vmin, Vmin), 0))
    diff = tf.multiply(NotMissing_ID_all, (X_all - output))
    R_loss = -lambdad * tf.reduce_mean(tf.log(D_fake))+ lambdar * frob(diff) 
             + lambdav * frob(v) 
             + lambdau * frob(u1) 
             + lambdax * fuse_norm 


    # Only update R(X)'s parameters
    train_op_R = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(R_loss, var_list=theta_R)
    # Only update D(X)'s parameters, so var_list = theta_D
    train_op_D = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(D_loss, var_list=theta_D)

    def feed_dict_R(notmissing_sample_ID, missing_sample_ID):
        return {X_all: X_all_data, X_missing: X_missing_data, X_notmissing: X_notmissing_data,
                real_sample_ID: notmissing_sample_ID, fake_sample_ID: missing_sample_ID}

    def feed_dict_D(notmissing_sample_ID, missing_sample_ID):
        return {X_missing: X_missing_data, X_notmissing: X_notmissing_data,
                real_sample_ID: notmissing_sample_ID, fake_sample_ID: missing_sample_ID}



    tf.global_variables_initializer().run()

    #pre-train discriminator
    for iter in range(pretrainstep):
        notmissing_sample_ID, missing_sample_ID = create_sample_ID(n_missing, samplesize)
        sess.run(train_op_D, feed_dict=feed_dict_D(notmissing_sample_ID, missing_sample_ID))
    for iter in range(maxiter):
        notmissing_sample_ID, missing_sample_ID = create_sample_ID(n_missing, samplesize)
        sess.run(train_op_D, feed_dict=feed_dict_D(notmissing_sample_ID, missing_sample_ID))
        for inneriter in range(step):
            notmissing_sample_ID, missing_sample_ID = create_sample_ID(n_missing, samplesize)
            sess.run(train_op_R, feed_dict=feed_dict_R(notmissing_sample_ID, missing_sample_ID))

    R_sample = sess.run(R_sample, feed_dict={X_notmissing: X_notmissing_data,
                                X_missing: X_missing_data, X_all: X_all_data,
                                real_sample_ID: notmissing_sample_ID,
                                fake_sample_ID: missing_sample_ID})
    sess.close()
    return R_sample