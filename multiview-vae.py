import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

#prepare dataset
data = mnist.train.images
n_data = data.shape[0]
dim = data.shape[1]
anomaly_rate = 0.1
train_rate = 0.8
n_anomaly = int(n_data*anomaly_rate)
n_train = int(n_data*train_rate)
seed = 1

#split normal and anomaly data
anomaly_data = data[n_data-n_anomaly:,:]
normal_data = data[:n_data-n_anomaly,:]

#create multiview anomaly data
# [a1,a2]  =>   [a1,b1] 
# [b1,b2]  =>   [b1,a2]
#
a1,a2 = anomaly_data[:n_anomaly//2,:dim//2],anomaly_data[:n_anomaly//2,dim//2:]
b1,b2 = anomaly_data[n_anomaly//2:,:dim//2],anomaly_data[n_anomaly//2:,dim//2:]
anomaly_data1 = np.concatenate((a1,b2),axis = 1)
anomaly_data2 = np.concatenate((b1,a2),axis = 1)
anomaly_data = np.concatenate((anomaly_data1,anomaly_data2),axis = 0)
label = np.array([1]*(n_data-n_anomaly) + [0]*n_anomaly) # 1: normal 0: anomaly
normal_and_anomaly = np.concatenate((normal_data,anomaly_data),axis = 0)

#shuffle the data
random.seed(1)
random.shuffle(normal_and_anomaly)
random.seed(1)
random.shuffle(label)

# split traing and testing data
train_data = normal_and_anomaly[:n_train,:]
train_label = label[:n_train]
test_data = normal_and_anomaly[n_train:,:]
test_label = label[n_train:]

# training setup
X_dim = 28*28
y_dim = 2
mb_size = 64
z_dim = 10
h_dim = 128
c = 0
lr = 1e-4

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# define encoder
# =============================== Q(z|X) ======================================

X1 = tf.placeholder(tf.float32, shape=[None, X_dim//2])
X2 = tf.placeholder(tf.float32, shape=[None, X_dim//2])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W11 = tf.Variable(xavier_init([X_dim//2, h_dim]))
Q_b11 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W12 = tf.Variable(xavier_init([X_dim//2, h_dim]))
Q_b12 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu1 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu1 = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma1 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma1 = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_mu2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu2 = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma2 = tf.Variable(tf.zeros(shape=[z_dim]))

def Q(X1):

    h = tf.nn.relu(tf.matmul(X1, Q_W11) + Q_b11)

    z_mu1 = tf.matmul(h, Q_W2_mu1) + Q_b2_mu1
    z_logvar1 = tf.matmul(h, Q_W2_sigma1) + Q_b2_sigma1

    z_mu2 = tf.matmul(h, Q_W2_mu2) + Q_b2_mu2
    z_logvar2 = tf.matmul(h, Q_W2_sigma2) + Q_b2_sigma2

    return z_mu1, z_logvar1, z_mu2 , z_logvar2


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# define decoder
# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_Wh2 = tf.Variable(xavier_init([h_dim, h_dim]))
P_bh2 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim//2]))


def P1(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    h2 = tf.nn.relu(tf.matmul(h, P_Wh2) + P_bh2)

    logits = tf.matmul(h2, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


P2_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P2_Wh2 = tf.Variable(xavier_init([h_dim, h_dim]))
P2_bh2 = tf.Variable(tf.zeros(shape=[h_dim]))

P2_W2 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P2_b2 = tf.Variable(tf.zeros(shape=[X_dim//2]))


def P2(z):
    h = tf.nn.relu(tf.matmul(z, P2_W1) + P2_b1)
    h2 = tf.nn.relu(tf.matmul(h, P2_Wh2) + P2_bh2)

    logits = tf.matmul(h2, P2_W2) + P2_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

def get_batch(data,it,mb_size):
    n = len(data)
    n_of_batch = int(n//mb_size)
    it = it%n_of_batch
    batch = data[it*mb_size:(it+1)*mb_size,:]
    return batch

# =============================== TRAINING ====================================
z_mu1, z_logvar1,z_mu2, z_logvar2 = Q(X1)
z_sample1 = sample_z(z_mu1, z_logvar1)
z_sample2 = sample_z(z_mu2, z_logvar2)

_, logits1 = P1(z_sample1)
_, logits2 = P2(z_sample2)

# E[log P(X|z)]
recon_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits1, labels=X1), 1)
recon_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits2, labels=X2), 1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss1 = 0.5 * tf.reduce_sum(tf.exp(z_logvar1) + z_mu1**2 - 1. - z_logvar1, 1)
kl_loss2 = 0.5 * tf.reduce_sum(tf.exp(z_logvar2) + z_mu2**2 - 1. - z_logvar2, 1)

mean_recon_loss = tf.reduce_mean(recon_loss1 + recon_loss2)
score = recon_loss1 + recon_loss2
vae_loss = tf.reduce_mean(recon_loss1 + recon_loss2 + kl_loss1 + kl_loss2)
solver = tf.train.AdamOptimizer().minimize(vae_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_iter = 3000
for it in range(total_iter):
    X_mb = get_batch(train_data,it,mb_size)
    _, loss,r_loss= sess.run([solver, vae_loss ,mean_recon_loss], feed_dict={X1: X_mb[:,:X_dim//2],X2: X_mb[:,X_dim//2:]})

    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print('recon_loss: {:.4}'. format(r_loss))
        print()

#------------test----------------
from sklearn.metrics import roc_auc_score
score_,r_loss= sess.run([score ,mean_recon_loss], feed_dict={X1: test_data[:,:X_dim//2], X2: test_data[:,X_dim//2:]})
auc_score = roc_auc_score(test_label, -score_)
print("auc_score is", auc_score)