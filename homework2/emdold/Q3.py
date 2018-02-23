
# coding: utf-8

# In[2]:


import math
import numpy as np
import tensorflow as tf


# In[3]:


# sample n points from circle with radius r
def sample_from_circle(r, n):
    return [[math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r, 0] for x in range(0,n)]


# In[4]:


point_set = []
n = 1
for r in range(1, 11):
    pts = sample_from_circle(r, n)
    point_set.append(pts)
point_set = np.array(point_set)


# In[5]:


print(np.shape(point_set))


# In[6]:


total_points = len(point_set)

d = tf.placeholder(tf.float32, [n, 3])

w = tf.get_variable("w2", [3, n], initializer = tf.random_normal_initializer(stddev = 5))


# In[7]:


import tf_emddistance


# In[8]:


print(d.shape)

d1 = tf.reshape(d, [1, n, 3])
w1 = tf.reshape(tf.transpose(w), [1, n, 3])
# define optimizer
l, idx1, idx2 = tf_emddistance.emd_distance(w1, d1)
loss = tf.reduce_sum(l)
optim = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


# In[ ]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_iter = 1000
for i in range(num_iter):
    idx = np.random.choice(10, [2], False)
    _, l = sess.run([optim, loss], feed_dict = {d: point_set[idx][0][:]})
    if i % 10 == 0:
        print(l)


# In[47]:


idx = np.random.choice(10, [1], False)
print(idx)
point_set[idx][0][:]
print(np.shape(point_set[idx][:]))

