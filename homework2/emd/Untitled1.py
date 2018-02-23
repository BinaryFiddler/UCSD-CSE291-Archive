
# coding: utf-8

# In[ ]:


import math
import numpy as np
import tensorflow as tf

# sample n points from circle with radius r
def sample_from_circle(r, n):
    return [[math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r, 0] for x in range(0,n)]

point_set = []
n = 500
m = 10
for r in range(1, 11):
    pts = sample_from_circle(r, n)
    point_set.append(pts)
point_set = np.array(point_set)

print(np.shape(point_set))

# total_points = len(point_set)

d = tf.placeholder(tf.float32, [m, n, 3])

w = tf.get_variable("w1", [3, n], initializer = tf.random_normal_initializer(stddev = 0.1))

import tf_emddistance

x = tf.transpose(w)
xs = []
for i in range(m):
    xs.append(x)

w1 = tf.stack(xs)    
print((w1.shape))
# define optimizer
l, idx1, idx2 = tf_emddistance.emd_distance(d, w1)
loss = tf.reduce_sum(l)
optim = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_iter = 10
for i in range(num_iter):
    _, l = sess.run([optim, loss], feed_dict = {d: point_set})
    print(l)
    if i % 10 == 0:
        print(l)

