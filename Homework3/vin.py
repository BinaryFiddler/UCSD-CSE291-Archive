
# coding: utf-8

# In[69]:


import tensorflow as tf
import utils
import numpy as np
import os


# In[71]:


# load training data and labels
data0 = utils.load_h5("ply_data_train0.h5")
data1 = utils.load_h5("ply_data_train1.h5")
data2 = utils.load_h5("ply_data_train2.h5")
data3 = utils.load_h5("ply_data_train3.h5")
data4 = utils.load_h5("ply_data_train4.h5")

# train_data = data0[0]
# print(np.shape(train_data))

# train_labels = data0[1]
# catagory_names = utils.get_category_names()
# print(np.shape(train_labels))


# In[72]:


# aggregate training data, training label
train_data = np.append(data0[0], data1[0], axis=0)
train_data = np.append(train_data, data2[0], axis=0)
train_data = np.append(train_data, data3[0], axis=0)
train_data = np.append(train_data, data4[0], axis=0)
print(np.shape(train_data))

train_labels = np.append(data0[1], data1[1], axis=0)
train_labels = np.append(train_labels, data2[1], axis=0)
train_labels = np.append(train_labels, data3[1], axis=0)
train_labels = np.append(train_labels, data4[1], axis=0)
print(np.shape(train_labels))


# In[73]:


# load test data
test0 = utils.load_h5("ply_data_test0.h5")
test1 = utils.load_h5("ply_data_test1.h5")
# print(np.shape(test1[0]))


# In[74]:


# aggregate test data, test label
test_data = np.append(test0[0], test1[0], axis=0)
test_labels = np.append(test0[1], test1[1], axis=0)
print(np.shape(test_data))
print(np.shape(test_labels))


# In[75]:


# one hot encode train_labels
train_labels_one_hot = []
for l in train_labels:
    one_hot = np.zeros(40, dtype=np.int)
    one_hot[l[0]] = 1
    train_labels_one_hot.append(one_hot)
train_labels_one_hot = np.array(train_labels_one_hot)
print(np.shape(train_labels_one_hot))
# train_labels = train_labels.reshape((1, -1))


# In[76]:


# one hot encode test_labels
test_labels_one_hot = []
for l in test_labels:
    one_hot = np.zeros(40, dtype=np.int)
    one_hot[l[0]] = 1
    test_labels_one_hot.append(one_hot)
test_labels_one_hot = np.array(test_labels_one_hot)
print(np.shape(test_labels_one_hot))


# In[77]:


cloud = tf.placeholder(tf.float32, [None, 2048, 3])
pt_cloud = tf.expand_dims(cloud, -1)
print(np.shape(pt_cloud))

# placeholder for one-hot labels
y = tf.placeholder(tf.float32, [None, 40])

# placeholder for labels
y_labels = tf.placeholder(tf.int64, [None])

# 1st mlp layer
layer_conv1 = tf.contrib.layers.conv2d(inputs=pt_cloud, num_outputs=64, kernel_size=[1, 3], padding="VALID", activation_fn=tf.nn.relu)
print(np.shape(layer_conv1))
layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1)

# 2nd mlp layer
layer_conv2 = tf.contrib.layers.conv2d(inputs=layer_conv1, num_outputs=64, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv2))
layer_conv2 = tf.contrib.layers.batch_norm(layer_conv2)


# 3rd mlp layer
layer_conv3 = tf.contrib.layers.conv2d(inputs=layer_conv2, num_outputs=64, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv3))
layer_conv3 = tf.contrib.layers.batch_norm(layer_conv3)


# 4th cnn
layer_conv4 = tf.contrib.layers.conv2d(inputs=layer_conv3, num_outputs=128, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv4))
layer_conv4 = tf.contrib.layers.batch_norm(layer_conv4)


# 5th cnn
layer_conv5 = tf.contrib.layers.conv2d(inputs=layer_conv4, num_outputs=1024, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv5))
layer_conv5 = tf.contrib.layers.batch_norm(layer_conv5)

# max pooling
max_pool = tf.contrib.layers.max_pool2d(inputs=layer_conv5, kernel_size=[2048, 1])

print(np.shape(max_pool))

# fnn1
layer_fnn1 = tf.contrib.layers.fully_connected(inputs=max_pool, num_outputs=512, activation_fn=tf.nn.relu)
print(np.shape(layer_fnn1))
layer_fnn1 = tf.contrib.layers.batch_norm(layer_fnn1)


# fnn2
layer_fnn2 = tf.contrib.layers.fully_connected(inputs=layer_fnn1, num_outputs=256, activation_fn=tf.nn.relu)
print(np.shape(layer_fnn2))
layer_fnn2 = tf.contrib.layers.batch_norm(layer_fnn2)

layer_fnn2 = tf.contrib.layers.dropout(inputs=layer_fnn2, keep_prob=0.7)

# fnn3
logits = tf.contrib.layers.fully_connected(inputs=layer_fnn2, num_outputs=40, activation_fn=tf.nn.relu)
logits = tf.squeeze(logits, [1, 2])
print(np.shape(logits))

# softmax
output = tf.nn.softmax(logits)
output_class = tf.argmax(output,axis=1)
print(np.shape(output))
print(np.shape(output_class))


# In[78]:
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
l_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20*32, 0.5, staircase=True)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))

# optimizer
optim = tf.train.AdamOptimizer(learning_rate=l_rate)
optimizer = optim.minimize(loss, global_step=global_step)


# In[ ]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# print('Learning rate: %f' % (sess.run(optim ._lr)))

num_iter = 200*32
batch_size = 32

for i in range(num_iter):
    idx = np.random.choice(9840, [batch_size], False)
    batch_img = train_data[idx][:]
    batch_y = train_labels_one_hot[idx][:]
#     batch_img, batch_y = data.train.next_batch(batch_size)
	# if i % 20 == 0:
	# 	l = 
    _, l, lr= sess.run([optimizer, loss, optim._lr], feed_dict = {cloud: batch_img , y: batch_y})
    if i % (32*20) == 0:
        print(l, lr)
        save_path = tf.train.Saver().save(sess, os.path.join("model/", "model.ckpt"))
        print("Model saved in file: %s" % save_path)

# In[68]:


# accuracy

# correct_labels = tf.equal(output_class, y_labels)
# accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))
# # compute accuracy on test data
# labels = np.array([label.argmax() for label in test_labels_one_hot[:100]])
# #print(np.shape(labels))
# accuracy = sess.run([accuracy],feed_dict = {cloud: test_data[:100], y: test_labels_one_hot[:100], y_labels: labels})
# print ("Accuracy" ,accuracy)

right_count = 0
i = 0
while i < len(test_data):
	j = min(i + 64, len(test_data))
	correct_labels = tf.equal(output_class, y_labels)
	accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))
	# compute accuracy on test data
	labels = np.array([label.argmax() for label in test_labels_one_hot[i:j]])
	#print(np.shape(labels))
	accuracy = sess.run([accuracy],feed_dict = {cloud: test_data[i:j], y: test_labels_one_hot[i:j], y_labels: labels})
	print ("Accuracy from " ,i, j, accuracy)
	right_count = right_count + accuracy[0] * (j - i)
	i += 64
final_accuracy = right_count / len(test_data)
print("Final Accuracy", final_accuracy)
# In[ ]:


# print(np.shape(test_labels_one_hot[:10]))
# labels = np.array([label.argmax() for label in test_labels_one_hot[:10]])
# print(np.shape(labels))
# print(np.shape(test_data[:10]))

