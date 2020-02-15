import tensorflow as tf
import numpy as np
import imageUnits
import os
import glob


def conv2d(x, w, b, keep_prob=1., strides=[1, 1, 1, 1]):
    with tf.name_scope('conv'):
        conv = tf.nn.conv2d(input=x, filter=w, strides=strides, padding='SAME')
        conv_b = tf.nn.bias_add(conv, b)
        return tf.nn.dropout(conv_b, keep_prob=keep_prob)


def max_pool(x, size):
    with tf.name_scope('pool'):
        pool = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')
        return pool


def deconv(x, w, b, outshape, strides=[1, 2, 2, 1]):
    with tf.name_scope('deconv'):
        deconv = tf.nn.conv2d_transpose(x, w, outshape, strides=strides, padding='SAME')
        deconv_b = tf.nn.bias_add(deconv, b)
        return deconv_b


def weights(shape, stddev=0.1):
    shape = tf.cast(shape, tf.int32)
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias(shape, init=0.1):
    return tf.Variable(tf.constant(init, shape=shape))


def create_net(input, keep_prob, channels, nclass, filter_size, pool_size, layers, data_size=64):
    current = input
    input_shape = tf.shape(input)
    layer = {}
    Variables = []
    feature_num = [channels, 64, 128, 256, 512, 512]
    for i in range(layers):
        with tf.name_scope('down_' + str(i + 1)):
            w1 = weights([filter_size, filter_size, feature_num[i], feature_num[i + 1]], stddev=0.2)
            b1 = bias([feature_num[i + 1]], init=0.1)
            conv_b1 = conv2d(current, w1, b1, keep_prob, strides=[1, 1, 1, 1])
            relu1 = tf.nn.relu(conv_b1)

            w2 = weights([filter_size, filter_size, feature_num[i + 1], feature_num[i + 1]], stddev=0.2)
            b2 = bias([feature_num[i + 1]])
            conv_b2 = conv2d(relu1, w2, b2, keep_prob, strides=[1, 1, 1, 1])
            relu2 = tf.nn.relu(conv_b2)

            current = max_pool(relu2, size=pool_size)
            layer['layer_' + str(i + 1)] = current

            Variables.append(w1)
            Variables.append(b1)
            Variables.append(w2)
            Variables.append(b2)

    num = 4096
    with tf.name_scope('down_6'):
        w6 = weights([data_size / (2 ** layers), data_size / (2 ** layers), feature_num[layers], num], stddev=0.2)
        b6 = bias([num], init=0.1)
        conv_b6 = conv2d(current, w6, b6, keep_prob, strides=[1, 1, 1, 1])
        relu6 = tf.nn.relu(conv_b6)
        Variables.append(w6)
        Variables.append(b6)

    with tf.name_scope('down_7'):
        w7 = weights([1, 1, num, num], stddev=0.2)
        b7 = bias([num], init=0.1)
        conv_b7 = conv2d(relu6, w7, b7, keep_prob, strides=[1, 1, 1, 1])
        relu7 = tf.nn.relu(conv_b7)
        Variables.append(w7)
        Variables.append(b7)

    with tf.name_scope('down_8'):
        w8 = weights([1, 1, num, nclass], stddev=0.2)
        b8 = bias([nclass], init=0.1)
        conv_b8 = conv2d(relu7, w8, b8, keep_prob, strides=[1, 1, 1, 1])
        Variables.append(w8)
        Variables.append(b8)

    with tf.name_scope('deconv1'):
        deconv_shape1 = tf.shape(layer['layer_4'])
        outshape = tf.stack([deconv_shape1[0], deconv_shape1[1], deconv_shape1[2], nclass])
        wd1 = weights([4, 4, nclass, nclass], stddev=0.2)
        bd1 = bias([nclass], init=0.1)
        deconv_bd1 = deconv(conv_b8, wd1, bd1, strides=[1, 2, 2, 1], outshape=outshape)

        w_c = weights([1, 1, feature_num[4], nclass], stddev=0.2)
        b_c = bias([nclass], init=0.1)
        score_pool4 = conv2d(layer['layer_4'], w_c, b_c, strides=[1, 1, 1, 1], keep_prob=1.)

        add1 = tf.add(score_pool4, deconv_bd1)
        Variables.append(wd1)
        Variables.append(bd1)

    with tf.name_scope('deconv2'):
        deconv_shape2 = tf.shape(layer['layer_3'])
        outshape = tf.stack([deconv_shape2[0], deconv_shape2[1], deconv_shape2[2], nclass])
        wd2 = weights([4, 4, nclass, nclass], stddev=0.1)
        bd2 = bias([nclass])
        deconv_bd2 = deconv(add1, wd2, bd2, strides=[1, 2, 2, 1], outshape=outshape)

        w_c = weights([1, 1, feature_num[3], nclass], stddev=0.1)
        b_c = bias([nclass], init=0.1)
        score_pool3 = conv2d(layer['layer_3'], w_c, b_c, strides=[1, 1, 1, 1], keep_prob=1.)

        add2 = tf.add(score_pool3, deconv_bd2)
        Variables.append(wd2)
        Variables.append(bd2)

    with tf.name_scope('output'):
        deconv_shape3 = tf.stack([input_shape[0], input_shape[1], input_shape[2], nclass])
        wd3 = weights([16, 16, nclass, nclass], stddev=0.1)
        bd3 = bias([nclass], init=0.1)
        logits = deconv(add2, wd3, bd3, strides=[1, 8, 8, 1], outshape=deconv_shape3)
        Variables.append(wd3)
        Variables.append(bd3)

    return logits, Variables


class FC_net():
    def __init__(self, data_provider, batch_size, conv_size=3, pool_size=2, channels=3, nclass=2, save_path='train_out',
                 white_channel_weight=0.1, layers=5):
        self.data_provider = data_provider
        self.batch_size = batch_size
        self.save_path = save_path
        self.nclass = nclass
        self.channels = channels
        self.white_weight = white_channel_weight
        self.x = tf.placeholder(tf.float32, [None, 64, 64, channels])
        self.y = tf.placeholder(tf.float32, [None, 64, 64, nclass])
        self.keep_prob = tf.placeholder(tf.float32)
        self.logits, self.variables = create_net(self.x, self.keep_prob, channels, nclass, conv_size, pool_size, layers,
                                                 data_size=64)

        self.prediction = imageUnits.pixel_softmax(self.logits)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.prediction, 3), tf.argmax(self.y, 3)), tf.float32))

    def create_optimizer(self, global_step, learn_rate, loss_name, decay_steps=50, decay_rate=0.95):
        loss = self.compute_loss(self.logits, self.y, loss_name)
        with tf.name_scope('learning_rate'):
            self.learn_rate_node = tf.train.exponential_decay(learn_rate, global_step, decay_steps, decay_rate)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step)
        return optimizer, loss

    def compute_loss(self, logits, labels, loss_name):
        if loss_name == 'cross':
            print('Cross_entropy_loss')
            with tf.name_scope('cross_entropy_loss'):
                logits = tf.reshape(logits, [-1, self.nclass])
                labels = tf.reshape(labels, [-1, self.nclass])
                weigths = tf.constant([self.white_weight, 0.1], tf.float32, [self.nclass, 1], name='channel_weight')
                weigths_map = tf.matmul(labels, weigths)
                weigths_map = tf.reduce_sum(weigths_map, axis=1)
                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

                weighted_loss = tf.multiply(loss_map, weigths_map)
                # loss = tf.reduce_mean(weighted_loss)
                loss = tf.reduce_mean(loss_map)
        elif loss_name == 'dice':
            print('Dice_loss')
            with tf.name_scope('loss_dice'):
                eps = 1e-5
                prediction = imageUnits.pixel_softmax(logits)
                intersection = tf.reduce_sum(prediction * labels)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
                loss = -(2 * intersection / (union))
        elif loss_name == 'focal':
            print('focal_loss')
            with tf.name_scope('focal_loss'):
                gamma = 2
                alpha = 0.85
                prediction = imageUnits.pixel_softmax(logits)
                pt_1 = tf.where(tf.equal(labels, 1), prediction, tf.ones_like(prediction))
                pt_0 = tf.where(tf.equal(labels, 0), prediction, tf.zeros_like(prediction))
                pt_1 = tf.clip_by_value(pt_1, 1e-9, .999)
                pt_0 = tf.clip_by_value(pt_0, 1e-9, .999)
                loss = -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_sum(
                    (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
        else:
            loss = 0
            print('None loss')
        # L2
        # L2 = 0.0001 * sum([tf.nn.l2_loss(i) for i in self.variables])
        # loss += L2
        return loss

    def save_sess(self, sess, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        saver = tf.train.Saver()
        return saver.save(sess, os.path.join(self.save_path, name))

    def sess_restore(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    def test(self, sess, name):
        data = self.data_provider.open_data_image('data\\train\TCGA_CS_4941_19960909_14.tif')
        label = self.data_provider.open_label_image('data\\train\TCGA_CS_4941_19960909_14_mask.tif')
        predition, accuracy = sess.run((self.prediction, self.accuracy),
                                       feed_dict={self.x: data,
                                                  self.y: label,
                                                  self.keep_prob: 1.})
        imageUnits.create_save_img(predition, 'test/' + name + '_acc' + str(accuracy) + '.jpg')
        # np.savetxt('out_txt/predition_0.txt', predition[0, :, :, 0])
        # np.savetxt('out_txt/predition_1.txt', predition[0, :, :, 1])
        return

    def predite(self):
        init = tf.global_variables_initializer()
        with tf.name_scope('predition'):
            with tf.Session() as sess:
                sess.run(init)
                self.sess_restore(sess)
                sum_acc = 0
                for i in range(100):
                    batch_x, batch_y = self.data_provider.next_batch()
                    predition, logits, acc = sess.run((self.prediction, self.logits, self.accuracy),
                                                      feed_dict={self.x: batch_x,
                                                                 self.y: batch_y,
                                                                 self.keep_prob: 1.}
                                                      )
                    imageUnits.create_save_img(batch_y, 'output/mask/mask' + str(i) + '.jpg')
                    imageUnits.create_save_img(predition, 'output/predition/predition' + str(i) + '.jpg')
                    sum_acc += acc
                print(str(sum_acc / 100))
                # np.savetxt('out_txt/logits_0.txt', logits[0, :, :, 0])
                # np.savetxt('out_txt/logits_1.txt', logits[0, :, :, 1])
        return

    def output(self):
        test_data_path = 'data/test-images/*.tif'
        images = glob.glob(test_data_path)
        label = np.zeros((1, 64, 64, 2))
        init = tf.global_variables_initializer()
        with tf.name_scope('output'):
            with tf.Session() as sess:
                sess.run(init)
                self.sess_restore(sess)
                for i in range(len(images)):
                    path = images[i]
                    name = path[17:]
                    data = self.data_provider.open_data_image(path)

                    predition = sess.run(self.prediction,
                                         feed_dict={self.x: data,
                                                    self.y: label,
                                                    self.keep_prob: 1.}
                                         )
                    imageUnits.create_save_img(predition, 'output/output/' + name[0:-4] + '_mask.tif')
        return

    def trian(self, epochs, train_iters, keep_prob, learn_rate=0.2, restore=False, save_steps=100, loss_name='cross'):
        """
        train the net had created
        :param epochs: number of epochs
        :param train_iters: number of training every epoch
        :param keep_prob: dropout probability tensor
        :param learn_rate: the started learning rate
        :return:
        """
        sum_steps = tf.Variable(epochs * train_iters, name="global_step")
        self.optimizer, self.loss = self.create_optimizer(global_step=sum_steps, learn_rate=learn_rate,
                                                          loss_name=loss_name,
                                                          decay_steps=train_iters, decay_rate=0.95)
        init = tf.global_variables_initializer()
        sum_acc = 0
        best_acc = 0
        with tf.Session() as sess:
            sess.run(init)
            if restore:
                self.sess_restore(sess)
                print('restore model')
            for epoch in range(epochs):
                if self.data_provider.shuffle_data:
                    self.data_provider.shuffle()
                for iter in range(train_iters):
                    batch_x, batch_y = self.data_provider.next_batch()
                    _, loss, accuracy = sess.run((self.optimizer, self.loss, self.accuracy),
                                                 feed_dict={self.x: batch_x,
                                                            self.y: batch_y,
                                                            self.keep_prob: keep_prob})
                    sum_acc += accuracy

                    print('epoch ' + str(epoch) + ',iter' + str(iter) + ': loss:' + str(
                        loss / self.batch_size) + ', accuracy:' + str(accuracy))
                    if (iter + 1) % 10 == 0:
                        self.test(sess, name='epoch' + str(epoch) + 'iter' + str(iter))

                    if (epoch * train_iters + iter + 1) % save_steps == 0:
                        # if sum_acc > best_acc:
                        self.save_sess(sess, 'model_epoch' + str(epoch) + '_iter' + str(iter) + '.ckpt')
                        best_acc = sum_acc
                        print('save model')
                        sum_acc = 0
            self.save_sess(sess, 'last_model.ckpt')
        return
