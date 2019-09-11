import numpy as np
import tensorflow as tf
# import global_variable
import tflearn as tfl
# from tensorflow.contrib.slim.nets import resnet_v2
from dropBlock.dropblock import DropBlock

class vgg16seg:

    def __init__(self, imgs,batchsize,classNum):
        self.batchsize = batchsize
        self.classNum = classNum
        self.parameters = []
        self.showout = []
        self.imgs = imgs
        self.convlayers()
        # self.fc_layers()
        self.getProbsOp = tf.nn.softmax(self.conv9_4 )
        self.showout.append(self.getProbsOp)


    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, ksize,input_data, out_channel,trainable,restrain=True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [ksize, ksize, in_channel, out_channel], dtype=tf.float32,initializer=tfl.initializations.xavier_initializer(),trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], initializer=tf.constant_initializer(0.0),dtype=tf.float32,trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            if restrain==True:
                out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
            else:
                out = res
        self.parameters += [kernel, biases]
        return out

    def upconv(self,name, input_data, out_channel,trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [2, 2, out_channel, in_channel], dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=trainable)
            ###############################################################
            # batch_size = tf.shape(something_or_other)[0]
            # deconv_shape = tf.pack([batch_size, 40, 40, 32])
            # conv2d_transpose(..., output_shape=deconv_shape, ...)
            #########################################################################
            # batch_size = tf.shape(input_data)[0]
            # deconv_shape = tf.stack([batch_size,tf.shape(input_data)[1]*2,tf.shape(input_data)[2]*2,out_channel])
            conv_res = tf.nn.conv2d_transpose(input_data, kernel, output_shape = [self.batchsize,int(input_data.shape[1])*2,int(input_data.shape[2])*2,out_channel],strides = [1, 2, 2, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
        self.parameters += [kernel, biases]
        return out

    def conct(self,tensor1,tensor2):
        layerConc = tf.concat([tensor1, tensor2], 3)
        return layerConc

    def pontWiseAdd(self,tensor1,tensor2):
        ret = tf.layers.batch_normalization(tf.nn.relu(tensor1 + tensor2))
        return ret

    def convlayers(self):
        # zero-mean input
        #conv1
        self.conv1_1 = self.conv("conv1re_1",3,self.imgs,64,trainable=True) #1
        self.conv1_2 = self.conv("conv1_2",3,self.conv1_1,64,trainable=True)#1
        self.pool1 = self.maxpool("poolre1",self.conv1_2) #1/2

        #conv2
        self.conv2_1 = self.conv("conv2_1",3,self.pool1,128,trainable=True)#1/2
        self.conv2_2 = self.conv("convwe2_2",3,self.conv2_1,128,trainable=True)#1/2
        self.pool2 = self.maxpool("pool2",self.conv2_2)#1/4

        #conv3
        self.conv3_1 = self.conv("conv3_1",3,self.pool2,256,trainable=True)#1/4
        self.conv3_2 = self.conv("convrwe3_2",3,self.conv3_1,256,trainable=True)#1/4
        self.conv3_3 = self.conv("convrew3_3",3,self.conv3_2,256,trainable=True)#1/4
        self.pool3 = self.maxpool("poolre3",self.conv3_3)#1/8

        #conv4
        self.conv4_1 = self.conv("conv4_1",3,self.pool3,512,trainable=True)#1/8
        self.conv4_2 = self.conv("convrwe4_2",3,self.conv4_1,512,trainable=True)#1/8
        self.conv4_3 = self.conv("conv4rwe_3",3,self.conv4_2,512,trainable=True)#1/8
        self.pool4 = self.maxpool("pool4",self.conv4_3)#1/16

        #conv5
        self.conv5_1 = self.conv("conv5_1",3,self.pool4,512,trainable=True)#1/16
        self.conv5_2 = self.conv("convrwe5_2",3,self.conv5_1,512,trainable=True)#1/16
        self.conv5_3 = self.conv("conv5_3",3,self.conv5_2,512,trainable=True)#1/16
        # self.pool5 = self.maxpool("poorwel5",self.conv5_3)

        #upconv1
        self.upconv1 = self.upconv("upconv1", self.conct(self.conv5_3,self.pool4), 512,trainable=True)#1/8

        # conv6
        self.conv6_1 = self.conv("conv6_1", 3,self.upconv1, 512,trainable=True)#1/8
        self.conv6_2 = self.conv("conv6_2", 3,self.conv6_1, 512,trainable=True)#1/8
        self.conv6_3 = self.conv("conv6_3", 3,self.conv6_2, 512,trainable=True)#1/8

        # upconv2
        self.upconv2 = self.upconv("upconv2", self.conct(self.conv6_3,self.pool3), 256,trainable=True)#1/4

        # conv7
        self.conv7_1 = self.conv("conv7_1", 3,self.upconv2, 256,trainable=True)#1/4
        self.conv7_2 = self.conv("conv7_2", 3,self.conv7_1, 256,trainable=True)#1/4
        self.conv7_3 = self.conv("conv7_3", 3,self.conv7_2, 256,trainable=True)#1/4

        # upconv3
        self.upconv3 = self.upconv("upconv3", self.conct(self.conv7_3,self.pool2), 128,trainable=True)#1/2

        # conv8
        self.conv8_1 = self.conv("conv8_1", 3,self.upconv3, 128,trainable=True)#1/2
        self.conv8_2 = self.conv("conv8_2", 3,self.conv8_1, 128,trainable=True)#1/2

        # upconv4
        self.upconv4 = self.upconv("upconv4", self.conct(self.conv8_2, self.pool1), 64,trainable=True)  # 1

        # conv9
        self.conv9_1 = self.conv("conv9_1", 3,self.upconv4, 32,trainable=True)  # 1
        self.conv9_2 = self.conv("conv9_2", 3,self.conv9_1, 16,trainable=True)  # 1

        self.conv9_3 = self.conv("conv9_3", 3,self.conct(self.conv9_2, self.imgs), self.classNum,trainable=True)  #1
        self.conv9_4 = self.conv("conv9_4", 1, self.conv9_3, self.classNum, trainable=True,restrain=False)  # 1


    def buildAndGetLossSimple(self,label):
        self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
        self.loss = tf.reduce_mean(self.loss_map)
        return self.loss

    def buildAndGetLoss(self,label):
        self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
        self.loss_mid = tf.reduce_mean(self.loss_map,[1,2])

        num_examples = tf.shape(label)[0]
        n_selected = num_examples / 2  # 根据个人喜好只选最难的一半

        n_selected = tf.cast(n_selected, tf.int32)
        vals, _ = tf.nn.top_k(self.loss_mid, k=n_selected)
        th = vals[-1]
        selected_mask = self.loss_mid >= th

        loss_weight = tf.cast(selected_mask, tf.float32)
        self.loss = tf.reduce_sum(self.loss_mid * loss_weight) / tf.reduce_sum(loss_weight)

        return self.loss

    def buildAndGetLossTest(self,label):
        self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
        self.loss = tf.reduce_mean(self.loss_map)
        return self.loss

    # self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
    #                                                       self.decay, staircase=True))

    def buildOptmrAndGetTrainOp(self,lr,global_step):
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=global_step)
        return self.train_op


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [26,27,28,29,30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")