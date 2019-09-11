import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize,imsave
import MyModel_simple as model
from multiprocessing import Process, Queue
import SimpleITK as si
import os

# lr = 0.0001
batchsize = 1
volSize = 256
dataProdProcNum = 1
trainDataPath = '../../../../data/trainSlice/'
imgList = []
classNum = 2
cnt = None
# trainNum = 100
modelPath = './model_XY_simple'
heights = np.load('./height.npy')
resPath = '../res/sliceRes_XY/'
reguAlpha = 0.001


def test(dataQueue):

    x = tf.placeholder(
        tf.float32, [
            batchsize,
            volSize,
            volSize,
            3
        ],
        name='x-input')

    y = tf.placeholder(
        tf.int32, [
            batchsize,
            volSize,
            volSize,
            classNum
        ],
        name='y-input')



    global_step = tf.Variable(0, trainable=False)
    myModel = model.vgg16seg(x,batchsize,classNum)
    getProbsOp = myModel.getProbsOp
    getLossOp = myModel.buildAndGetLossSimple(y)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(modelPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('can not find checkpoint')
        while True:
            [imgMat, lblMat, imgKey] = dataQueue.get()

            loss, prop,step = sess.run(
                [getLossOp,getProbsOp,global_step],
                feed_dict={
                    x:imgMat,
                    y: lblMat,
                })

            res1 = prop[:, :, :, 1]

            resImg1 = si.GetImageFromArray(np.squeeze(res1).astype(np.float32))

            si.WriteImage(resImg1, resPath + imgKey + '_res1.nii')

            print(imgKey + ": loss %f" % loss)


def prepareDataThread(dataQueue):
    # permutation = np.load("./permutation_%d.npy" % trainNum)
    for k in range(0,201):
        i = k
        for j in range(0,heights[i]):
            imgKey = 'volumeSlice-%d-%d'%(i,j)
            img = si.ReadImage(trainDataPath+imgKey+'.nii')
            imgMat = (imresize(si.GetArrayFromImage(img),0.5)/255).astype(np.float32)
            imgMat = np.reshape(np.stack([imgMat, imgMat, imgMat], axis=2), [1, 256, 256, 3])
            lbl = si.ReadImage(trainDataPath+imgKey+'_segmentation.nii')
            lblMat = (si.GetArrayFromImage(lbl)[0::2,0::2] > 0).astype(np.int32)
            lblMat2 = np.zeros([1, 256, 256, classNum], dtype=np.int32)
            for ci in range(0, classNum):
                lblMat2[0, :, :, ci] = (lblMat == ci).astype(np.int32)
            dataQueue.put(tuple((imgMat, lblMat2, imgKey)))
    print('data gene end!!!')

if __name__ == '__main__':

    dataQueue = Queue(30)  # max 50 images in queue
    dataPreparation = [None] *1
    # print('params.params[ModelParams][nProc]: ', dataProdProcNum)
    for proc in range(0, 1):
        # dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
        dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue,))
        dataPreparation[proc].daemon = True
        dataPreparation[proc].start()
    # while True:
    #     tt=1
    test(dataQueue)
