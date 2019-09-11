import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize,imsave
import MyModel_simple as model
from multiprocessing import Process, Queue
import SimpleITK as si
import os

# trlist = np.load('trlist.npy')
# valist = np.load('valist.npy')
baselr = 0.001
batchsize = 13
volSize = 256
dataProdProcNum = 1
trainDataPath = '/media/ljq/DATA/data/trainSlice/'
modelPath = './model_XY_simple'
imgList = []
classNum = 2
# cnt = None
trainNum = 131
# reguScale = 0.0001
heights = np.load('height.npy')


def train(dataQueue):
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

    lrPL = tf.placeholder(
        tf.float32,
        name='lr')
    global_step = tf.Variable(0, trainable=False)
    myModel = model.vgg16seg(x, batchsize, classNum)
    getProbsOp = myModel.getProbsOp
    getLossOp = myModel.buildAndGetLossSimple(y)
    trainOp = myModel.buildOptmrAndGetTrainOp(lrPL, global_step)


    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        myModel.load_weights("../../../../model/vgg16_weights.npz", sess)
        ckpt = tf.train.get_checkpoint_state(modelPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('can not find checkpoint')
        sumLossTr = 0
        lossQTr = []
        sumLossVa = 0
        lossQVa = []
        count = 0
        imgMats = np.zeros(shape=[batchsize, volSize, volSize, 3], dtype=np.float32)
        vimgMats = np.zeros(shape=[batchsize, volSize, volSize, 3], dtype=np.float32)
        lblMats = np.zeros(shape=[batchsize, volSize, volSize, classNum], dtype=np.int32)
        vlblMats = np.zeros(shape=[batchsize, volSize, volSize, classNum], dtype=np.int32)
        bi = 0
        vbi = 0
        lr = baselr
        while count<2000001:
            count += 1
            [imgMat, lblMat, imgKey, flag] = dataQueue.get()
            if flag==1:
                imgMats[bi, :, :, :] = imgMat
                lblMats[bi, :, :, :] = lblMat
                bi += 1
                if bi==batchsize:
                    _, loss, prop, step = sess.run(
                        [trainOp, getLossOp,
                         getProbsOp, global_step],
                        feed_dict={
                            x: imgMats,
                            y: lblMats,
                            lrPL: lr
                        })
                    print('train: ' + imgKey + ": loss %f " % loss + ' liverResMean: %f' % np.mean(prop[:, :, :, 1]))
                    lossQTr.append(loss)
                    sumLossTr += loss
                    if len(lossQTr) == 101:
                        sumLossTr -= lossQTr[0]
                        lossQTr.pop(0)
                    meanLossTr = sumLossTr / len(lossQTr)
                    print('trainLoss: %f' % meanLossTr + ' lr:%f' % lr)

                    if step%100==0:
                        saver.save(sess, modelPath+'/model', global_step=global_step)
                        print('save step %d suceess'%step)
                    bi = 0
                    if loss < 0.1:
                        lr = baselr/2
            else:
                vimgMats[bi, :, :, :] = imgMat
                vlblMats[bi, :, :, :] = lblMat
                vbi += 1
                if vbi == batchsize:
                    loss, prop = sess.run(
                        [getLossOp,
                         getProbsOp],
                        feed_dict={
                            x: vimgMats,
                            y: vlblMats,
                            lrPL: lr
                        })
                    print('vali: ' + imgKey + ": loss %f" % loss)
                    lossQVa.append(loss)
                    sumLossVa += loss
                    if len(lossQVa)==101:
                        sumLossVa -= lossQVa[0]
                        lossQVa.pop(0)
                    meanLossVa = sumLossVa/len(lossQVa)
                    print('valiLoss: %f'%meanLossVa)
                    vbi = 0

def prepareDataThread(dataQueue):

    if os.path.isfile("./permutation_%d.npy"%trainNum)!=True:
        print("no permutation file!!! generate new!!!")
        np.save("permutation_%d.npy"%trainNum, np.random.permutation(131))
    permutation = np.load("./permutation_%d.npy"%trainNum)
    print(permutation)
    while True:
        ind0 = int(np.random.random()*trainNum)
        ind1 = permutation[ind0]
        hei = heights[ind1]
        ind2 = int(np.random.random()*hei)
        rdRedef = np.random.random()
        if os.path.isfile(trainDataPath + 'volumeSlice-%d-%d_def.nii' % (ind1, ind2)):
            if rdRedef < 0.5:
                print('use deform volumeSlice-%d-%d_def.nii' % (ind1, ind2))
                img = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d_def.nii' % (ind1, ind2))
            else:
                img = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d.nii' % (ind1, ind2))
        else:
            img = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d.nii' % (ind1, ind2))

        if os.path.isfile(trainDataPath + 'volumeSlice-%d-%d_segmentation_def.nii' % (ind1, ind2)):
            if rdRedef < 0.5:
                print('use deform volumeSlice-%d-%d_segmentation_def.nii' % (ind1, ind2))
                lbl = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d_segmentation_def.nii' % (ind1, ind2))
            else:
                lbl = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d_segmentation.nii' % (ind1, ind2))
        else:
            lbl = si.ReadImage(trainDataPath + 'volumeSlice-%d-%d_segmentation.nii' % (ind1, ind2))

        imgMat = (imresize(si.GetArrayFromImage(img), 0.5) / 255).astype(np.float32)
        lblMat = (si.GetArrayFromImage(lbl)[0::2, 0::2] > 0).astype(np.int32)
        imgMat = np.reshape(np.stack([imgMat, imgMat, imgMat], axis=2), [1, 256, 256, 3])
        lblMat2 = np.zeros([1, 256, 256, classNum], dtype=np.int32)
        for i in range(0, classNum):
            lblMat2[0, :, :, i] = (lblMat == i).astype(np.int32)
        if ind0 < trainNum-0:
            flag = 1
        else:
            flag = 0
        dataQueue.put(tuple((imgMat, lblMat2, 'volumeSlice-%d-%d.nii'%(ind1,ind2),flag)))

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
    train(dataQueue)
