# python face_model.py -i "../data/image/face/han2.JPG" -o "../data/image/face/han.JPG"

import os
import time
import cv2
import argparse
import numpy as np
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from keras.utils.vis_utils import plot_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input images")
ap.add_argument("-o", "--output", required=True,
                help="path to output images")
args = vars(ap.parse_args())


# 1.Load Model :
model_path = "../data/models/facenet_keras.h5"
model = load_model(model_path,  compile=False)

# img = cv2.imread("../data/image/face/han2.JPG")
img = cv2.imread(args["image"])

compare_img = cv2.imread(args["output"])
# compare_img = cv2.imread("../data/image/face/han.JPG")
i = "han2.JPG"
o = "han.JPG"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_size = 160


def prewhiten(x):  # a.圖像白化（whitening）可用於對過度曝光或低曝光的圖片進行處理，處理的方式就是改變圖像的平均像素值為 0 ，改變圖像的方差為單位方差 1。

    if x.ndim == 4:

        axis = (1, 2, 3)

        size = x[0].size

    elif x.ndim == 3:

        axis = (0, 1, 2)

        size = x.size

    else:

        raise ValueError("Dimension should be 3 or 4")

    mean = np.mean(x, axis=axis, keepdims=True)

    std = np.std(x, axis=axis, keepdims=True)

    std_adj = np.maximum(std, 1.0/np.sqrt(size))

    y = (x-mean) / std_adj

    return y


def preProcess(img):  # d.圖像的預處理(即前述的幾項步驟)

    whitenImg = prewhiten(img)

    whitenImg = whitenImg[np.newaxis, :]

    return whitenImg


def l2_normalize(x, axis=-1, epsilon=1e-10):  # b.使用L1或L2標準化圖像，可強化其特徵。

    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                                           axis=axis, keepdims=True), epsilon))

    return output


img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
faceImg = preProcess(img)
print("faceImg shape : ", faceImg.shape)
print("faceImg :", faceImg)
embs = l2_normalize(np.concatenate(model.predict(faceImg)))
print('embs_valid : ', embs)
print('embs_valid_shape : ', embs.shape)

compare_img = cv2.resize(
    compare_img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
compare_faceImg = preProcess(compare_img)
print("faceImg shape : ", compare_faceImg)
print("faceImg :", compare_faceImg)
compare_embs = l2_normalize(np.concatenate(model.predict(compare_faceImg)))

distanceNum = distance.euclidean(embs, compare_embs)
print("The distance between {} with {} is {}".format(i, o, distanceNum))

# -------------------------------------------------------------


# 2.cascade :
# cascade_path = "../data/models/haarcascades/haarcascade_frontalface_default.xml"
# cascade = cv2.CascadeClassifier(cascade_path)
# faces = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]

# cv2.imwrite('Ben.jpg', img)


# img_path = "../data/image/"
# md_path = "../data/models/"

# #   Put all detect image under "validPicPath"
# validPicPath = "../data/image/face/"
# valid = "david2.jpg"
# compares_path = "../data/image/head/"
# compares = ["david.jpg", ]  # "david2.jpg", "david3.jpg", "david4.jpg",
# "david5.jpg", "tom_crop.jpg", "Elton_John.jpg", "Ben.jpg", "captain.jpg", "spyder.jpg"

# cascade_path = "../data/models/haarcascades/haarcascade_frontalface_default.xml"


# #   此版Facenet model需要的相片尺寸為160×160
# image_size = 160
# #   使用MS-Celeb-1M dataset pretrained好的Keras model
# model = load_model(model_path,  compile=False)


# 2.def preprocessing :


# def align_image(img, margin):  # c.偵測並取得臉孔area，接著再resize為模型要求的尺寸(下方例子並未作alignment)

#     cascade = cv2.CascadeClassifier(cascade_path)

#     faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
#     print('faces', faces)
#     if(len(faces) > 0):

#         (x, y, w, h) = faces[0]

#         face = img[y:y+h, x:x+w]

#         faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype="uint8")

#         faceMargin[margin:margin+h, margin:margin+w] = face

#         # 儲存抓到的臉
#         cv2.imwrite(str(time.time())+".jpg", faceMargin)
#         # resize output face-size
#         aligned = resize(faceMargin, (image_size, image_size), mode="reflect")

#         cv2.imwrite(str(time.time())+"_aligned.jpg", aligned)

#         return aligned

#     else:

#         return None

# def prewhiten(x):  # a.圖像白化（whitening）可用於對過度曝光或低曝光的圖片進行處理，處理的方式就是改變圖像的平均像素值為 0 ，改變圖像的方差為單位方差 1。

#     if x.ndim == 4:

#         axis = (1, 2, 3)

#         size = x[0].size

#     elif x.ndim == 3:

#         axis = (0, 1, 2)

#         size = x.size

#     else:

#         raise ValueError("Dimension should be 3 or 4")

#     mean = np.mean(x, axis=axis, keepdims=True)

#     std = np.std(x, axis=axis, keepdims=True)

#     std_adj = np.maximum(std, 1.0/np.sqrt(size))

#     y = (x-mean) / std_adj

#     return y


# def l2_normalize(x, axis=-1, epsilon=1e-10):  # b.使用L1或L2標準化圖像，可強化其特徵。

#     output = x / np.sqrt(np.maximum(np.sum(np.square(x),
#                                            axis=axis, keepdims=True), epsilon))

#     return output


# def preProcess(img):  # d.圖像的預處理(即前述的幾項步驟)

#     whitenImg = prewhiten(img)

#     whitenImg = whitenImg[np.newaxis, :]

#     return whitenImg


# imgValid = validPicPath + valid

# aligned = align_image(cv2.imread(imgValid), 6)
# print("aligned.shape :", aligned.shape)
# # cv2.imwrite("align.jpg", aligned)

# # 3.Check aligned can find or not :
# if(aligned is None):

#     print("Cannot find any face in image: {}".format(imgValid))

# else:

#     faceImg = preProcess(aligned)
#     # print('faceImg : ', faceImg)
#     # print('faceImg_shape : ', faceImg.shape)

#     # –> model會輸出128維度的臉孔特徵向量，接著我們將它們合併並進行L2正規化。Z

#     embs_valid = l2_normalize(np.concatenate(model.predict(faceImg)))
#     # print('embs_valid : ', embs_valid)
#     # print('embs_valid_shape : ', embs_valid.shape)
#     print(" ")
#     print("-----Compare with other face below-----")
#     print(" ")

# 4.Compare the face with others(face)
#     同上方的valid圖片，依序取得各圖片人臉的臉孔特徵向量，再與valid進行歐氏距離計算。

# for i in compares:
#     img_file = compares_path + i
#     aligned = align_image(cv2.imread(img_file), 6)
#     print('aligned : ', aligned)

#     if(aligned is not None):
#         faceImg = preProcess(aligned)
#         print('faceImg_shape : ', faceImg.shape)
#         print('faceImg : ', faceImg)
#         embs = l2_normalize(np.concatenate(model.predict(faceImg)))
#         print('Emb_shape : ', embs.shape)
#         print('Embeddings : ', embs)
#         distanceNum = distance.euclidean(embs_valid, embs)
#         print("The distance with {} is {}".format(i, distanceNum))
#         print("")
# ------------------------------
