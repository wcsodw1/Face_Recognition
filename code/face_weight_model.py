# python face_weight_model.py -i "../data/image/face/david.jpg" -o "../data/image/face/"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

# 1.Parse the argument : 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input images")
ap.add_argument("-o", "--output", required=True,
                help="path to output images")
args = vars(ap.parse_args())

# 2.Define the model : 
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('../data/models/vgg_face_weights.h5')
vgg_face_descriptor = keras.Model(inputs=model.layers[0].input
, outputs=model.layers[-2].output)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    from keras.applications.vgg16 import preprocess_input
    img = preprocess_input(img)
    return img

img1_representation = vgg_face_descriptor.predict(preprocess_image("../data/image/face/david.jpg"))[0,:]
print("img1_representation : ",img1_representation)
print("img1_shape : ",img1_representation.shape)

img2_representation = vgg_face_descriptor.predict(preprocess_image("../data/image/face/tom.jpg"))[0,:]
print("img2_representation : ",img2_representation)
print("img2_shape : ",img2_representation.shape)


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

epsilon = 0.40 #cosine similarity
#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
 
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    print("cosine_similarity : ",cosine_similarity)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    print("euclidean_distance : ",euclidean_distance)

    if (cosine_similarity < epsilon):
        print("verified - same person")
        return '1', euclidean_distance 
    else:
        print("Unverified !! they are not same person! Call the Security !!!")
        return '0', euclidean_distance

    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img('C:/Users/user/Desktop/Python/dat/image/face/%s' % (img1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img('C:/Users/user/Desktop/Python/data/image/face/%s' % (img2)))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    print("-----------------------------------------")

print("")
print("1.Not David below : ")
print("--------------------")

print("< Comapre with tom >")
verifyFace(args["image"], args["output"]+ "/tom.jpg")
print("---------------------------")

print("< Comapre with Ben >")
verifyFace(args["image"], args["output"]+ "/Ben.jpg")
print("---------------------------")

print("< Comapre with captain american >")
verifyFace(args["image"], args["output"]+ "/captain.jpg")
print("---------------------------")

print("< Comapre with Black-man >")
verifyFace(args["image"], args["output"]+ "/black.jpg")
print("---------------------------")

print("< Comapre with girl >")
verifyFace(args["image"], args["output"]+ "/girl.JPG")
print("---------------------------")

print("< Comapre with girl2 - han >")
verifyFace(args["image"], args["output"]+ "/han.JPG")
print("---------------------------")

print("< Comapre with girl3 - han2 >")
verifyFace(args["image"], args["output"]+ "/han2.JPG")
print("---------------------------")

# TEST.B - david compare with david : 
print("")
print("2.David below : ")
print("--------------------")

print("< david1_resize1 >")
verifyFace(args["image"], args["output"]+ "/david2.PNG")
print("---------------------------")

print("< david1_resize2 >")
verifyFace(args["image"], args["output"]+ "/david_crop.JPG")
print("---------------------------")

print("< david_without glasses >")
verifyFace(args["image"], args["output"]+ "/cebu.JPG")
print("---------------------------")

print("< david_hope >")
verifyFace(args["image"], args["output"]+ "/david_hope.JPG")
print("---------------------------")

print("< david_Youth_Fellowship >")
verifyFace(args["image"], args["output"]+ "/Youth.JPG")
print("---------------------------")

print("< david_head 80 degree >")
verifyFace(args["image"], args["output"]+ "/side_david.JPG")
print("---------------------------")

print("< david_head 120 degree >")
verifyFace(args["image"], args["output"]+ "/david_head.JPG")
print("---------------------------")

