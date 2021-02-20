# python 5_68_landmarks_image.py -i "../data/image/face/girl.JPG" -m "../data/models/68_landmarks/shape_predictor_68_face_landmarks.dat" -o "../data/output/68_landmarks/girl.JPG" -d "3"
# python 5_68_landmarks_image.py -i "../data/image/face/david_hope.jpg" -m "../data/models/5_landmarks/shape_predictor_5_face_landmarks.dat" -o "../data/output/5_landmarks/david_hope.jpg" -d "5"

import cv2
import dlib
import argparse

# 1.Parse the argument :
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--model", required=True, help="Path to the model")
ap.add_argument("-d", "--dot", required=True, type=int,
                help="size of dot(5 dot can be bigger)")
ap.add_argument("-o", "--output", required=True,
                help="Path to output the image")
args = vars(ap.parse_args())


# 2.set the model/image path :
# detector為臉孔偵測，model為landmarks偵測
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["model"])
# img = cv2.imread(face_path)
img = cv2.imread(args["image"])


# 3.Define Bondingbox informnation :
def renderFace(im, landmarks, color=(0, 255, 255), radius=args["dot"]):
    for p in landmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)


dets = detector(img, 1)  # detector function
print(" Faces-Detected-Number : {}".format(len(dets)))

for index, face in enumerate(dets):
    print('face {}; left {}; top {}; right {}; bottom {}'.format(
        index, face.left(), face.top(), face.right(), face.bottom()))
    shape = predictor(img, face)
    renderFace(img, shape)

# 4. Output result :
cv2.imwrite(args["output"], img)

# 5.show image :
# cv2.imshow("face-rendered", img)
# cv2.waitKey(0)


# Set path (can replace by argument)
#   Abs_Path = "/mnt/d/David_From_C/CAFFE/Caffe_Linux_Convert/Python/Face_Recognition/data"
#   face_path = Abs_Path + "/image/face/youth_fellowship.jpg"
#   predictor_path = Abs_Path + "/models/5_landmarks/shape_predictor_5_face_landmarks.dat"

# 5 & 68 Landmarks :
#   -i "../data/models/5_landmarks/shape_predictor_5_face_landmarks.dat"
#   -o "../data/output/5_landmarks/hope.jpg"
