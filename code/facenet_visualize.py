# python facenet_visualize.py

from keras.models import load_model
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + \
    'C:/Users/V002866/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'

# 1.Load Model :
model_path = "../data/models/facenet_keras.h5"
model = load_model(model_path,  compile=False)

plot_model(model, to_file="model.png", show_shapes=True)
