import numpy as np
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo-character.cfg", 
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 0.9,
           "train": True,
           "annotation": "./data/AnnotationsXML/characters/",
           "dataset": "./data/Images/characters/"}

tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()

