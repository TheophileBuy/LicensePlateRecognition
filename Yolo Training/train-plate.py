import numpy as np
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo-plate.cfg", 
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 0.9,
           "train": True,
           "annotation": "./data/AnnotationsXML/plates/",
           "dataset": "./data/Images/plates/"}

tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()

