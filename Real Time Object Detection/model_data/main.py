from Detector import *
import os

def main():
    videoPath =r"C:\Users\mouni\Downloads\WhatsApp Video 2025-02-18 at 20.54.35_4d6741e1.mp4"
    #videoPath = 0
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data","coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()