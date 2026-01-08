import depthai as dai

pipeline = dai.Pipeline()

# RGB Kamera Setup
cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False) # WICHTIG: Erzeugt Planar-Format
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB) # WICHTIG: RGB statt BGR

# ImageManip um Verzerrungen zu vermeiden (Letterboxing)
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResizeThumbnail(128, 128) # Erhält das Seitenverhältnis mit Padding
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p) # 'p' steht für Planar

cam.preview.link(manip.inputImage)

# Neural Network Setup
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("palm_detection_sh6.blob")
manip.out.link(nn.input)