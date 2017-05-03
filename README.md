# Boiling Milk Detector
A WIP Machine Learning app using Tensorflow which will be able to tell if a pot of milk is boiling.

## Requirements

1. Python 3.5
2. Tensorflow
3. OpenCV for python
4. numpy

## How to use

1. In the `preprocess` folder, run `ProcessVideos.py` > `WindowsToOneHot.py` > `Combine.py` in order.
2. Run `BasicCNN.py` or `TemporalCNN.py` or `TinyCNN.py`
3. After the model is trained, select the video to test in `ManualVerification.py` and run
4. For deep dream visualization run the py files in `DeepDreamTest` folder