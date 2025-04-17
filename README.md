# Large Scale Visual Recognition Challenge 2012 highscore
My attempt at conquering LSVRC 2012's classification task leaderboard.

<p align="center">
  <img src="./preview.png" />
</p>

All models can be found in src/model.py as well as their interface - IClassifier.

* IClassifier
  * Interface for all classification models. Contains methods such as train_epoch, get_competition_error and forward
* CustomClassifier
  * My custom model inspired by one of VGG19 modificatins with residual connections
  * I used 3 residual connections spaced evenly throughout the model to prevent gradient vanishing and to achieve stable weight convergence
* Resnet50BasedClassifier
  * Base of the model is a pretrained Resnet50 on top of which linear layers were added
* VGG11BasedClassifier
  * Base of the model is a pretrained VGG11's convolutional network
  * Adaptive 7x7 avg pool and subsequent linear layers replaced by Adaptive 5x5 avg pool and linear layers of reduced size

Currently, my custom classification model achieves top5 accuracy of 55% on unseen and 75% (!!) on seen data which indicates serious overfitting. I added random flips to image preprocessing in hope of reducing overfitting. I will publish weights of my model here after another round of training.

Notes:
* Training lasted for 3 nights in a row
* Saving and loading model state (weights) is currently done manually
* Batch size and number of workers are optimised for my workstation (RTX 3060 12GB)
* ADAM with low learning rate shows the best results in a long run
* This projects is still WIP!!

Topics covered:
* Image classification
* Transfer learning
* Residual connections in CNNs
* Training on a big dataset (>100GB)
* Image augmentation
* Experimentation with hiperparameters with hardware limits in mind (VRAM, RAM, memory bus ...)
