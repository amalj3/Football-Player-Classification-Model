# Leveraging Deep Learning for Player Position Recognition in Football
In this project, we tackled the problem of player position recognition in American football using images. The aim was to develop a model capable of accurately recognizing players and distinguishing between different player positions, specifically Quarterback (QB), Defensive Back (DB), Skill Position (SKILL), Linebacker (LB), and Center (C).

## Data Used
The dataset was collected and prepared from Roboflow, found [here](https://universe.roboflow.com/bronkscottema/football-players-zm06l/dataset/15). It consists of images of players in action, with annotations for player positions. The data is divided into training, validation, and test sets for model development and evaluation.

## Methodology
Our approach involves the application of a Convolutional Neural Network (CNN) architecture. Specifically, we utilized a pre-trained VGG16 model, commonly used for image classification tasks, as a feature extractor. We added custom fully-connected layers on top of this to perform the final classification.

In order to combat overfitting and to ensure that our model generalizes well, we adopted several techniques including data augmentation, early stopping, and dropout layers.

## Code and Implementation
Our model's implementation involved integrating existing methodologies and adding new techniques to ensure robustness. The pre-existing work involves the use of TensorFlow/Keras API for model building and the VGG16 pre-trained model. The novel contributions include developing a custom data generator for efficient data feeding to the model, implementing a mechanism to handle the specific task of player position recognition, and utilizing several measures to avoid overfitting.

## Results
Our model demonstrates promising results in identifying player positions, underlining the potential of deep learning for such tasks. However, there is more work to be done. Currently we are getting around 0.75-0.80 accuracy on our predictions, which can be improved. More extensive hyperparameter tuning may be necessary to increase the accuracy, and in the future we can also experiment with other preprocessing techniques such as grayscaling, using different pretrained models for our base layering, or even changing our custom layers on top.