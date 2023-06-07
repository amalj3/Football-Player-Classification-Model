# Leveraging Deep Learning for Player Position Recognition in Football
In this project, we harness the power of computer vision to tackle a fascinating task from the world of sports: classifying football players into specific groups based on their in-game positions. Using machine learning techniques and image processing, our goal is to identify and categorize players into one of the five classes: Quarterback (QB), Defensive Back (DB), Skill Positions (SKILL), Linebacker (LB), and Center (C). This is a crucial task in sports analytics and player performance assessment.

## Data Used
The dataset was collected and prepared from Roboflow, found [here](https://universe.roboflow.com/bronkscottema/football-players-zm06l/dataset/15). It consists of images of players in action, with annotations for player positions. The data is divided into training, validation, and test sets for model development and evaluation. Each instance in our dataset consists of an image and corresponding annotations. These annotations specify the bounding box for each player (given by 'xmin', 'ymin', 'xmax', 'ymax') and the player's class (one of QB, DB, SKILL, LB, C). See the _annotations.csv files for more info.

## Methodology
In this project, we employed several advanced machine learning and computer vision techniques to achieve our goal of classifying football players in an image.

Pretrained VGG16 Model
The core of our approach leverages the power of convolutional neural networks (CNNs). Specifically, we used the VGG16 model, a deep CNN developed by the Visual Geometry Group (VGG) at Oxford. VGG16 is known for its strong performance in image recognition tasks. Importantly, we used a version of VGG16 pre-trained on the ImageNet dataset, which contains millions of images across thousands of categories. This pre-training step allows us to leverage learned features from a wide array of images, providing a strong basis for our specific task.

Transfer Learning
We employed a technique known as transfer learning, where a model developed for a task is repurposed on a second related task. In our case, we took the base VGG16 model, froze the initial layers (thus retaining their pre-trained features), and appended our own layers onto the end. The added layers were trained on our specific task, enabling the model to fine-tune its understanding based on the nuances of our football player dataset. After the first round of training, we unfroze some of the last layers of the base model and continued training, allowing for more specialized feature extraction tailored to our dataset.

Data Generators
To manage our image data and annotations, we developed custom DataGenerators using TensorFlow's utilities. These DataGenerators serve up our data in a format suitable for training our model, performing vital tasks such as reading the image files, cropping the images based on the annotations, resizing the cropped images to a standard size, and converting the class labels into a form suitable for training the model (one-hot encoding).

Model Training and Evaluation
To train our model, we used the Adam optimizer, a widely-used optimization algorithm in the deep learning field due to its efficiency and low memory requirements. We also added early stopping in our training process. Early stopping monitors a specified metric (in our case, validation loss) and stops training when the metric stops improving, saving computational resources and helping prevent overfitting.

Overfitting Mitigation Strategies
Overfitting is a common challenge in machine learning, where a model performs well on the training data but poorly on unseen data. To combat overfitting in our project, we used several strategies:

1. Transfer Learning: One of the primary techniques we used to prevent overfitting was transfer learning. By using a pre-trained VGG16 model, we started with a model that already knew how to extract important features from images. This pre-training helps to avoid overfitting as the model has already generalized to a wide range of images.

2. Early Stopping: During the training process, we implemented an early stopping mechanism. This technique monitors a specified metric — in our case, validation loss — during the training process. If the metric stops improving for a set number of epochs (we set this to 5), the training process automatically stops, preventing the model from continuing to adapt too closely to the training data.

3. Dropout: Within our custom layers added to the VGG16 base model, we included a dropout layer. Dropout is a regularization technique that randomly sets a fraction of input neurons to 0 during training, which helps to prevent overfitting. In our model, we set this rate to 50%.

For model evaluation, we employed standard metrics for classification problems: accuracy (the proportion of correctly classified instances) and loss (cross-entropy loss in our case, a common measure for classification tasks).

## Code and Implementation
Our model's implementation involved integrating existing methodologies and adding new techniques to ensure robustness. The pre-existing work involves the use of TensorFlow/Keras API for model building and the VGG16 pre-trained model. The novel contributions include developing a custom data generator for efficient data feeding to the model, implementing a mechanism to handle the specific task of player position recognition, and utilizing several measures to avoid overfitting.

## Results
Our model demonstrates promising results in identifying player positions, underlining the potential of deep learning for such tasks. However, there is more work to be done. Currently we are getting around 0.75-0.80 accuracy on our predictions in the validation set, which can be improved. More extensive hyperparameter tuning may be necessary to increase the accuracy, and in the future we can also experiment with other preprocessing techniques such as grayscaling, using different pretrained models for our base layering, or even changing our custom layers on top. Additionally, our model seems to still be overfitting our training data, so there is room to improve on that field as well.