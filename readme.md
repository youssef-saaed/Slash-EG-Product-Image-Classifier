# Product Category Image Classifier

This image classifier is trained to categorize images of various product categories. The classifier is built using transfer learning with the VGG16 pre-trained model architecture.

## Dataset

The dataset consists of images from various product categories, including Accessories, Beauty, Fashion, Games, Home, Nutrition, Sports, Artifacts, and Stationary. The images were collected from the Slash Application and manually labeled for training purposes.

### Dataset Distribution

- Accessories: 84 images
- Beauty: 30 images
- Fashion: 215 images
- Games: 13 images
- Home: 148 images
- Nutrition: 28 images
- Sports: 1 image
- Artifacts: 57 images
- Stationary: 270 images

The dataset is split into training and testing sets with a ratio of approximately 80:20.

## Model Architecture

The model architecture consists of a VGG16 base model followed by custom classification layers. The base model's weights are initialized with pre-trained weights on the ImageNet dataset. The custom classification layers include a global average pooling layer followed by fully connected layers with ReLU activation and a softmax output layer.

### Model Training

- **Batch Size:** 16
- **Epochs:** 15
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

The model is trained using data augmentation techniques such as rotation, width and height shifting, shearing, zooming, and horizontal flipping.

### Model Evaluation

After training, the model achieved an accuracy of approximately **83.6%** on the training data. When evaluated on the testing set, the model achieved an accuracy of approximately **76%**.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras

## License

This project is licensed under the [MIT License](LICENSE).
