# Enhancing Cat and Dog Classification with Data Augmentation and Transfer Learning


This experiment aims to classify cats and dogs using convolutional neural networks (CNN). The process draws inspiration from a popular blog post on deep learning, incorporating techniques like data augmentation and transfer learning to improve classification accuracy.
<img width="397" alt="image" src="https://github.com/user-attachments/assets/2d088d40-b1e1-488b-985b-790c463db92a">
<img width="286" alt="image" src="https://github.com/user-attachments/assets/1617600b-6649-4942-842f-22eb20aae61c">



# Objective

The objective of the experiment was to use deep learning to classify images of cats and dogs using TensorFlow. The main goal was to practice convolutional neural networks and explore techniques such as data augmentation and transfer learning to improve model performance and generalization capabilities.


# Methodology

Data Preparation: The dataset consists of cat and dog images sourced from an online database, with about 11,000 images in total. The images were split into training and validation datasets, with 2000 training images and 1000 validation images for each class. Preprocessing involved resizing all images to 150x150 pixels and normalizing pixel values to fall between 0 and 1 to aid in model convergence.

Model 1: The initial model used a Sequential architecture with several convolutional and pooling layers, followed by dense layers for classification. The architecture consisted of three convolutional layers, each followed by a ReLU activation function and max-pooling, and two fully connected layers at the end. The model used a sigmoid activation function for binary classification. Data augmentation was not applied here, and the model achieved moderate accuracy, with training accuracy significantly higher than validation accuracy, indicating overfitting. The validation accuracy fluctuated around 48% to 56%, and the loss ranged between 0.68 and 0.69, showing the model's limited ability to generalize.

<img width="343" alt="image" src="https://github.com/user-attachments/assets/032a3b8d-6d84-4ae9-8d20-814223406c7b">
<img width="425" alt="image" src="https://github.com/user-attachments/assets/b6ec2e87-62e3-4970-b7dd-3e008c271d43">


Model 2: Data augmentation was introduced using random transformations such as rotation (up to 30 degrees), width and height shift (up to 10%), zoom (up to 20%), and horizontal flip. The goal was to increase the diversity of the training data and improve the model's ability to generalize to new data. The same Sequential model architecture from Model 1 was used. The introduction of data augmentation led to a noticeable improvement in the model's validation accuracy, reaching around 82%. The data augmentation helped reduce overfitting by making the model more robust to variations in the training images.

<img width="425" alt="image" src="https://github.com/user-attachments/assets/72f1e07d-e9ec-4c2c-b33d-c91783426c73">
<img width="425" alt="image" src="https://github.com/user-attachments/assets/107debac-2153-4efd-bd64-e93b81ab58b4">


Model 3: Transfer learning was implemented using ResNet50 as the base model, which had been pre-trained on the ImageNet dataset. The pre-trained layers were frozen to retain the learned features, and new dense layers were added for classification. The custom top layers included a global average pooling layer, a dense layer with 256 units, and a final output layer with a sigmoid activation function. Despite using transfer learning, the performance did not significantly improve, with validation accuracy fluctuating between 48% and 56%, and the loss remaining in the range of 0.68 to 0.69. This suggested that the pre-trained features might not have been fully suitable for this specific classification task without additional fine-tuning.

<img width="425" alt="image" src="https://github.com/user-attachments/assets/cb675c1c-8f29-4f98-9219-16011b7e850c">
<img width="425" alt="image" src="https://github.com/user-attachments/assets/6982d58c-6180-4fc3-8f36-a91507504094">


Model 4: To further refine the model, custom callbacks were added to stop training once 95% accuracy was achieved on the training set, preventing overfitting. The optimizer was changed to Adam to enhance performance by adapting the learning rate during training. Additionally, a dropout layer with a rate of 0.5 was added to reduce overfitting by randomly dropping units during training. This model also used the ResNet50 base, with some of the deeper layers unfrozen to allow fine-tuning. The validation accuracy showed stability, and training and validation accuracies were close to each other without significant divergence. However, the maximum accuracy achieved was around 68%, indicating that there was still room for performance improvement.

<img width="425" alt="image" src="https://github.com/user-attachments/assets/2008156a-948d-4e98-8e5c-5b8a67dd1096">
<img width="425" alt="image" src="https://github.com/user-attachments/assets/dfb4b5f8-8a37-4a86-8f34-6af9fb438511">


# Results

Each model's performance was evaluated using training and validation loss and accuracy plots:

Model 1: Achieved a validation accuracy fluctuating between 48% and 56%. The model showed significant overfitting, with validation loss remaining in the range of 0.68 to 0.69, indicating poor generalization.

Model 2: The use of data augmentation resulted in better generalization, with validation accuracy improving to around 82%. The gap between training and validation accuracy was reduced, indicating less overfitting compared to Model 1.

Model 3: Transfer learning with ResNet50 did not yield significant improvements, with validation accuracy fluctuating between 48% and 56%, and the validation loss staying between 0.68 and 0.69. The pre-trained features might not have been fully suitable for this specific classification task without additional fine-tuning.

Model 4: Fine-tuning the ResNet50 model and using the Adam optimizer led to more stable training, with validation accuracy reaching a maximum of around 68%. Training and validation accuracies were close, suggesting reduced overfitting, but the overall accuracy was still lower than desired.


# Conclusion

Data augmentation proved to be an effective method for improving the model's performance, as it increased the diversity of the training data and reduced overfitting. Transfer learning, while generally powerful, did not yield the expected benefits for this dataset, likely due to a mismatch between the pre-trained features and the target task. Fine-tuning the pre-trained model and using adaptive optimization techniques like Adam further improved performance, though the improvements were incremental. The experiment demonstrated that different methods should be tailored to the specific problem and dataset to achieve optimal performance. Future work could involve experimenting with different pre-trained models, more extensive fine-tuning, and the use of larger datasets to further enhance the model's generalization capabilities.

