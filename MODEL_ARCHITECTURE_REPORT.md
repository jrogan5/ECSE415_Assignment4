# Model Architecture - RetinaNet for Traffic Sign Detection

## Architecture Description

The primary model architecture employed for this traffic sign detection task is RetinaNet with a ResNet-50 Feature Pyramid Network (FPN) backbone. RetinaNet is a one-stage object detection architecture that addresses the class imbalance problem inherent in dense object detectors through its innovative focal loss function. The model consists of three main components: a ResNet-50 convolutional backbone network that extracts rich feature representations from input images, a Feature Pyramid Network that constructs a multi-scale feature pyramid to detect objects at various scales, and two parallel task-specific subnetworks for classification and bounding box regression.

The ResNet-50 backbone utilizes deep residual learning with skip connections to enable effective training of deep networks, extracting hierarchical features at multiple spatial resolutions. The FPN component takes these multi-resolution feature maps from the backbone and constructs a feature pyramid with both bottom-up and top-down pathways, enriching semantic information at all scales through lateral connections. This architecture is particularly well-suited for traffic sign detection, as signs can appear at various scales depending on their distance from the camera.

The classification subnet predicts the probability of object presence for each of the 14 traffic sign classes at each spatial location, while the regression subnet outputs four coordinates representing the bounding box offset relative to anchor boxes at each location. The model employs nine anchor boxes at each pyramid level, with three different aspect ratios (1:2, 1:1, 2:1) and three scales, providing robust detection across diverse object shapes and sizes.

## Pre-trained Weights

The RetinaNet model was initialized with pre-trained weights from the COCO (Common Objects in Context) dataset, specifically using the RetinaNet_ResNet50_FPN_Weights.DEFAULT from PyTorch's torchvision model zoo. These weights were obtained from a model trained on the COCO 2017 dataset, which contains 118,000 training images across 80 object categories. Pre-training on COCO provides the model with robust feature extraction capabilities for general object detection tasks, which significantly accelerates convergence and improves performance when fine-tuning on the traffic sign detection task.

Only the final classification head was replaced to match the number of traffic sign classes (14 classes plus background), while all other layers including the backbone and FPN retained their pre-trained weights. This transfer learning approach leverages the rich visual representations learned from COCO's diverse object categories, which generalize well to traffic sign detection despite the domain difference.

## References

1. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal loss for dense object detection." In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988). [Original RetinaNet paper introducing focal loss and the architecture]

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). [ResNet-50 backbone architecture]

3. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). "Feature pyramid networks for object detection." In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125). [FPN architecture]

4. PyTorch torchvision.models.detection documentation: https://pytorch.org/vision/stable/models/retinanet.html

## Input Preprocessing

All input images undergo a standardized preprocessing pipeline before being fed into the model. The original dataset images, which have a native resolution of 416×416 pixels, are explicitly resized to 416×416 to ensure consistency, even though they are already at this resolution. This explicit resizing step guarantees uniform input dimensions regardless of any potential variations in the test set.

The pixel values, originally in the range [0, 255] with uint8 data type, are converted to float32 and normalized to the range [0, 1] by dividing by 255.0. This normalization is standard practice for neural networks and helps with training stability and convergence. The images are then converted from NumPy arrays to PyTorch tensors and transformed from the (Height, Width, Channels) format to the (Channels, Height, Width) format expected by PyTorch convolutional layers.

For the bounding box annotations, the YOLO-format normalized coordinates (center_x, center_y, width, height) are converted to absolute pixel coordinates in the (x1, y1, x2, y2) format, where (x1, y1) represents the top-left corner and (x2, y2) represents the bottom-right corner. Class labels are incremented by 1 to account for the background class (label 0), making the traffic sign classes range from 1 to 14 instead of 0 to 13.

## Data Augmentation Techniques

To improve model generalization and robustness, an extensive set of data augmentation techniques is applied during training. These augmentations are applied probabilistically to introduce controlled variability without distorting the semantic content of traffic signs.

**Horizontal Flipping:** With a 50% probability, images are horizontally flipped along with their corresponding bounding boxes. The bounding box coordinates are appropriately adjusted to maintain spatial consistency after flipping.

**Random Scaling:** With a 60% probability, images are randomly scaled by a factor between 0.75 and 1.25. After scaling, the image is placed on a 416×416 canvas, either cropping if the scaled image is larger or padding if smaller. Bounding box coordinates are scaled proportionally and clipped to remain within image boundaries.

**Brightness Jittering:** With a 50% probability, image brightness is randomly adjusted by multiplying pixel values by a factor between 0.7 and 1.3, simulating different lighting conditions encountered in real-world scenarios.

**Contrast Adjustment:** With a 30% probability, image contrast is modified by applying the transformation: `new_pixel = 128 + alpha * (old_pixel - 128)`, where alpha ranges from 0.7 to 1.3. This helps the model handle varying contrast levels in captured images.

**Color Saturation Adjustment:** With a 30% probability, the saturation channel in HSV color space is multiplied by a factor between 0.8 and 1.2, exposing the model to variations in color intensity that may occur due to different camera sensors or weather conditions.

These augmentations are applied only during training; validation and test sets undergo only the standard preprocessing without augmentation to provide consistent evaluation conditions.

## Training Hyperparameters

The model training process employs carefully selected hyperparameters optimized for the traffic sign detection task:

**Optimizer:** AdamW (Adam with weight decay) is used with an initial learning rate of 2×10⁻⁴ and weight decay of 1×10⁻⁴. AdamW provides adaptive learning rates for each parameter while incorporating L2 regularization through weight decay to prevent overfitting.

**Learning Rate Schedule:** A MultiStepLR scheduler reduces the learning rate by a factor of 0.1 at epochs 30 and 45, allowing the model to make larger updates early in training and finer adjustments in later epochs.

**Batch Size:** A batch size of 8 images per iteration is used, balancing memory constraints with training stability and gradient estimation quality.

**Number of Epochs:** The model is trained for 60 epochs, providing sufficient iterations for convergence while monitoring validation loss to detect potential overfitting.

**Image Size:** All images are resized to 416×416 pixels, matching the original dataset resolution and providing a good balance between computational efficiency and spatial detail preservation.

**Gradient Clipping:** Gradient norms are clipped to a maximum of 1.0 to prevent exploding gradients and improve training stability, particularly important given the multi-scale nature of the FPN architecture.

**Loss Function:** The model uses the combined focal loss for classification and smooth L1 loss for bounding box regression, as defined in the original RetinaNet paper. The focal loss addresses class imbalance by down-weighting well-classified examples and focusing training on hard negatives.

## Prediction Generation

During inference, the RetinaNet model processes an input image through the ResNet-50 backbone and FPN to generate predictions at multiple scales. At each spatial location in the feature pyramid, the classification subnet outputs class probabilities for each of the 14 traffic sign classes plus the background class, while the regression subnet outputs bounding box coordinate offsets.

The model generates predictions in the (x1, y1, x2, y2) format in pixel coordinates, where (x1, y1) represents the top-left corner and (x2, y2) represents the bottom-right corner of each predicted bounding box. Each prediction is accompanied by a confidence score representing the model's certainty in the detection, derived from the classification subnet's output after applying a sigmoid activation function.

For submission to Kaggle, these predictions are converted to YOLO format with normalized coordinates: (class_label, x_center, y_center, width, height), where all spatial values are normalized to the [0, 1] range by dividing by the image dimensions (416 pixels). The class labels are adjusted from the model's output range [1, 14] to the required range [0, 13] by subtracting 1.

## Post-processing Steps

Several critical post-processing steps are applied to the raw model predictions to improve detection quality and ensure compatibility with the evaluation format:

**Confidence Thresholding:** Only predictions with confidence scores above a threshold (typically 0.15) are retained. This threshold was determined through empirical evaluation on the validation set to balance precision and recall.

**Non-Maximum Suppression (NMS):** To eliminate duplicate detections of the same object, NMS is applied independently for each class with an IoU (Intersection over Union) threshold of 0.5. For each class, predictions are sorted by confidence score in descending order, and overlapping boxes with IoU exceeding the threshold are suppressed, retaining only the highest-confidence detection for each object.

**Background Class Filtering:** Predictions with class label 0 (background) are removed, as the task focuses solely on traffic sign detection rather than background classification.

**Coordinate Normalization:** Bounding box coordinates are normalized by dividing by the image size (416) and clipped to ensure all values remain within the [0, 1] range, preventing any invalid coordinates due to numerical imprecision.

**Score-based Ranking:** After NMS, remaining predictions for each image are sorted by confidence score in descending order. For Kaggle submission, the top predictions for each image are selected to fill the required prediction slots, with placeholder entries used for images with fewer detections than slots.

**Bounding Box Validation:** All bounding boxes are validated to ensure positive width and height values. Any degenerate boxes (width ≤ 0 or height ≤ 0) that may arise from edge cases in coordinate transformation are filtered out before final submission.

This comprehensive post-processing pipeline ensures that the final predictions are high-quality, non-redundant, and properly formatted for evaluation, significantly improving the model's performance on the Kaggle leaderboard compared to raw model outputs.
