# Baseline Model Architecture - YOLOv8 for Traffic Sign Detection

## Architecture Description

The baseline model employs YOLOv8-nano (YOLOv8n), the most compact variant of the YOLOv8 (You Only Look Once, version 8) object detection architecture. YOLOv8 is a state-of-the-art, anchor-free, one-stage detection model that predicts bounding boxes and class probabilities directly from full images in a single evaluation. The architecture consists of three main components: a CSPDarknet backbone with cross-stage partial connections for efficient feature extraction, a Path Aggregation Network (PAN) neck that combines features from different scales, and a decoupled detection head that separates classification and localization tasks.

The YOLOv8n variant is specifically designed for real-time applications and resource-constrained environments, featuring only 3.2 million parameters while maintaining competitive accuracy. The backbone employs a series of convolutional layers with batch normalization and SiLU activation functions, organized into C2f modules (CSP Bottleneck with 2 convolutions) that enable efficient gradient flow and feature reuse. The model processes images through multiple spatial resolutions, extracting features at three different scales (P3, P4, P5 corresponding to 1/8, 1/16, and 1/32 of input resolution) to detect objects of varying sizes effectively.

Unlike previous YOLO versions that relied on anchor boxes, YOLOv8 adopts an anchor-free approach, directly predicting bounding box centers and dimensions, which simplifies the detection pipeline and improves generalization across different object scales. The decoupled head architecture processes classification and box regression through separate convolutional branches, allowing each task to learn specialized features without interference from the other task.

## Pre-trained Weights

The baseline model utilizes pre-trained weights from the yolov8n.pt checkpoint, which was trained on the COCO 2017 dataset containing approximately 118,000 training images across 80 object categories. These weights provide the model with robust general-purpose object detection capabilities, including feature extraction skills for identifying edges, textures, shapes, and spatial relationships that are crucial for traffic sign detection.

The pre-trained model was fine-tuned on the traffic sign dataset using transfer learning, where all layers including the backbone and detection head were retained but adapted to the 14 traffic sign classes. This approach leverages the rich visual representations learned from COCO's diverse objects while specializing the model for traffic sign characteristics such as geometric shapes, distinctive colors, and text patterns.

## References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). "Ultralytics YOLOv8." https://github.com/ultralytics/ultralytics [Official YOLOv8 implementation and documentation]

2. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You only look once: Unified, real-time object detection." In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). [Original YOLO paper establishing the one-stage detection paradigm]

3. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An incremental improvement." arXiv preprint arXiv:1804.02767. [YOLOv3 introducing multi-scale predictions]

4. Ultralytics Documentation: https://docs.ultralytics.com/models/yolov8/

## Input Preprocessing

The YOLOv8 model employs an automated preprocessing pipeline managed by the Ultralytics framework. Input images are resized to 416×416 pixels using letterbox resizing, which maintains the original aspect ratio by adding gray padding to the shorter dimension. This approach prevents distortion of traffic signs that could affect detection accuracy, particularly important for signs containing text or specific geometric shapes.

Pixel values are automatically normalized to the [0, 1] range through division by 255.0, and images are converted from BGR (OpenCV default) to RGB color format to match the training data format. The Ultralytics framework handles the conversion to PyTorch tensors and the transformation to (Channels, Height, Width) format internally.

Bounding box annotations in YOLO format (class_id, x_center, y_center, width, height with normalized coordinates) are directly compatible with the model's expected input format and require no conversion. The framework automatically generates the required grid-cell-based targets during training, assigning each ground truth box to appropriate grid cells based on its center location.

## Data Augmentation Techniques

YOLOv8's training pipeline incorporates extensive built-in augmentation techniques applied automatically by the Ultralytics framework:

**Mosaic Augmentation:** Four training images are combined into a single image with a 2×2 grid layout, forcing the model to learn object detection across different contexts and scales. This technique is particularly effective for small object detection and reduces the need for large batch sizes.

**MixUp Augmentation:** Two images are blended together with their labels, creating composite training samples that encourage the model to learn more robust feature representations and reduce overfitting to specific visual patterns.

**Random Scaling:** Images are randomly scaled between 0.5× and 1.5× of their original size, exposing the model to traffic signs at various distances from the camera.

**Random Translation:** Images are randomly shifted horizontally and vertically by up to 10% of the image dimensions, teaching the model to detect signs regardless of their position in the frame.

**Random Rotation:** Small rotations up to ±10 degrees are applied to account for camera tilt and road inclines.

**HSV Color Augmentation:** Hue, saturation, and value channels are randomly adjusted to simulate different lighting conditions, weather effects, and camera color responses.

**Random Horizontal Flip:** Images are flipped horizontally with 50% probability, doubling the effective dataset size and improving model generalization to different viewing angles.

These augmentations are automatically managed by the Ultralytics framework and can be controlled through hyperparameters in the training configuration.

## Training Hyperparameters

The YOLOv8 baseline model was trained with the following hyperparameters:

**Optimizer:** Stochastic Gradient Descent (SGD) with momentum of 0.937 is used by default in YOLOv8, providing stable convergence for object detection tasks.

**Initial Learning Rate:** lr0 = 1×10⁻³, which controls the step size of parameter updates during training.

**Learning Rate Schedule:** Cosine annealing learning rate scheduler reduces the learning rate from the initial value to near zero following a cosine curve, allowing aggressive learning early in training and fine-tuning in later epochs.

**Batch Size:** 16 images per batch, providing a good balance between training speed and gradient estimation quality.

**Number of Epochs:** 30 epochs of training, sufficient for the model to converge on the relatively small traffic sign dataset.

**Image Size:** 416×416 pixels, matching the native dataset resolution.

**Warmup:** 3 epochs of warmup with linearly increasing learning rate from 0.1× to 1.0× of the initial learning rate, stabilizing training in the early stages.

**Weight Decay:** 5×10⁻⁴ for regularization, preventing overfitting by penalizing large weight values.

**Close Mosaic:** Mosaic augmentation is disabled for the final 10 epochs to allow the model to adapt to natural, non-augmented images before evaluation.

**Device:** The model automatically detects and utilizes available GPU (CUDA device 0) for accelerated training, falling back to CPU if no GPU is detected.

## Prediction Generation

During inference, YOLOv8 processes the entire image through the backbone and neck networks, producing predictions at three different scales corresponding to the P3, P4, and P5 feature pyramid levels. At each spatial location in these feature maps, the model directly predicts bounding box coordinates (center x, center y, width, height) and class probabilities for all 14 traffic sign classes simultaneously.

The bounding box predictions are generated in normalized coordinates relative to the image dimensions, with the center coordinates representing the box center position and width/height representing the box dimensions. These predictions are decoded from the network's raw outputs by applying sigmoid activation to normalize coordinates to the [0, 1] range and exponential transformation to the width and height predictions.

Class predictions are generated through the classification head, which outputs logits for each of the 14 classes. These logits are passed through a softmax function to produce probability distributions, with the highest probability indicating the predicted class. Each prediction is associated with a confidence score (objectness × class probability) representing the model's certainty in both the presence of an object and its class assignment.

The Ultralytics framework automatically handles the conversion between different coordinate formats, internally maintaining predictions in (x, y, w, h) normalized format and converting to other formats (such as (x1, y1, x2, y2)) when needed for evaluation or visualization.

## Post-processing Steps

YOLOv8's prediction pipeline includes several built-in post-processing steps to refine raw model outputs:

**Confidence Filtering:** Predictions with confidence scores below a threshold (default 0.25, configurable) are filtered out. This threshold can be adjusted based on the desired balance between precision and recall for the specific application.

**Non-Maximum Suppression (NMS):** Redundant bounding boxes are eliminated through NMS with an IoU threshold of 0.45 by default. For each class, predictions are sorted by confidence, and overlapping boxes exceeding the IoU threshold are suppressed, keeping only the highest-confidence detection for each object.

**Class-wise NMS:** NMS is applied independently for each class, preventing the suppression of different object classes that may occupy the same spatial region (though less relevant for traffic signs, which rarely overlap).

**Multi-scale Predictions Fusion:** Predictions from the three different scale levels (P3, P4, P5) are combined, with NMS handling redundant detections across scales. This ensures that objects are detected at their optimal scale regardless of size.

**Coordinate Denormalization:** For visualization and evaluation, normalized coordinates are converted back to pixel coordinates by multiplying by the image dimensions.

**Batch Processing:** The Ultralytics framework efficiently handles batch processing during inference, automatically managing memory and computational resources for optimal throughput.

The post-processing pipeline is highly optimized in the Ultralytics framework, leveraging vectorized operations and GPU acceleration where available to maintain real-time performance even with complex NMS operations across multiple scales and classes.

## Comparison with Custom RetinaNet Model

The YOLOv8 baseline serves as a reference point for evaluating the custom RetinaNet implementation. While both models are one-stage detectors suitable for real-time applications, they differ in several key aspects: YOLOv8 uses an anchor-free approach versus RetinaNet's anchor-based method, YOLOv8 employs a coupled detection head versus RetinaNet's decoupled heads, and YOLOv8 uses standard cross-entropy loss versus RetinaNet's focal loss designed specifically for class imbalance handling. These architectural differences result in distinct performance characteristics, making their comparison valuable for understanding which design choices are most effective for traffic sign detection.
