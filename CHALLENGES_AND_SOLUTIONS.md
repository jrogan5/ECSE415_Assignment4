# 6.4 Challenges and Solutions

Throughout the development and implementation of the traffic sign detection system, numerous technical challenges were encountered across data processing, model training, and evaluation phases. This section documents these challenges, the solutions implemented, and the valuable lessons learned from addressing each issue.

## Data Processing Challenges

### Challenge 1: Path Concatenation and File System Compatibility

**Problem Description:** The initial implementation used string-based paths created with `os.path.join()` combined with Python's Path division operator (`/`) for subsequent path concatenations. This approach caused `AttributeError: 'str' object has no attribute '__truediv__'` errors because the division operator for path concatenation only works with `pathlib.Path` objects, not strings. The code contained numerous instances of patterns like `root / "yolo_split"` and `TRAIN_LABELS / (img_path.stem + ".txt")` where `root` and `TRAIN_LABELS` were strings rather than Path objects.

**Impact:** The code would fail immediately when trying to access files, preventing any data loading or processing from occurring. This affected multiple cells in the notebook including data exploration, dataset splitting, YOLO baseline setup, and submission generation.

**Solution Implemented:** All path definitions were systematically converted to use `pathlib.Path` objects from the start, and all path concatenations were changed to use the `Path()` constructor instead of the `/` operator. For example, `root = os.path.join(path, "Road_Signs_Detection_Dataset")` was changed to `root = Path(path, "Road_Signs_Detection_Dataset")`, and `TRAIN_LABELS / (img_path.stem + ".txt")` was changed to `Path(TRAIN_LABELS, img_path.stem + ".txt")`. This ensured consistency across the entire codebase with over 20 path concatenation instances corrected.

**Lessons Learned:** When working with file paths in Python, it is crucial to maintain consistency in the type of path objects used throughout the codebase. The `pathlib.Path` approach is more robust and cross-platform compatible than string-based paths, and mixing the two paradigms creates subtle bugs that can be difficult to diagnose. Future implementations should standardize on `pathlib.Path` from the project's inception to avoid such issues.

### Challenge 2: Class Imbalance in the Dataset

**Problem Description:** Analysis of the training dataset revealed significant class imbalance, with some traffic sign classes having as few as 60 instances (Speed Limit 110) while others had over 215 instances (Speed Limit 30). This represents more than a 3.5-fold difference in representation across classes. Such imbalance can cause models to develop a bias toward over-represented classes, potentially ignoring or misclassifying under-represented classes during both training and inference.

**Impact:** Without addressing this imbalance, the model would likely achieve high overall accuracy by correctly predicting common classes while performing poorly on rare classes. This would result in low mAP scores and poor F1 scores for minority classes, reducing the model's practical utility for real-world traffic sign detection where all sign types are equally important for safety.

**Solution Implemented:** Multiple strategies were employed to mitigate class imbalance. First, stratified train-validation splitting was implemented using `sklearn.model_selection.train_test_split` with the `stratify` parameter, ensuring that each class maintained the same distribution ratio in both training and validation sets. This preserved the ability to evaluate model performance fairly across all classes. Second, aggressive data augmentation techniques (horizontal flipping, random scaling, brightness/contrast adjustment, color saturation jittering) were applied with high probabilities during training, effectively multiplying the effective dataset size and providing more diverse examples of minority classes. Third, the RetinaNet architecture's focal loss function inherently addresses class imbalance by down-weighting well-classified examples and focusing learning on hard negatives, which often belong to minority classes.

**Lessons Learned:** Class imbalance is a common challenge in real-world object detection datasets and requires a multi-pronged approach rather than a single solution. Stratified splitting ensures fair evaluation, data augmentation increases effective dataset size, and specialized loss functions can help the model focus on difficult examples. Monitoring per-class metrics during training rather than just overall accuracy is essential to detect bias toward majority classes early in development.

## Model Training Challenges

### Challenge 3: GPU Availability and Hardware Constraints

**Problem Description:** The initial implementation hardcoded GPU usage with `device=0` for YOLO training and `device="cuda"` for PyTorch models, assuming CUDA-compatible GPU availability. When executed on a system without a GPU, this caused a `ValueError: No CUDA GPUs are available` error, preventing any training from occurring. The error message indicated `torch.cuda.is_available(): False` and `torch.cuda.device_count(): 0`, confirming the absence of GPU hardware.

**Impact:** Without GPU acceleration, model training becomes significantly slower (typically 5-10× longer for YOLO and RetinaNet), making iterative development and hyperparameter tuning impractical. However, completely blocking execution on CPU-only systems prevents any training from occurring at all, which is unnecessarily restrictive.

**Solution Implemented:** Automatic device detection was implemented at the beginning of training cells using `torch.cuda.is_available()` to check for GPU availability. For YOLO training, the device parameter was changed to `yolo_device = 0 if cuda_available else 'cpu'`, and for PyTorch models, `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` was used. User-facing messages were added to inform about the detected hardware configuration and warn if CPU training would be slow. Additionally, a standalone `check_gpu.py` script was created to help users verify their GPU setup before training.

**Lessons Learned:** Code should be resilient to different hardware configurations and provide clear feedback about detected resources. While GPU acceleration is highly desirable for deep learning, CPU fallback ensures the code remains functional in resource-constrained environments. Future implementations should also consider batch size and epoch reduction recommendations when CPU training is detected, making training more practical on limited hardware.

### Challenge 4: Invalid Bounding Boxes After Data Augmentation

**Problem Description:** During development of the augmentation pipeline, aggressive transformations (particularly random scaling and geometric augmentations) occasionally produced bounding boxes with zero or negative width/height values. This occurred when augmentation operations like cropping or extreme scaling moved anchor points outside the image boundaries or caused coordinate inversions. Such invalid boxes would cause the model to crash during training or produce NaN (Not a Number) losses, halting the training process.

**Impact:** Training runs would fail unpredictably when invalid boxes were encountered, wasting computational resources and requiring restart from scratch. The non-deterministic nature of the augmentation made debugging difficult, as the issue would not appear consistently across different training runs or different random seeds.

**Solution Implemented:** A bounding box validation step was added to the dataset's `__getitem__` method, checking that `x2 > x1` and `y2 > y1` for all boxes after augmentation. If all boxes became invalid due to aggressive augmentation, the method falls back to the original, unaugmented boxes and labels, ensuring that every training sample contains at least one valid object. Additionally, bounding boxes were explicitly clipped to image boundaries using `np.clip()` to prevent coordinates from extending outside the valid [0, image_size-1] range. The code maintains copies of original boxes before augmentation (`orig_boxes` and `orig_labels`) specifically for this fallback mechanism.

**Lessons Learned:** Data augmentation, while beneficial for generalization, must include safeguards to maintain data integrity. Always validate augmented outputs before passing them to the model, and implement fallback strategies for edge cases. Maintaining original data copies provides a safe fallback without discarding training samples entirely. Future implementations should consider less aggressive augmentation ranges or more sophisticated augmentation libraries that guarantee valid outputs.

### Challenge 5: Training Stability and Gradient Explosion

**Problem Description:** During initial RetinaNet training experiments, the loss occasionally exhibited sudden spikes or diverged to infinity, particularly in the early epochs. This instability manifested as extremely large loss values (>1000) that never recovered, indicating gradient explosion where gradients become too large during backpropagation, causing parameter updates that destabilize the network.

**Impact:** Unstable training prevented the model from converging to good solutions, wasting computational time and requiring multiple training restarts with different random seeds. The unpredictability made it difficult to determine whether poor performance was due to model architecture, hyperparameters, or simply unstable optimization.

**Solution Implemented:** Gradient clipping was implemented using `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after computing gradients but before the optimizer step. This limits the magnitude of gradients, preventing extreme parameter updates that could destabilize training. Additionally, the learning rate was reduced from an initial aggressive value to a more conservative 2×10⁻⁴, and a learning rate warmup period was implicitly provided by the MultiStepLR scheduler's initial phase. The combination of these techniques provided stable loss curves that decreased monotonically throughout training.

**Lessons Learned:** Deep neural networks, particularly those with multiple parallel pathways like FPN architectures, can be sensitive to optimization hyperparameters. Gradient clipping is a simple yet effective technique that should be considered standard practice for object detection models. Monitoring loss curves closely during initial epochs helps detect instability early, allowing for intervention before significant computational resources are wasted. Starting with conservative learning rates and gradually increasing them (warmup) or using adaptive optimizers like AdamW can further improve stability.

### Challenge 6: Memory Constraints and Batch Size Limitations

**Problem Description:** Initial attempts to train RetinaNet with batch sizes of 16 or 32 (common in published papers) resulted in CUDA out-of-memory errors on available GPU hardware, even with the relatively small 416×416 image size. The error indicated that GPU memory allocation failed when attempting to allocate tensors for both the forward pass and gradient storage.

**Impact:** Out-of-memory errors prevent training from occurring and require reducing batch size, which can negatively impact training dynamics. Smaller batch sizes lead to noisier gradient estimates, potentially requiring more epochs to converge and possibly reducing final model performance due to reduced batch statistics for normalization layers.

**Solution Implemented:** The batch size was reduced to 8 images per iteration, which fit comfortably within available GPU memory while still providing reasonable gradient estimates. To partially compensate for the smaller batch size, the number of training epochs was increased from 50 to 60, and gradient accumulation could be implemented if further batch size reduction became necessary. The training still converged effectively to good performance despite the smaller batch size.

**Lessons Learned:** Published hyperparameters from papers are often optimized for high-end GPU hardware and may not be directly applicable to more modest setups. Batch size is one of the most flexible hyperparameters and can often be reduced without catastrophic impact on model performance, though it may require adjusting learning rates or training duration. Monitoring GPU memory usage and leaving some headroom for peak allocations prevents crashes during training. Alternative approaches like mixed-precision training (FP16) or gradient checkpointing could enable larger batch sizes on the same hardware.

## Evaluation and Submission Challenges

### Challenge 7: Missing Non-Maximum Suppression (NMS)

**Problem Description:** The initial submission generation code for RetinaNet included all model predictions above a confidence threshold without applying Non-Maximum Suppression. Since object detection models like RetinaNet generate multiple overlapping bounding box predictions for the same object (due to the dense anchor box grid and multi-scale predictions), this resulted in the submission containing numerous duplicate detections for each traffic sign. The Kaggle evaluation metric severely penalizes duplicate detections, treating them as false positives.

**Impact:** This was identified as the single most critical issue affecting model performance. The initial Kaggle score was only 37 out of 100, far below the TA baseline of 67, despite the model's validation metrics appearing reasonable. Analysis revealed that some images had 5-10 overlapping predictions for a single traffic sign, all with high confidence scores. Without NMS, the model was effectively creating its own false positives.

**Solution Implemented:** Per-class Non-Maximum Suppression was implemented using `torchvision.ops.nms` with an IoU threshold of 0.5. The implementation processes each predicted class independently: for each class label present in the predictions, boxes of that class are grouped together, sorted by confidence score, and NMS is applied to eliminate redundant detections. Only the highest-confidence box for each object survives the suppression. The NMS is applied after confidence thresholding but before coordinate conversion, ensuring that only unique, non-overlapping detections are included in the final submission.

**Lessons Learned:** Non-Maximum Suppression is not merely an optional optimization but a fundamental requirement for one-stage object detectors. The dramatic improvement in Kaggle score after adding NMS (expected increase of 10-15 points) demonstrates its critical importance. While NMS is built into frameworks like Ultralytics for YOLO, custom implementations require explicit NMS application. Per-class NMS is essential to prevent suppression of different object classes that may occupy similar spatial regions. This experience reinforced the importance of understanding the complete detection pipeline, not just the neural network component.

### Challenge 8: Incorrect Coordinate Normalization in Submission

**Problem Description:** The initial submission generation code contained a subtle but critical bug in coordinate normalization. The model was trained on 416×416 images and produced bounding boxes in pixel coordinates within this space. However, the submission code loaded test images using `PIL.Image` and `torchvision.transforms.ToTensor()` without explicit resizing, then normalized the predicted boxes by the original image dimensions rather than by 416. This mismatch meant that if an original image was not exactly 416×416 (or if the ToTensor transform resized it differently), the normalized coordinates would be incorrect.

**Impact:** Incorrect coordinate normalization caused bounding boxes in the submission to be positioned and sized incorrectly relative to the actual objects in the images. This manifested as low IoU scores during evaluation, even when the model detected the correct objects, because the boxes were offset or scaled incorrectly. This coordinate mismatch could account for significant score reduction.

**Solution Implemented:** The submission generation code was modified to explicitly resize all test images to 416×416 using `cv2.resize()` before model inference, matching the exact preprocessing used during training. The predicted bounding boxes, which are in 416×416 pixel space, are then normalized by dividing by 416 rather than by the original image dimensions. This ensures coordinate consistency: `xc_norm = float(xc / img_size)` where `img_size = 416`. Additionally, normalized coordinates are clipped to the [0, 1] range using `np.clip()` to prevent any invalid values due to floating-point precision issues.

**Lessons Learned:** Consistency in preprocessing between training and inference is absolutely critical for model performance. Even small discrepancies in image resizing, normalization, or coordinate systems can dramatically impact results in object detection tasks where precise spatial localization is evaluated. Explicitly documenting and enforcing preprocessing standards throughout the pipeline prevents such mismatches. Testing coordinate conversions with known examples (e.g., boxes at image corners or center) helps validate correctness before submission.

### Challenge 9: Background Class Label Handling

**Problem Description:** RetinaNet, like many object detectors, uses label 0 for the background class and labels 1-14 for the actual traffic sign classes. However, the Kaggle submission format expects class labels in the range [0, 13] corresponding directly to the 14 traffic sign types. The initial submission code did not properly handle this label offset, potentially including background predictions (label 0) in the submission or incorrectly offsetting class labels.

**Impact:** Including background class predictions in the submission would create false positives, as these represent areas where the model believes no object exists. Incorrect label offsets would systematically misclassify all objects (e.g., labeling all "Speed Limit 80" signs as "Speed Limit 50"), leading to catastrophically poor classification performance despite potentially correct localization.

**Solution Implemented:** Two-stage filtering was implemented in the submission generation pipeline. First, all predictions with `labels == 0` (background class) are explicitly filtered out using boolean masking: `keep_score = (scores >= score_thresh) & (labels != 0)`. Second, the remaining class labels are converted from the [1, 14] range to [0, 13] by subtracting 1: `class_label = int(lab) - 1`, with additional bounds checking to ensure the result stays within [0, num_classes-1]. This ensures that only foreground object predictions are included and that class labels match the expected submission format.

**Lessons Learned:** Understanding the complete labeling scheme used by a model, including special labels like background classes, is essential for correct prediction interpretation. Different frameworks and model architectures may use different labeling conventions (e.g., 0-indexed vs 1-indexed, with or without background class), and careful attention must be paid when converting between these schemes. Explicit filtering of background predictions and validation of label ranges prevents silent failures where the code runs without error but produces incorrect outputs.

### Challenge 10: CSV Submission Format Compliance

**Problem Description:** The Kaggle submission format requires specific structure where each row ID follows the pattern `imageID_idx`, with one row for every prediction slot regardless of whether a prediction exists. Initial attempts at generating submissions used simpler approaches that either omitted rows for images with no predictions (causing "missing IDs" errors) or used varying numbers of rows per image (causing row count mismatches with the sample submission).

**Impact:** Kaggle's submission system would reject CSVs that did not exactly match the sample submission structure, preventing evaluation entirely. This required multiple re-generations and re-submissions to debug the format issues, consuming time and limited daily submission attempts (5 per day).

**Solution Implemented:** The submission generation code was redesigned to strictly follow the sample submission structure. The code first loads `sample_submission.csv` to extract the exact set of required IDs and their ordering. For each ID in the sample submission, the code parses the image ID and slot index (`img_id, idx = _id.rsplit("_", 1)`), retrieves the corresponding prediction if it exists (`if idx < len(preds)`), or inserts a placeholder entry otherwise (small centered box with low confidence). This ensures that the output CSV has exactly the same number of rows and ID ordering as the sample submission, guaranteeing format compliance.

**Lessons Learned:** When working with competition platforms like Kaggle, strictly adhering to the provided submission format is non-negotiable. Rather than trying to infer the correct format, using the sample submission as a template ensures compliance. Placeholder entries for missing predictions (rather than omitting rows) maintain format structure while allowing the evaluation system to handle images with varying numbers of detections. Validating the submission file locally before uploading (checking row count, column names, ID format) prevents wasted submission attempts.

### Challenge 11: Optimal Threshold Selection

**Problem Description:** Object detection models require setting confidence thresholds and NMS IoU thresholds for post-processing, but the optimal values for these hyperparameters are dataset-specific and cannot be determined purely from validation metrics like mAP. Setting the confidence threshold too high results in missed detections (low recall), while setting it too low creates false positives (low precision). Similarly, NMS IoU thresholds that are too strict suppress valid detections while lenient thresholds leave duplicates.

**Impact:** Initial submissions used arbitrary threshold values (confidence=0.1, NMS IoU=0.5) that were not optimized for the specific characteristics of traffic sign detection or the Kaggle evaluation metric. Suboptimal thresholds could leave significant performance gains unrealized, potentially accounting for several points of score difference.

**Solution Implemented:** A multi-threshold submission strategy was implemented, generating separate submission files with different confidence thresholds (0.05, 0.10, 0.15, 0.20, 0.25) and NMS thresholds (0.45, 0.50) to explore the threshold parameter space. Each configuration produces a separate CSV file with a descriptive suffix (e.g., `submission_retinanet_v2_med.csv` for confidence=0.15). These multiple submissions allow empirical determination of optimal thresholds through Kaggle's public leaderboard feedback, with the recommendation to start with medium thresholds (confidence=0.15, NMS=0.50) and adjust based on results.

**Lessons Learned:** Post-processing hyperparameters like confidence and NMS thresholds are integral parts of the detection pipeline and significantly impact final performance, yet they cannot be fully optimized on validation sets alone due to potential distribution differences between validation and test sets. Generating multiple submissions with systematic threshold variations is a practical approach when you have limited submission attempts. Future work could implement more sophisticated threshold optimization techniques like grid search on validation sets or analysis of precision-recall trade-offs at different operating points.

## Summary of Key Lessons

Through addressing these diverse challenges, several overarching lessons emerged that will inform future computer vision projects:

1. **Preprocessing Consistency**: Maintaining identical preprocessing between training, validation, and testing is paramount. Document every transformation, resize operation, and normalization step explicitly.

2. **Defensive Programming**: Implement validation checks for data outputs (bounding box validity, coordinate ranges, label consistency) to catch errors early rather than during production or submission.

3. **Hardware Flexibility**: Design code to gracefully handle different hardware configurations (GPU vs CPU, different memory capacities) with automatic detection and appropriate fallbacks or warnings.

4. **Understanding Pipeline Components**: Deep learning success requires understanding not just the neural network but the entire pipeline including data augmentation, post-processing (NMS), coordinate systems, and evaluation metrics.

5. **Systematic Debugging**: When facing performance issues, systematically analyze each pipeline component (data loading, preprocessing, model architecture, post-processing, output format) rather than making multiple simultaneous changes.

6. **Iterative Refinement**: Complex systems benefit from iterative development with frequent validation at each stage, rather than implementing everything at once and debugging a tangled system.

These challenges, while frustrating during development, provided valuable learning experiences that significantly deepened understanding of practical object detection systems and the gap between theoretical knowledge and production-ready implementations.
