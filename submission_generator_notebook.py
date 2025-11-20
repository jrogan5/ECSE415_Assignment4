"""
Complete Kaggle Submission Generator for RetinaNet
Generates submission.csv in the correct format from a .pth model file

CSV Format:
ID,class_label,x_center,y_center,width,height
880_1,2,0.5207,0.5405,0.6347,0.5952

Where:
- ID: {image_id}_{detection_number} (e.g., 880_1, 880_2)
- class_label: 0-13 (0 is used for dummy detections when no objects found)
- x_center, y_center, width, height: YOLO format, normalized [0,1]
- Images with no detections get: {image_id}_1,0,0.5,0.5,0.0,0.0

Copy this entire cell into your Google Colab notebook and run it.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.ops import nms
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# ========================================
# CONFIGURATION - CHANGE THESE
# ========================================

# Path to your saved .pth model
MODEL_PATH = "model_retina_v2.pth"  # Change to your model filename

# Detection thresholds
SCORE_THRESH = 0.15  # Confidence threshold (try 0.10, 0.15, 0.20)
NMS_THRESH = 0.5     # NMS IoU threshold

# Training image size (must match what you used during training)
IMG_SIZE = 416

# Output filename
OUTPUT_CSV = "submission.csv"

print("=" * 70)
print("KAGGLE SUBMISSION GENERATOR")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Score threshold: {SCORE_THRESH}")
print(f"NMS threshold: {NMS_THRESH}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Output: {OUTPUT_CSV}")
print("=" * 70)

# ========================================
# STEP 1: Load Model
# ========================================

print("\n[1/4] Loading model architecture...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Initialize RetinaNet
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

# Modify classification head for traffic signs (14 classes + background)
cls_head = model.head.classification_head
in_channels = cls_head.cls_logits.in_channels
num_anchors = cls_head.num_anchors
new_num_classes = num_classes + 1  # 14 + 1 background = 15

cls_head.cls_logits = nn.Conv2d(
    in_channels,
    num_anchors * new_num_classes,
    kernel_size=3,
    stride=1,
    padding=1
)
cls_head.num_classes = new_num_classes

# Load trained weights
print(f"\n[2/4] Loading weights from {MODEL_PATH}...")
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Weights loaded successfully")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
    print("\nAvailable .pth files:")
    for f in Path('.').glob('*.pth'):
        print(f"  - {f}")
    raise

model.to(device)
model.eval()

# ========================================
# STEP 2: Generate Predictions
# ========================================

print(f"\n[3/4] Running inference on test images...")

# Get all test images
test_images = sorted(TEST_IMAGES.glob("*.jpg"))
print(f"Found {len(test_images)} test images")

# Store results
all_detections = []

with torch.no_grad():
    for idx, img_path in enumerate(test_images):
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")

        # Extract image ID (filename without extension)
        image_id = img_path.stem

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Warning: Failed to load {img_path.name}, adding dummy detection")
            all_detections.append({
                'ID': f"{image_id}_1",
                'class_label': 0,
                'x_center': 0.5,
                'y_center': 0.5,
                'width': 0.0,
                'height': 0.0
            })
            continue

        # Preprocess image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CRITICAL: Resize to training size (416x416)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor [0,1], shape (3, H, W)
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Run inference
        outputs = model(img_tensor)

        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()      # (x1, y1, x2, y2) format
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()    # [1, num_classes] range

        # Filter by confidence threshold
        keep_score = scores >= SCORE_THRESH
        boxes = boxes[keep_score]
        scores = scores[keep_score]
        labels = labels[keep_score]

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []

        if len(boxes) > 0:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_mask = labels == label
                label_boxes = boxes[label_mask]
                label_scores = scores[label_mask]

                # Apply NMS
                keep_nms = nms(
                    torch.from_numpy(label_boxes),
                    torch.from_numpy(label_scores),
                    NMS_THRESH
                )
                keep_nms = keep_nms.cpu().numpy()

                final_boxes.append(label_boxes[keep_nms])
                final_scores.append(label_scores[keep_nms])
                final_labels.append(np.full(len(keep_nms), label))

            if len(final_boxes) > 0:
                final_boxes = np.concatenate(final_boxes, axis=0)
                final_scores = np.concatenate(final_scores, axis=0)
                final_labels = np.concatenate(final_labels, axis=0)
            else:
                final_boxes = np.array([])

        # Convert boxes to YOLO format (x_center, y_center, width, height)
        if len(final_boxes) > 0:
            # Boxes are in (x1, y1, x2, y2) format on 416x416 image
            # Convert to normalized (x_center, y_center, width, height) in [0, 1]

            x1 = final_boxes[:, 0] / IMG_SIZE
            y1 = final_boxes[:, 1] / IMG_SIZE
            x2 = final_boxes[:, 2] / IMG_SIZE
            y2 = final_boxes[:, 3] / IMG_SIZE

            # Calculate center and dimensions
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            # Clip to [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)

            # Add detections for this image
            for i in range(len(final_boxes)):
                # Convert RetinaNet label [1, num_classes] to class_label [0, num_classes-1]
                class_label = int(final_labels[i]) - 1

                # Skip invalid class labels
                if class_label < 0 or class_label >= num_classes:
                    continue

                detection_num = len([d for d in all_detections if d['ID'].startswith(f"{image_id}_")]) + 1

                all_detections.append({
                    'ID': f"{image_id}_{detection_num}",
                    'class_label': class_label,
                    'x_center': float(x_center[i]),
                    'y_center': float(y_center[i]),
                    'width': float(width[i]),
                    'height': float(height[i])
                })
        else:
            # No detections - add dummy detection with class_label=0
            all_detections.append({
                'ID': f"{image_id}_1",
                'class_label': 0,
                'x_center': 0.5,
                'y_center': 0.5,
                'width': 0.0,
                'height': 0.0
            })

print(f"\n✓ Generated {len(all_detections)} total entries")

# ========================================
# STEP 3: Create Submission CSV
# ========================================

print(f"\n[4/4] Creating submission CSV: {OUTPUT_CSV}...")

if len(all_detections) == 0:
    print("❌ ERROR: No detections generated!")
    raise ValueError("No detections - check your model and threshold")

# Create DataFrame
df = pd.DataFrame(all_detections)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"✓ Saved {len(df)} entries to {OUTPUT_CSV}")

# ========================================
# STATISTICS
# ========================================

print("\n" + "=" * 70)
print("SUBMISSION STATISTICS")
print("=" * 70)

# Count images with actual detections vs dummy detections
num_images_with_detections = len(df[(df['class_label'] != 0) | (df['width'] != 0.0)].groupby(df['ID'].str.split('_').str[0]))
num_images_with_dummy = len(df[(df['class_label'] == 0) & (df['width'] == 0.0)])
total_real_detections = len(df[(df['class_label'] != 0) | (df['width'] != 0.0)])

print(f"Total entries: {len(df)}")
print(f"Images with real detections: {num_images_with_detections}")
print(f"Images with dummy detections: {num_images_with_dummy}")
print(f"Total real detections: {total_real_detections}")
print(f"Average detections per image: {total_real_detections / len(test_images):.2f}")

print("\nDetections per class:")
for class_id in range(num_classes):
    count = len(df[df['class_label'] == class_id])
    if count > 0 and not (class_id == 0 and df[df['class_label'] == 0]['width'].iloc[0] == 0.0):
        print(f"  Class {class_id:2d} ({class_names[class_id]:>15}): {count}")

# Show sample rows
print("\nFirst 10 rows of submission:")
print(df.head(10).to_string(index=False))

# Verify format
print("\n" + "=" * 70)
print("FORMAT VERIFICATION")
print("=" * 70)

# Check all required columns exist
required_cols = ['ID', 'class_label', 'x_center', 'y_center', 'width', 'height']
if list(df.columns) == required_cols:
    print("✓ Column names correct")
else:
    print(f"❌ ERROR: Columns should be {required_cols}")
    print(f"   Got: {list(df.columns)}")

# Check ID format
sample_ids = df['ID'].head(5).tolist()
print(f"✓ Sample IDs: {sample_ids}")

# Check value ranges
print(f"✓ class_label range: [{df['class_label'].min()}, {df['class_label'].max()}]")
print(f"✓ x_center range: [{df['x_center'].min():.4f}, {df['x_center'].max():.4f}]")
print(f"✓ y_center range: [{df['y_center'].min():.4f}, {df['y_center'].max():.4f}]")
print(f"✓ width range: [{df['width'].min():.4f}, {df['width'].max():.4f}]")
print(f"✓ height range: [{df['height'].min():.4f}, {df['height'].max():.4f}]")

# Check for invalid values
if df['x_center'].max() > 1 or df['x_center'].min() < 0:
    print("⚠️  WARNING: x_center values outside [0, 1] range")
if df['y_center'].max() > 1 or df['y_center'].min() < 0:
    print("⚠️  WARNING: y_center values outside [0, 1] range")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"\nSubmission file ready: {OUTPUT_CSV}")
print(f"\nTo submit to Kaggle, run:")
print(f"  !kaggle competitions submit -c ecse-415-object-recognition -f {OUTPUT_CSV} -m \"RetinaNet - thresh {SCORE_THRESH}\"")
print("=" * 70)
