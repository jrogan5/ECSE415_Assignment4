# ========================================
# Complete Kaggle Submission Generator
# Copy this entire cell into your notebook
# ========================================

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

# Path to your saved model
MODEL_PATH = "model_retina_v2.pth"  # Change this to your .pth file

# Submission settings
SCORE_THRESH = 0.15  # Confidence threshold (try 0.10, 0.15, 0.20)
NMS_THRESH = 0.5     # NMS IoU threshold
IMG_SIZE = 416       # Must match training size

# Kaggle submission message
SUBMISSION_MESSAGE = "RetinaNet v2 - val loss 0.17 - thresh 0.15"

# Output CSV filename
OUTPUT_CSV = "submission_retinanet_v2.csv"

print("=" * 70)
print("KAGGLE SUBMISSION GENERATOR")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Score threshold: {SCORE_THRESH}")
print(f"NMS threshold: {NMS_THRESH}")
print(f"Output CSV: {OUTPUT_CSV}")
print("=" * 70)

# ========================================
# 1. Load Model Architecture
# ========================================

print("\n[1/5] Loading model architecture...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model with same architecture as training
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

# Modify classification head for your num_classes
cls_head = model.head.classification_head
in_channels = cls_head.cls_logits.in_channels
num_anchors = cls_head.num_anchors
new_num_classes = num_classes + 1  # Should be 15 (14 classes + background)

cls_head.cls_logits = nn.Conv2d(
    in_channels,
    num_anchors * new_num_classes,
    kernel_size=3,
    stride=1,
    padding=1
)
cls_head.num_classes = new_num_classes

# ========================================
# 2. Load Trained Weights
# ========================================

print(f"\n[2/5] Loading weights from {MODEL_PATH}...")

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Weights loaded successfully!")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
    print("\nAvailable .pth files in current directory:")
    for f in Path('.').glob('*.pth'):
        print(f"  - {f}")
    raise
except Exception as e:
    print(f"❌ ERROR loading weights: {e}")
    raise

model.to(device)
model.eval()

print("✓ Model ready for inference!")

# ========================================
# 3. Generate Predictions on Test Set
# ========================================

print(f"\n[3/5] Generating predictions on test set...")

# Get test images
test_images = sorted(TEST_IMAGES.glob("*.jpg"))
print(f"Found {len(test_images)} test images")

results = []
total_detections = 0

with torch.no_grad():
    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processing image {idx + 1}/{len(test_images)}...")

        # Load and preprocess image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Warning: Failed to load {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # CRITICAL: Resize to training size (416x416)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor [0,1], shape (3, H, W)
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension

        # Run inference
        outputs = model(img_tensor)

        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        # Filter by score threshold
        keep_score = scores >= SCORE_THRESH
        boxes = boxes[keep_score]
        scores = scores[keep_score]
        labels = labels[keep_score]

        if len(boxes) == 0:
            # No detections for this image
            continue

        # Apply NMS per class to remove duplicates
        final_boxes = []
        final_scores = []
        final_labels = []

        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]

            # Convert to torch tensors for NMS
            label_boxes_t = torch.from_numpy(label_boxes)
            label_scores_t = torch.from_numpy(label_scores)

            # Apply NMS
            keep_nms = nms(label_boxes_t, label_scores_t, NMS_THRESH)

            final_boxes.append(label_boxes[keep_nms.cpu().numpy()])
            final_scores.append(label_scores[keep_nms.cpu().numpy()])
            final_labels.append(np.full(len(keep_nms), label))

        if len(final_boxes) == 0:
            continue

        final_boxes = np.concatenate(final_boxes, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        final_labels = np.concatenate(final_labels, axis=0)

        # Convert boxes from 416x416 coordinates to normalized [0,1] coordinates
        # IMPORTANT: Boxes are in (x1, y1, x2, y2) format on 416x416 image
        x1 = final_boxes[:, 0] / IMG_SIZE
        y1 = final_boxes[:, 1] / IMG_SIZE
        x2 = final_boxes[:, 2] / IMG_SIZE
        y2 = final_boxes[:, 3] / IMG_SIZE

        # Clip to [0, 1]
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        # Create submission entries
        for i in range(len(final_boxes)):
            # RetinaNet labels are [1, num_classes], convert back to [0, num_classes-1]
            cls_id = int(final_labels[i]) - 1

            # Ensure valid class ID
            if cls_id < 0 or cls_id >= num_classes:
                continue

            results.append({
                'image_id': img_path.stem,
                'class_id': cls_id,
                'confidence': float(final_scores[i]),
                'xmin': float(x1[i]),
                'ymin': float(y1[i]),
                'xmax': float(x2[i]),
                'ymax': float(y2[i])
            })
            total_detections += 1

print(f"\n✓ Generated {total_detections} detections across {len(test_images)} images")
print(f"  Average detections per image: {total_detections / len(test_images):.2f}")

# ========================================
# 4. Create Submission CSV
# ========================================

print(f"\n[4/5] Creating submission CSV: {OUTPUT_CSV}...")

if len(results) == 0:
    print("❌ ERROR: No detections generated! Check your score threshold.")
    print(f"   Current threshold: {SCORE_THRESH}")
    print("   Try lowering it to 0.05 or 0.10")
else:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(results)} detections to {OUTPUT_CSV}")

    # Show sample of submission
    print("\nSample of submission (first 5 rows):")
    print(df.head())

    # Statistics
    print("\nDetection statistics:")
    print(f"  Total detections: {len(df)}")
    print(f"  Images with detections: {df['image_id'].nunique()}")
    print(f"  Detections per class:")
    for cls_id in range(num_classes):
        count = len(df[df['class_id'] == cls_id])
        if count > 0:
            print(f"    Class {cls_id} ({class_names[cls_id]}): {count}")

# ========================================
# 5. Submit to Kaggle
# ========================================

print(f"\n[5/5] Submitting to Kaggle...")
print(f"Message: '{SUBMISSION_MESSAGE}'")

# Check if Kaggle CLI is available
import subprocess

try:
    # Submit to Kaggle
    result = subprocess.run(
        [
            'kaggle', 'competitions', 'submit',
            '-c', 'ecse-415-object-recognition',
            '-f', OUTPUT_CSV,
            '-m', SUBMISSION_MESSAGE
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        print("✓ Submission successful!")
        print(result.stdout)
    else:
        print("❌ Submission failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("\nYou can manually submit by running:")
        print(f"  !kaggle competitions submit -c ecse-415-object-recognition -f {OUTPUT_CSV} -m \"{SUBMISSION_MESSAGE}\"")

except FileNotFoundError:
    print("❌ Kaggle CLI not found!")
    print("\nPlease install Kaggle CLI or manually submit:")
    print(f"  1. Download {OUTPUT_CSV}")
    print(f"  2. Go to https://www.kaggle.com/c/ecse-415-object-recognition/submit")
    print(f"  3. Upload {OUTPUT_CSV}")

except subprocess.TimeoutExpired:
    print("❌ Submission timed out!")
    print(f"\nPlease manually submit:")
    print(f"  !kaggle competitions submit -c ecse-415-object-recognition -f {OUTPUT_CSV} -m \"{SUBMISSION_MESSAGE}\"")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"CSV file: {OUTPUT_CSV}")
print(f"Total detections: {total_detections}")
print("\nNext steps:")
print("1. Check Kaggle for your score")
print("2. If score is too low, try different thresholds:")
print("   - Lower SCORE_THRESH (0.10, 0.05) for more detections")
print("   - Higher SCORE_THRESH (0.20, 0.25) for fewer, confident detections")
print("3. Generate multiple submissions with different thresholds")
print("=" * 70)
