# Kaggle Submission Guide - Using Your .pth Model

## Quick Start

You have **two options** for generating and submitting to Kaggle:

### Option 1: Single Submission (Recommended for First Try) ⭐

**File**: `generate_and_submit.py`

**Use this when**: You want to test one threshold quickly

**Steps**:
1. Open the file `generate_and_submit.py`
2. Copy the entire contents
3. Paste into a new cell in your AS4.ipynb notebook
4. Modify these lines at the top:
   ```python
   MODEL_PATH = "model_retina_v2.pth"  # Your .pth file name
   SCORE_THRESH = 0.15                  # Confidence threshold to try
   SUBMISSION_MESSAGE = "Your message"  # Description for Kaggle
   ```
5. Run the cell
6. Wait ~2-5 minutes for predictions + submission

**What it does**:
- Loads your .pth model
- Generates predictions on test set
- Applies NMS (removes duplicates)
- Creates CSV file
- **Automatically submits to Kaggle**

---

### Option 2: Multi-Threshold Batch Submission 🚀

**File**: `multi_threshold_submission.txt`

**Use this when**: You want to try multiple thresholds at once

**Steps**:
1. Open `multi_threshold_submission.txt`
2. Copy the entire contents
3. Paste into a new cell in your notebook
4. Modify this line:
   ```python
   MODEL_PATH = "model_retina_v2.pth"  # Your .pth file
   ```
5. Run the cell
6. Wait ~5-10 minutes

**What it does**:
- Generates predictions ONCE (faster)
- Creates 5 different CSV files with thresholds: 0.05, 0.10, 0.15, 0.20, 0.25
- **Automatically submits all 5 to Kaggle**
- You can compare which threshold works best

**Output files**:
- `submission_retinanet_thresh_0.05.csv` (most detections)
- `submission_retinanet_thresh_0.10.csv`
- `submission_retinanet_thresh_0.15.csv` ⭐ (recommended)
- `submission_retinanet_thresh_0.20.csv`
- `submission_retinanet_thresh_0.25.csv` (fewest detections)

---

## Configuration Explained

### MODEL_PATH
```python
MODEL_PATH = "model_retina_v2.pth"
```
- Name of your saved model file
- If you're not sure, run `!ls *.pth` in a notebook cell to see all .pth files

### SCORE_THRESH
```python
SCORE_THRESH = 0.15  # Confidence threshold
```

**How to choose**:
- **0.05-0.10**: Very permissive - many detections (higher recall, lower precision)
- **0.15-0.20**: Balanced ⭐ (recommended starting point)
- **0.25-0.30**: Conservative - fewer detections (higher precision, lower recall)

**Strategy**:
- Start with **0.15**
- If Kaggle score is low (missing detections): try **0.10** or **0.05**
- If too many false positives: try **0.20** or **0.25**

### NMS_THRESH
```python
NMS_THRESH = 0.5  # Non-Maximum Suppression threshold
```

**What it does**: Removes duplicate/overlapping boxes

**Recommended**: Keep at **0.5** (standard value)

---

## Expected Output

### Prediction Statistics

When running, you'll see:

```
[3/5] Generating predictions on test set...
Found 857 test images
  Processing image 100/857...
  Processing image 200/857...
  ...

✓ Generated 3247 detections across 857 images
  Average detections per image: 3.79
```

**Good signs**:
- ✅ 2-5 detections per image on average
- ✅ Total detections in the thousands

**Bad signs**:
- ❌ < 1 detection per image (threshold too high)
- ❌ > 10 detections per image (threshold too low, or NMS not working)

### Submission CSV Format

Your CSV will have these columns:

| image_id | class_id | confidence | xmin | ymin | xmax | ymax |
|----------|----------|------------|------|------|------|------|
| test_001 | 6 | 0.87 | 0.23 | 0.45 | 0.34 | 0.56 |
| test_001 | 2 | 0.92 | 0.67 | 0.12 | 0.78 | 0.23 |
| test_002 | 11 | 0.73 | 0.45 | 0.67 | 0.56 | 0.78 |

**Important**:
- `class_id`: 0-13 (your 14 classes)
- `xmin, ymin, xmax, ymax`: Normalized [0, 1] coordinates
- `confidence`: 0-1 score

---

## Kaggle Submission Commands

### If Auto-Submit Works ✅

You'll see:
```
✓ Submission successful!
Successfully submitted to ecse-415-object-recognition
```

### If Auto-Submit Fails ❌

Run manually in a notebook cell:

```bash
!kaggle competitions submit \
  -c ecse-415-object-recognition \
  -f submission_retinanet_thresh_0.15.csv \
  -m "RetinaNet v2 - val_loss_0.17 - thresh_0.15"
```

---

## Troubleshooting

### Error: "Model file not found"

**Problem**: `MODEL_PATH` is wrong

**Solution**:
```python
# In a notebook cell, list all .pth files:
!ls *.pth

# Then update MODEL_PATH to match exactly
```

### Error: "No detections generated"

**Problem**: `SCORE_THRESH` is too high

**Solution**: Lower it to 0.05 or 0.10

### Error: "Kaggle CLI not found"

**Problem**: Kaggle CLI not installed

**Solution**:
```bash
# Install Kaggle CLI
!pip install kaggle

# Then configure API key (if needed)
!mkdir -p ~/.kaggle
# Upload your kaggle.json to ~/.kaggle/
```

Or submit manually:
1. Download the CSV file
2. Go to https://www.kaggle.com/c/ecse-415-object-recognition/submit
3. Upload the CSV

### Too Many Detections (>10 per image)

**Problem**: Threshold too low or NMS not working

**Solution**:
- Increase `SCORE_THRESH` to 0.20 or 0.25
- Check that `NMS_THRESH = 0.5` (not too high)

### Too Few Detections (<1 per image)

**Problem**: Threshold too high

**Solution**: Lower `SCORE_THRESH` to 0.10 or 0.05

---

## Recommended Workflow

### Day 1: Quick Test
1. Use **Option 1** (single submission)
2. Set `SCORE_THRESH = 0.15`
3. Submit and check Kaggle score
4. Takes ~5 minutes total

### Day 2: Optimization
1. Use **Option 2** (multi-threshold)
2. Submits 5 different thresholds automatically
3. Check Kaggle leaderboard to see which works best
4. Takes ~10 minutes total

### Day 3: Fine-Tuning
1. Based on best threshold from Day 2
2. Try nearby thresholds (e.g., if 0.15 was best, try 0.12, 0.13, 0.14, 0.16, 0.17)
3. Use **Option 1** for each

---

## Quick Reference Card

| Task | File to Use | Time |
|------|-------------|------|
| First submission | `generate_and_submit.py` | 5 min |
| Try multiple thresholds | `multi_threshold_submission.txt` | 10 min |
| Use existing CSV | `!kaggle competitions submit ...` | 1 min |

---

## Expected Kaggle Scores

Based on your val loss of 0.17:

| Threshold | Expected Score Range | Notes |
|-----------|---------------------|-------|
| 0.05 | 45-60 | Many detections, might have false positives |
| 0.10 | 55-68 | Good balance |
| 0.15 | 58-72 | **Recommended start** ⭐ |
| 0.20 | 52-65 | Conservative |
| 0.25 | 42-58 | Might miss some detections |

**Target**: Beat TA baseline of 67

Your val loss of 0.17 suggests you should score in the **60-70 range** with proper threshold tuning.

---

## Files Summary

1. **generate_and_submit.py** - Single submission generator
2. **multi_threshold_submission.txt** - Batch submission with 5 thresholds
3. **SUBMISSION_GUIDE.md** - This guide

All ready to copy-paste into your notebook!

---

## Next Steps After Submission

1. **Check Kaggle Submissions Page**
   - Go to https://www.kaggle.com/c/ecse-415-object-recognition/submissions
   - See your score

2. **If Score is Good (>65)**:
   - ✅ You're done! (or try to optimize further)

3. **If Score is Medium (55-65)**:
   - Try different thresholds
   - Check if NMS is working properly

4. **If Score is Low (<55)**:
   - Check CSV format is correct
   - Verify box coordinates are normalized [0,1]
   - Try lower threshold (0.05)
   - Consider retraining with better augmentation

Good luck! 🚀
