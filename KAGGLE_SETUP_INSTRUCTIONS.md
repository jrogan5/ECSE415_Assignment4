# Kaggle Setup Instructions for ECSE 415 Assignment 4

This guide will help you set up and run the traffic sign detection assignment in Kaggle with GPU support.

## Prerequisites

- A Kaggle account (free)
- Your Road_Signs_Detection_Dataset prepared and ready to upload

## Step 1: Upload Your Dataset to Kaggle

### Option A: Create a New Dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload your `Road_Signs_Detection_Dataset` folder containing:
   ```
   Road_Signs_Detection_Dataset/
   ├── train/
   │   ├── images/  (*.jpg files)
   │   └── labels/  (*.txt files in YOLO format)
   ├── test/
   │   └── images/  (*.jpg files)
   └── sample_submission.csv
   ```
4. Name it something like "road-signs-detection" or "traffic-signs-dataset"
5. Set visibility to "Private" if you don't want to share it publicly
6. Click "Create"
7. **Note the dataset name** - you'll need it for the notebook!

### Option B: Upload Files Directly to Notebook (for smaller datasets)
1. In your Kaggle notebook, click "Add Data" → "Upload"
2. Upload the entire dataset folder
3. The files will be available in `/kaggle/input/`

## Step 2: Create a New Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. In the top-right, click "File" → "Import Notebook"
4. Upload the `AS4_Kaggle.ipynb` file from this repository

**OR**

1. Create a blank notebook
2. Copy the entire content of `AS4_Kaggle.ipynb` and paste it

## Step 3: Configure the Notebook

### A. Add Your Dataset
1. In the right sidebar, click "+ Add Data"
2. Search for your dataset name (the one you uploaded in Step 1)
3. Click "Add" to attach it to your notebook

### B. Update the Dataset Path
1. Find the first code cell (Environment Setup)
2. Update this line:
   ```python
   BASE_PATH = Path("/kaggle/input/road-signs-detection")
   ```
   Replace `"road-signs-detection"` with your actual dataset name

3. If you're not sure of the exact name, run the cell and check the warning message - it will list all available datasets

### C. Enable GPU
1. Click "Settings" in the right sidebar (or the ⚙️ icon)
2. Under "Accelerator", select **GPU T4 x2** or **GPU P100**
3. Click "Save"
4. The notebook session will restart with GPU enabled

### D. Enable Internet (if needed)
1. In Settings, turn on **Internet**
2. This is required for downloading pretrained models (YOLOv8)
3. Note: Internet-enabled notebooks cannot be made public

## Step 4: Run the Notebook

### Quick Start
1. Click "Run All" to execute all cells sequentially
2. Training will take approximately:
   - **YOLOv8 baseline**: 5-10 minutes
   - **RetinaNet v1**: 15-25 minutes (50 epochs)
   - **RetinaNet v2**: 20-30 minutes (60 epochs)

### Recommended Approach
Run cells one at a time to monitor progress:

1. **Cell 1-3**: Environment setup and GPU check
   - Verify GPU is available
   - Confirm dataset paths are correct

2. **Cell 4-6**: Data loading and visualization
   - Inspect your dataset
   - Verify images and labels are loading correctly

3. **Cell 7-8**: YOLOv8 baseline training
   - Quick baseline model (~10 min)
   - Generates initial results

4. **Cell 9**: RetinaNet v1 training (optional)
   - Basic RetinaNet model
   - Can skip if you want to go straight to v2

5. **Cell 10**: RetinaNet v2 training (recommended)
   - Improved model with better augmentation
   - This is the best-performing model

6. **Cell 11**: Model evaluation
   - Compute mAP, F1 scores
   - Generate confusion matrices

7. **Cell 12-13**: Submission generation
   - Creates multiple CSV files with different thresholds
   - Look for files in the output panel on the right

## Step 5: Download Submission Files

After running the submission cells:

1. Click the "Output" tab in the right sidebar
2. You'll see several CSV files:
   ```
   submission_retinanet_v2_very_low.csv
   submission_retinanet_v2_low.csv
   submission_retinanet_v2_med.csv       ← START WITH THIS ONE
   submission_retinanet_v2_med_high.csv
   ```

3. Click the download icon (⬇️) next to each file to save it

4. **Start by submitting `submission_retinanet_v2_med.csv`** to the competition

## Step 6: Submit to Kaggle Competition

1. Go to your competition page
2. Click "Submit Predictions"
3. Upload the CSV file
4. Add a description (e.g., "RetinaNet v2 with medium threshold")
5. Click "Submit"
6. Wait for the score!

### Adjusting Based on Results

- **Score too low** (missing detections)?
  - Try `submission_retinanet_v2_low.csv` (lower threshold = more detections)

- **Too many false positives**?
  - Try `submission_retinanet_v2_med_high.csv` (higher threshold = fewer detections)

## Troubleshooting

### "Dataset not found" Error
**Problem**: The BASE_PATH doesn't match your dataset name

**Solution**:
1. Run the environment setup cell
2. Check the warning message - it will list available datasets
3. Update the `BASE_PATH` variable to match exactly

### "CUDA out of memory" Error
**Problem**: GPU memory is full

**Solutions**:
1. Reduce batch size in training cells:
   ```python
   batch_size = 4  # Instead of 8
   ```
2. Restart the notebook session (Session → Restart)
3. Use a smaller model or fewer epochs

### Training is Very Slow
**Problem**: GPU is not enabled or not being used

**Solutions**:
1. Check the GPU cell - it should show "✓ GPU is available!"
2. If not, enable GPU in Settings → Accelerator
3. Restart the notebook session

### "No module named 'ultralytics'" Error
**Problem**: YOLOv8 library not installed

**Solution**:
Add this cell before the YOLOv8 training:
```python
!pip install -q ultralytics
```

### Internet Connection Required Error
**Problem**: Downloading pretrained models requires internet

**Solution**:
Enable Internet in Settings → Internet → On

## Expected Training Times (with GPU)

| Model | Epochs | Batch Size | Time |
|-------|--------|------------|------|
| YOLOv8n | 30 | 16 | 5-10 min |
| RetinaNet v1 | 50 | 8 | 15-25 min |
| RetinaNet v2 | 60 | 8 | 20-30 min |

*Times are approximate and depend on dataset size and GPU type (P100 vs T4)*

## Saving Your Work

Kaggle automatically saves your notebook, but to be safe:

1. Click "Save Version" (top-right)
2. Choose "Save & Run All" to create a checkpoint
3. Add a note (e.g., "Final submission version")

This creates a snapshot you can return to later.

## Tips for Best Results

1. **Always enable GPU** - Training on CPU is 10-20x slower
2. **Start with medium threshold** - `v2_med.csv` usually performs best
3. **Monitor training loss** - If it's not decreasing, something might be wrong
4. **Check sample predictions** - Visualize results before submitting
5. **Try multiple thresholds** - You can submit multiple times per day

## Dataset Structure Verification

Before training, verify your dataset structure:

```python
# Run this in a new cell to check
print("Train images:", len(list(TRAIN_IMAGES.glob("*.jpg"))))
print("Train labels:", len(list(TRAIN_LABELS.glob("*.txt"))))
print("Test images:", len(list(TEST_IMAGES.glob("*.jpg"))))

# Should see numbers like:
# Train images: 600-800
# Train labels: 600-800
# Test images: 200-300
```

## Getting Help

If you encounter issues:

1. Check the notebook output for error messages
2. Verify your dataset structure matches the expected format
3. Make sure GPU is enabled and working
4. Check that all file paths are correct
5. Try restarting the notebook session

## Next Steps After Successful Setup

1. Run all cells to train models
2. Download submission files
3. Submit to Kaggle competition
4. Iterate on thresholds based on leaderboard score
5. (Optional) Experiment with hyperparameters for better results

---

**Good luck with your assignment! 🚀**

For more details about the improvements made to the RetinaNet model, see `IMPROVEMENTS_SUMMARY.md`.
