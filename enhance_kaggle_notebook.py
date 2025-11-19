#!/usr/bin/env python3
"""
Script to further enhance the Kaggle notebook with additional features
"""
import json

def enhance_notebook():
    """Add additional Kaggle-specific enhancements"""

    # Read the converted notebook
    with open('/home/user/ECSE415_Assignment4/AS4_Kaggle.ipynb', 'r') as f:
        nb = json.load(f)

    # Create GPU check cell
    gpu_check_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ================================\n",
            "# GPU Availability Check\n",
            "# ================================\n",
            "import torch\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"✓ GPU is available!\")\n",
            "    print(f\"  Device name: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"  Device count: {torch.cuda.device_count()}\")\n",
            "    print(f\"  Current device: {torch.cuda.current_device()}\")\n",
            "    print(f\"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB\")\n",
            "    print(f\"  Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB\")\n",
            "else:\n",
            "    print(\"⚠ GPU is NOT available. Training will be slow.\")\n",
            "    print(\"  In Kaggle: Go to Settings > Accelerator > Select 'GPU'\")\n",
            "    print(\"  Then restart the notebook session.\")\n"
        ]
    }

    # Insert GPU check cell after the environment setup (position 2)
    nb['cells'].insert(2, gpu_check_cell)

    # Update the markdown header to mention Kaggle compatibility
    if nb['cells'][0]['cell_type'] == 'markdown':
        original_source = ''.join(nb['cells'][0]['source'])
        if 'Kaggle' not in original_source:
            # Update to include both Colab and Kaggle
            nb['cells'][0]['source'] = [
                '<a href="https://colab.research.google.com/github/jrogan5/ECSE415_Assignment4/blob/main/AS4_Kaggle.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>\n',
                '\n',
                '# ECSE 415 Assignment 4 - Traffic Sign Detection (Kaggle Version)\n',
                '\n',
                'This notebook has been adapted to run in Kaggle with GPU support.\n',
                '\n',
                '## Setup Instructions for Kaggle:\n',
                '\n',
                '1. **Add Dataset**: Add your "Road_Signs_Detection_Dataset" as a dataset to this notebook\n',
                '2. **Update Dataset Path**: In the first code cell, update `BASE_PATH` to match your dataset name\n',
                '3. **Enable GPU**: Settings → Accelerator → GPU (T4 or P100)\n',
                '4. **Internet**: Enable if you need to download pretrained models (Settings → Internet → On)\n',
                '\n',
                '## What this notebook does:\n',
                '\n',
                '- Trains **YOLOv8** baseline model for traffic sign detection\n',
                '- Trains improved **RetinaNet** models (v1 and v2)\n',
                '- Generates multiple submission files with different confidence thresholds\n',
                '- Evaluates models with mAP, F1 score, and confusion matrices\n'
            ]

    # Now update any remaining hardcoded paths
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Update any remaining Path(root, ...) constructions for outputs
            if i > 3:  # Skip the setup cells
                new_source = []
                modified = False
                for line in cell['source']:
                    # Check for submission file paths
                    if 'f"submission_' in line and 'Path(' not in line:
                        # This might be a filename only, check if we need to add OUTPUT_PATH
                        if '.csv"' in line and 'OUT' not in line:
                            # Already handled by previous conversion
                            new_source.append(line)
                        else:
                            new_source.append(line)
                    # Update yolo_split directory to use OUTPUT_PATH
                    elif 'Path(root, "yolo_split")' in line:
                        new_source.append(line.replace('Path(root, "yolo_split")', 'Path(OUTPUT_PATH, "yolo_split")'))
                        modified = True
                    else:
                        new_source.append(line)

                if modified:
                    cell['source'] = new_source
                    print(f"Updated paths in cell {i}")

    # Save the enhanced notebook
    with open('/home/user/ECSE415_Assignment4/AS4_Kaggle.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

    print("\nEnhancement complete!")
    print(f"Total cells: {len(nb['cells'])}")

if __name__ == "__main__":
    enhance_notebook()
