#!/usr/bin/env python3
"""
Script to convert the AS4.ipynb notebook to be Kaggle-compatible
"""
import json
import sys
from pathlib import Path

def convert_notebook_to_kaggle(input_path, output_path):
    """Convert the notebook to Kaggle format"""

    # Read the original notebook
    with open(input_path, 'r') as f:
        nb = json.load(f)

    # Create a new cell for Kaggle environment detection (will be inserted at position 1)
    kaggle_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ================================\n",
            "# Kaggle Environment Setup\n",
            "# ================================\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Detect if we're running in Kaggle\n",
            "IS_KAGGLE = os.path.exists('/kaggle')\n",
            "\n",
            "if IS_KAGGLE:\n",
            "    print(\"Running in Kaggle environment\")\n",
            "    # Note: Update 'road-signs-detection' with your actual Kaggle dataset name\n",
            "    # You can find this in the dataset URL or when you add data to your notebook\n",
            "    BASE_PATH = Path(\"/kaggle/input/road-signs-detection\")\n",
            "    OUTPUT_PATH = Path(\"/kaggle/working\")\n",
            "    \n",
            "    # Verify the dataset exists\n",
            "    if not BASE_PATH.exists():\n",
            "        print(f\"WARNING: Dataset not found at {BASE_PATH}\")\n",
            "        print(\"Available input datasets:\")\n",
            "        if Path(\"/kaggle/input\").exists():\n",
            "            for item in Path(\"/kaggle/input\").iterdir():\n",
            "                print(f\"  - {item.name}\")\n",
            "        print(\"\\nPlease update the BASE_PATH variable to match your dataset name.\")\n",
            "else:\n",
            "    print(\"Running in local/Colab environment\")\n",
            "    # For local development - update this path as needed\n",
            "    BASE_PATH = Path(\"./\")\n",
            "    OUTPUT_PATH = Path(\"./\")\n",
            "\n",
            "print(f\"Base path: {BASE_PATH}\")\n",
            "print(f\"Output path: {OUTPUT_PATH}\")\n"
        ]
    }

    # Modify Cell 1 (imports and path setup)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Find and update the path definition in Cell 1
            if 'path = "C:\\\\Users\\\\jrog1\\\\OneDrive' in source:
                print(f"Updating path definition in cell {i}")

                # Replace the Windows path line with Kaggle-compatible path
                new_source = []
                for line in cell['source']:
                    if 'path = "C:\\\\Users\\\\jrog1\\\\OneDrive' in line:
                        # Replace with dynamic path using environment detection
                        new_source.append('path = str(BASE_PATH)  # Set by environment detection cell above\n')
                    else:
                        new_source.append(line)

                cell['source'] = new_source
                break

    # Update all cells that save models or outputs
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Update YOLO training outputs
            if 'project=str(Path(root, "runs_yolov8_baseline"))' in source:
                print(f"Updating YOLO output path in cell {i}")
                new_source = []
                for line in cell['source']:
                    if 'project=str(Path(root, "runs_yolov8_baseline"))' in line:
                        new_source.append('        project=str(OUTPUT_PATH / "runs_yolov8_baseline"),\n')
                    else:
                        new_source.append(line)
                cell['source'] = new_source

            # Update submission generation paths
            if 'out_csv_path = Path(root,' in source or 'submission_retinanet' in source:
                print(f"Updating submission output paths in cell {i}")
                new_source = []
                for line in cell['source']:
                    if 'out_csv_path = Path(root,' in line:
                        # Extract the filename
                        if 'submission_' in line:
                            new_source.append(line.replace('Path(root,', 'Path(OUTPUT_PATH,'))
                        else:
                            new_source.append(line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source

            # Update model saving paths
            if 'torch.save(model' in source and '.pth' in source:
                print(f"Found model saving in cell {i}, checking if path update needed")
                # This might need manual review, but most saves use root which is fine

    # Insert the Kaggle setup cell at position 1 (after the markdown header)
    nb['cells'].insert(1, kaggle_setup_cell)

    # Save the modified notebook
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Saved to: {output_path}")
    print("\nNOTE: After uploading to Kaggle:")
    print("1. Update the dataset name in cell 1: BASE_PATH = Path('/kaggle/input/YOUR-DATASET-NAME')")
    print("2. Make sure to add your Road_Signs_Detection_Dataset as a Kaggle dataset")
    print("3. Enable GPU in notebook settings (Settings > Accelerator > GPU)")

if __name__ == "__main__":
    input_notebook = "/home/user/ECSE415_Assignment4/AS4.ipynb"
    output_notebook = "/home/user/ECSE415_Assignment4/AS4_Kaggle.ipynb"

    convert_notebook_to_kaggle(input_notebook, output_notebook)
