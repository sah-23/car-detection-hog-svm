## Car Detection with SVM (HOG + Linear SVM)

Build a classic computer-vision pipeline for car vs non-car detection using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM). This repository includes an end-to-end notebook demonstrating feature extraction, model training, evaluation, and qualitative visualization.

Dataset: `Car and Non-Car Dataset` from Kaggle — see the official source here: `https://www.kaggle.com/datasets/lachlannegus/car-and-non-car-dataset`

### Why this project is resume-ready
- **Clear problem**: binary image classification (car vs non-car)
- **Classic CV pipeline**: HOG features + SVM baseline
- **Reproducible**: instructions for setup, dataset download, and running the notebook
- **Results-focused**: sections for metrics and visual examples

### Repository structure
- `Car_detection_withSvm.ipynb` — main notebook with data prep, feature engineering, training, and evaluation
- `data/` — place the dataset here locally (not tracked)
- `outputs/` — optional directory for saved artifacts like figures or metrics (not tracked)

### Setup
1) Python 3.9–3.11 recommended
2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
```

3) Install dependencies (typical packages for HOG + SVM workflow)

```bash
pip install --upgrade pip
pip install numpy scikit-image scikit-learn opencv-python matplotlib seaborn jupyter
```

If you prefer, generate a `requirements.txt` from your environment:

```bash
pip freeze > requirements.txt
```

### Dataset
Download the dataset from Kaggle: `https://www.kaggle.com/datasets/lachlannegus/car-and-non-car-dataset`

To keep local paths private, use an environment variable for your data directory instead of hardcoding absolute paths in the notebook.

Example pattern you can adapt inside the notebook:

```python
import os

# Use an environment variable to point to your local data directory
DATA_DIR = os.getenv("DATA_DIR", "data")
cars_dir = os.path.join(DATA_DIR, "vehicles")
non_cars_dir = os.path.join(DATA_DIR, "non-vehicles")

# Example expected structure (you can adjust based on the dataset organization):
# data/
#   vehicles/
#     ... car images ...
#   non-vehicles/
#     ... non-car images ...
```

On Windows (PowerShell):

```powershell
$env:DATA_DIR = "D:\\path\\to\\your\\data"
```

On Bash (Git Bash, macOS, Linux):

```bash
export DATA_DIR="/absolute/path/to/your/data"
```

Alternatively, you can create a local `.env` with `DATA_DIR=...` and load it (e.g., with `python-dotenv`) if you prefer. The `.env` file is ignored by git in this repo.

### Running the notebook
1) Start Jupyter:

```bash
jupyter notebook
```

2) Open `Car_detection_withSvm.ipynb`
3) Ensure `DATA_DIR` points to your local dataset folder (see above)
4) Run the cells top-to-bottom

### Methodology
- **Features**: HOG descriptors via `scikit-image`
- **Model**: Linear SVM via `scikit-learn`
- **Preprocessing**: grayscale conversion or color channels, normalization, and train/validation split
- **Evaluation**: accuracy, precision/recall/F1, confusion matrix, and qualitative examples

### Results (to include)
- Quantitative metrics (e.g., accuracy, precision, recall, F1)
- Confusion matrix figure
- A few sample predictions with true/false positives and negatives

### Reproducibility tips
- Set a random seed for `train_test_split` and the SVM classifier
- Keep feature extraction parameters (HOG orientations, pixels per cell, cells per block) documented in one place in the notebook
- Save trained model and scaler locally if desired (e.g., in `outputs/`), but do not commit artifacts

### Notes on keeping data paths private
- Do not hardcode machine-specific absolute paths in the notebook
- Use `DATA_DIR` environment variable or a local `.env` file
- This repository’s `.gitignore` excludes `data/`, `outputs/`, and `.env`

### Citation
If you use this dataset, please follow the licensing and citation guidance on Kaggle.

### License
This project is provided as-is for educational and portfolio purposes. Choose and add a license (e.g., MIT) if you plan to distribute or collaborate.

