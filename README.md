# ROADNEX AI — Clickable Full Functional Dashboard

## What changed
- Sidebar modules are now real clickable navigation using a Streamlit radio navigation menu.
- The CSV upload control has been removed from the interface.
- The dataset is loaded automatically from the project folder.
- The six Colab visualizations and six ML algorithms are retained.
- The models train automatically when the app starts.
- Prediction, model comparison, confusion matrix, classification report,
  feature importance and data explorer are separate working modules.

## Required dataset placement
Put your dataset file in the same folder as `app.py` and name it:

`RTA Dataset.csv`

The app also checks common alternatives such as `rta_dataset.csv`,
`accident_dataset.csv`, `cleaned.csv` and `dataset.csv`. It also checks a
`data` subfolder.

The CSV is still required by the machine-learning project, but there is no
upload field in the dashboard. It is read in the background.

## Run on macOS
Open this folder in VS Code. In Terminal run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit (normally `http://localhost:8501`).

## Modules
1. Mission Control
2. Severity Prediction
3. Exploratory Data Analysis
4. Model Comparison
5. Confusion Matrix & Report
6. Feature Importance
7. Data Explorer
8. Project Profile

## Notebook alignment
The preprocessing, target-column candidates, feature selection, outlier
capping, 80/20 stratified split, six classifiers, macro evaluation metrics,
confusion matrix and Random Forest feature importance follow the supplied
`Transportation_Accident_Severity_Prediction_Final(1).ipynb`.

## Note
The notebook's numeric histogram, box plot, scatter plot and correlation
heatmap only appear when the loaded CSV contains the required numeric
features. The dashboard does not invent data to force a chart to appear.
