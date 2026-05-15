# Galaxy Classification using SDSS Dataset

An internship-ready machine learning project that classifies astronomical objects into `GALAXY`, `STAR`, and `QSO` classes using SDSS-like photometric features.

This project demonstrates the full ML workflow: dataset loading, data cleaning, exploratory data analysis, feature engineering, model training, evaluation, visualization, and model saving.

## Project Overview

The Sloan Digital Sky Survey (SDSS) is one of the most important astronomy surveys in the world. It captures photometric measurements of celestial objects through five filters: `u`, `g`, `r`, `i`, and `z`.

In this project, we use those measurements plus `redshift` to train machine learning models that predict the object class:

- `GALAXY`: a galaxy
- `STAR`: a star
- `QSO`: a quasar, also called a quasi-stellar object

The project is built so it runs immediately. If `data/sdss_galaxy_data.csv` is missing, the script creates a synthetic SDSS-like dataset for demonstration. Later, you can replace it with a real SDSS export.

## Project Structure

```text
galaxy-classification/
├── data/
│   └── README.md
├── notebooks/
│   └── galaxy_classification_eda.ipynb
├── src/
│   └── train.py
├── models/
├── images/
├── README.md
├── requirements.txt
├── .gitignore
└── run.sh
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset Explanation

The dataset contains astronomy-style features commonly found in SDSS classification tasks:

| Feature | Meaning |
|---|---|
| `u`, `g`, `r`, `i`, `z` | Brightness measurements through five SDSS photometric filters |
| `redshift` | Change in wavelength caused by object movement and cosmic expansion |
| `u_g`, `g_r`, `r_i`, `i_z` | Engineered color-index features |
| `class` | Target label: `GALAXY`, `STAR`, or `QSO` |

Color-index features are useful because object classes often have different brightness patterns across filters.

## Installation

```bash
cd galaxy-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

```bash
bash run.sh
```

Or run the Python file directly:

```bash
python src/train.py
```

After running, the project generates:

- `data/sdss_galaxy_data.csv`
- `models/best_galaxy_classifier.joblib`
- `models/model_comparison.csv`
- `models/feature_importance.csv`
- `images/class_distribution.png`
- `images/color_redshift_scatter.png`
- `images/correlation_heatmap.png`
- `images/model_accuracy_comparison.png`
- `images/confusion_matrix.png`
- `images/feature_importance.png`

## Models Trained

The project trains and compares three classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The best model is selected based on test accuracy and saved using Joblib.

## Results

The script prints accuracy and classification reports for all models. It also saves a model comparison chart, confusion matrix, and feature importance plot.

Latest verified run on the generated SDSS-like dataset:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9933 |
| Decision Tree | 0.9933 |
| Random Forest | 0.9983 |

The Random Forest model performed best because it can capture non-linear relationships between redshift, magnitudes, and color-index features.

### Visual Outputs

![Class Distribution](images/class_distribution.png)

![Correlation Heatmap](images/correlation_heatmap.png)

![Model Accuracy Comparison](images/model_accuracy_comparison.png)

![Confusion Matrix](images/confusion_matrix.png)

![Feature Importance](images/feature_importance.png)

## Interview Explanation

This project solves an astronomy classification problem using supervised machine learning. I built a complete ML pipeline that loads SDSS-like data, cleans missing values, engineers color-index features from photometric bands, performs EDA, trains multiple models, compares accuracy, visualizes errors using a confusion matrix, and saves the best model for reuse.

The key idea is that astronomical objects have different brightness patterns across filters. By using magnitude values, redshift, and color-index features, the model can learn patterns that separate stars, galaxies, and quasars.

## What I Learned

- How to structure a professional ML project
- How to clean and prepare tabular data
- How to create useful astronomy-inspired features
- How to train and compare multiple classification models
- How to evaluate model performance with accuracy, classification reports, and confusion matrices
- How to save a trained ML model for future use

## Future Improvements

- Replace the synthetic dataset with a real SDSS export
- Add hyperparameter tuning with GridSearchCV
- Add cross-validation
- Build a Streamlit web app for interactive predictions
- Track experiments using MLflow

## One-Minute Project Pitch

I built a galaxy classification project using SDSS-style astronomy data. The goal was to classify celestial objects as galaxies, stars, or quasars using photometric magnitudes and redshift. I cleaned the data, engineered color-index features like `u-g` and `g-r`, performed EDA, trained Logistic Regression, Decision Tree, and Random Forest models, and compared their accuracy. I also generated visualizations like class distribution, correlation heatmap, confusion matrix, and feature importance. The best model is saved using Joblib, making the project reusable and ready for deployment.

## Common Interview Questions

**Why did you use color-index features?**  
Color-index features capture the difference in brightness between filters. These differences help identify object types because stars, galaxies, and quasars emit light differently across wavelengths.

**Why train multiple models?**  
Training multiple models helps compare simple linear performance against tree-based non-linear models. It also shows that model choice is based on evidence, not guesswork.

**Why can Random Forest perform better here?**  
Random Forest can capture non-linear relationships and feature interactions. Astronomy data often has complex relationships between magnitudes, redshift, and object class.

**What is a confusion matrix?**  
A confusion matrix shows correct and incorrect predictions for each class. It helps identify which classes the model confuses most often.

**How would you improve this project?**  
I would use a real SDSS dataset, apply cross-validation, tune hyperparameters, and deploy the model using Streamlit or FastAPI.
