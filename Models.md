# Models for Calorie Expenditure Prediction

This document describes the machine learning models used in the [Machine-Learning-Predict-Calorie-Expenditure](https://github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure) project for the [Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5) competition. The goal is to predict calories burned based on features like `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, and `Body_Temp`. The models include XGBoost, a Neural Network (Keras), and an Ensemble, with a current Kaggle MAE of **0.05909**.

## 1. XGBoost
### Overview
XGBoost is a gradient boosting framework optimized for speed and performance. It was chosen for its robustness in handling tabular data and ability to capture non-linear relationships through tree-based learning.

### Parameters
- `n_estimators`: 764
- `max_depth`: 8
- `learning_rate`: 0.029356697754814032
- `subsample`: 0.7563204170158759
- `colsample_bytree`: 0.6022633900567109
- `random_state`: 42
- `n_jobs`: -1

### Features
- Engineered: `BMI`, `BMR`, `Intensity`, `METs`, `Temp_Anomaly`, `HR_Duration`, `Weight_BMI`, `METs_Duration`
- Polynomial: `Age`, `Heart_Rate` (degree=2)
- Categorical: One-hot encoded `Age_Group` (`31-40`, `41-50`, `51-60`, `61-80`)

### Performance
- **5-Fold CV MAE**: ~2-5
- **Validation MAE**: ~2.13
- **Kaggle MAE**: Contributes to ensemble score of 0.05909
- **Strengths**: Robust, handles feature interactions well.
- **Weaknesses**: May overfit with complex features like polynomial terms.

### Usage
1. Preprocess data:
   ```bash
   python scripts/Preprocess_Train_Test_Fixed.py

Run XGBoost:python scripts/XGBoost_Final_Submission_Download.py

Output: data/submission.csv (downloadable in Jupyter/Colab).

2. Neural Network (Keras)
Overview
A deep neural network (DNN) built with Keras/TensorFlow to capture complex, non-linear patterns in the data. It uses dropout and batch normalization to prevent overfitting.
Architecture

Layers:
Dense(128, ReLU) + BatchNormalization + Dropout(0.3)
Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
Dense(32, ReLU)
Dense(1, linear)

Optimizer: Adam
Loss: Mean Absolute Error (MAE)
Early Stopping: Patience=10, monitor val_loss

Features
Same as XGBoost, with scaled numerical features for better convergence.
Performance

Validation MAE: ~0.5-1.5 (target <1.0)
Kaggle MAE: Contributes to ensemble score of 0.05909
Strengths: Captures non-linear relationships, flexible architecture.
Weaknesses: Sensitive to hyperparameter tuning, requires more computational resources.

Usage

Preprocess data:python scripts/Preprocess_Train_Test_Fixed.py

Run Neural Network:python scripts/Neural_Network_Submission.py

Output: data/submission_nn.csv (downloadable in Jupyter/Colab).

3. Ensemble
Overview
Combines XGBoost and Neural Network predictions using a 50-50 weighted average to leverage the strengths of both models (robustness of XGBoost, flexibility of Neural Network).
Method

Blending: 0.5 * XGBoost_Predictions + 0.5 * Neural_Network_Predictions
Output: data/submission_ensemble.csv

Performance

Kaggle MAE: 0.05909 (public leaderboard)
Strengths: Reduces variance, improves generalization.
Weaknesses: Limited by the performance of individual models.

Usage

Generate XGBoost and Neural Network submissions:python scripts/XGBoost_Final_Submission_Download.py
python scripts/Neural_Network_Submission.py

Run Ensemble:python scripts/Ensemble_Submission.py

Output: data/submission_ensemble.csv (downloadable in Jupyter/Colab).

Running the Models

Setup:

Clone the repository:git clone https://github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure.git
cd Machine-Learning-Predict-Calorie-Expenditure

Install dependencies:pip install -r requirements.txt

Download train.csv and test.csv from Kaggle and place in data/.

Pipeline: 
Preprocess: python scripts/Preprocess_Train_Test_Fixed.py
Run models: python scripts/XGBoost_Final_Submission_Download.py, python scripts/Neural_Network_Submission.py
Ensemble: python scripts/Ensemble_Submission.py
Submit data/submission_ensemble.csv to Kaggle.

## Future Improvements
Add LightGBM: Introduce a third model for diversity in ensembling.
Feature Engineering: Add features like Heart_Rate / Age or interaction terms.
Hyperparameter Tuning: Use Optuna for Neural Network architecture.
Ensemble Weights: Experiment with 70-30 or other ratios.
Cross-Validation: Implement k-fold CV for Neural Network.

## Credits
Built with scikit-learn, XGBoost, TensorFlow, and pandas.

# For issues or contributions, please open a pull request or contact Souptik96.
</xArtifact>

---

### **Steps to Add Models.md to the Repository**
1. **Ensure Git is Set Up**:
   - Verify Git installation:
     ```bash
     git --version
     ```
   - If not installed, follow the [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

2. **Navigate to Repository**:
   - If not already cloned:
     ```bash
     git clone https://github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure.git
     cd Machine-Learning-Predict-Calorie-Expenditure
     ```
   - If already cloned, pull the latest changes:
     ```bash
     git pull origin main
     ```

3. **Create or Update Models.md**:
   - Save the `Models.md` content:
     ```bash
     nano Models.md
     # Paste the Models.md content, save, and exit
     ```
   - Alternatively, copy the content into `Models.md` using your editor.

4. **Verify Existing Scripts**:
   - Ensure the `scripts/` directory contains:
     - `Preprocess_Train_Test_Fixed.py` (artifact_version_id="644022ca-6ab6-4004-a9b7-8e3b038d44dc")
     - `XGBoost_Final_Submission_Download.py` (artifact_version_id="bfd42ecd-e459-4631-a00f-c1e150a19c61")
     - `Neural_Network_Submission.py` (artifact_version_id="1550bb26-f2ac-47b7-899a-e6b23692f809")
     - `Ensemble_Submission.py` (artifact_version_id="4817ce76-bc6c-4f8d-ba08-9e64f337c2d4")
   - Check:
     ```bash
     ls scripts/
     ```

5. **Stage and Commit Changes**:
   ```bash
   git add Models.md
   git commit -m "Added Models.md with details on XGBoost, Neural Network, and Ensemble"


Push to GitHub:
git push origin main


If main branch issues arise, ensure it’s set:git branch -M main
git push -u origin main

Verify on GitHub:

Visit https://github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure.
Confirm Models.md is present and renders correctly.
Check that links (e.g., Kaggle, EDA) work.

Run Scripts and Submit:

Follow Models.md instructions to run the pipeline:python scripts/Preprocess_Train_Test_Fixed.py
python scripts/XGBoost_Final_Submission_Download.py
python scripts/Neural_Network_Submission.py
python scripts/Ensemble_Submission.py


Download submission_ensemble.csv and submit to Kaggle.
Share the new MAE score.

Troubleshooting
Git Push Fails:

Verify write access to the repository.
Use a personal access token if prompted:git remote set-url origin https://<your-username>:<token>@github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure.git

Models.md Not Rendering:
Check Markdown syntax in Models.md.
Validate with a Markdown editor (e.g., Dillinger).


Scripts Missing:
If scripts aren’t in scripts/, add them from previous artifacts:nano scripts/Preprocess_Train_Test_Fixed.py
# Paste content, save, and repeat for others

Stage and commit:git add scripts/
git commit -m "Added missing scripts"
git push


Kaggle Submission Issues:
Verify submission_ensemble.csv format:submission = pd.read_csv('data/submission_ensemble.csv')
print(submission.head())
print(submission.dtypes)
Expected: id (int64), Calories (float64).


Next Steps
Add Models.md:
Follow the steps to add Models.md to the repository.
Verify it at https://github.com/Souptik96/Machine-Learning-Predict-Calorie-Expenditure/blob/main/Models.md.

Run Pipeline:
Execute the scripts as outlined in Models.md.
Download and submit submission_ensemble.csv.


Improve MAE:
If new MAE > 0.05909, try:
Adjusting ensemble weights (e.g., 70-30 XGBoost-NN).
Adding LightGBM (I can provide a script).
New features (e.g., Heart_Rate / Age).

Share the new MAE score.


Enhance Repository:
Update README.md with a link to Models.md.
Add a Jupyter notebook for an end-to-end workflow.
Include SHAP plots in Models.md or README.md.

Please confirm once Models.md is added to the repository or share any errors during the GitHub update. Also, share the new Kaggle MAE after submitting submission_ensemble.csv. Let’s make this repo shine and push that MAE below 0.05909!
