# Data can be downloaded from: 
- (https://www.kaggle.com/competitions/playground-series-s5e5/data)

## Goal:
- Your Goal: Your goal is to predict how many calories were burned during a workout.

## Usage:
1. Preprocess Data:
   python scripts/Preprocess_Train_Test_Fixed.py

   Outputs: data/preprocessed_train.csv, data/preprocessed_test.csv, scaler.pkl
   Features: BMI, BMR, Intensity, METs, HR_Duration, Weight_BMI, METs_Duration, polynomial features, etc.

2. Run XGBoost Model:
   python scripts/XGBoost_Final_Submission_Download.py

   Outputs: data/submission.csv, xgb_model.pkl, plots (residual_plot.png, shap_feature_importance.png)
   Downloads submission.csv (Jupyter/Colab).

Results:
  XGBoost: Validation MAE ~2.13, robust but slightly overfits.

Future Improvements:
Currently I am trying to improve the model using: 1) Neural Networks and 2) Ensemble.



