

This file describes the contents of this MID TERME PROJECT directory about "Patients-Data-Heart-Disease-Prediction" 

Here are the files included inside :
    - Patients-Data-Heart-Disease-Prediction.xlsx
    - notebook.ipynb
    - train.py
    - predict.py
    - predict-test.py
    - model_C=1.0.bin
    - Pipfile
    - Pipfile.lock
    - Dockerfile

About Patients-Data-Heart-Disease-Prediction.xlsx : the file is to large(36469 ko), github requires 25000 ko.
Here is the Kaggle links of the Data : 
https://www.kaggle.com/datasets/raghavramani3232/patients-data-used-for-heart-disease/data

Be careful, this is a large file which takes time to load its contents in the notebook.ipynb.
It has the shape of 237630 lines and 35 variables

Below are some points to understand in the files of this project

1. Patients Data : for Heart Disease Prediction 
==================================================

The dataset contains the following features:

    PatientID: Unique identifier for each patient.
    State: Geographic state of residence.
    Sex: Gender of the patient.
    GeneralHealth: Self-reported health status.
    AgeCategory: Categorized age group of the patient.
    HeightInMeters: Height of the patient in meters.
    WeightInKilograms: Weight of the patient in kilograms.
    BMI: Body Mass Index, calculated from height and weight.
    HadHeartAttack: Indicator of whether the patient had a heart attack.
    HadAngina: Indicator of whether the patient experienced angina.
    HadStroke: Indicator of whether the patient had a stroke.
    HadAsthma: Indicator of whether the patient has asthma.
    HadSkinCancer: Indicator of whether the patient had skin cancer.
    HadCOPD: Indicator of whether the patient had chronic obstructive pulmonary disease (COPD).
    HadDepressiveDisorder: Indicator of whether the patient was diagnosed with a depressive disorder.
    HadKidneyDisease: Indicator of whether the patient had kidney disease.
    HadArthritis: Indicator of whether the patient had arthritis.
    HadDiabetes: Indicator of whether the patient had diabetes.
    DeafOrHardOfHearing: Indicator of hearing impairment.
    BlindOrVisionDifficulty: Indicator of vision impairment.
    DifficultyConcentrating: Indicator of concentration difficulties.
    DifficultyWalking: Indicator of walking difficulties.
    DifficultyDressingBathing: Indicator of difficulties in dressing or bathing.
    DifficultyErrands: Indicator of difficulties in running errands.
    SmokerStatus: Status of whether the patient is a smoker.
    ECigaretteUsage: Indicator of e-cigarette usage.
    ChestScan: Indicator of whether the patient had a chest scan.
    RaceEthnicityCategory: Race or ethnicity of the patient.
    AlcoholDrinkers: Status of whether the patient consumes alcohol.
    HIVTesting: Status of whether the patient was tested for HIV.
    FluVaxLast12: Status of whether the patient received a flu vaccine in the last 12 months.
    PneumoVaxEver: Status of whether the patient ever received a pneumococcal vaccine.
    TetanusLast10Tdap: Status of whether the patient received a tetanus vaccine in the last 10 years.
    HighRiskLastYear: Indicator of whether the patient was at high risk in the last year.
    CovidPos: Status of whether the patient tested positive for COVID-19.
    
    
   
2. Description of the problem
=============================
From this dataset, we will train, validate, test, and use a Machine Learning model to predicte wether a given patient has a heart attach or not.
Our target variable is HadHeartAttack

The Exploratory Data Analysis Used with python libraries : Pandas, numpy, matplotlib, seaborn. 
Besides, scikitlearn is used for :
    * Setting up the validation framework
    * Feature importance 
    * Logistic Regression
        - Training et validating a small sample of the dataset
        - Training and validating full dataset
        - Using the model with all the features   
                     
        
3. Instructions on how to run the project
========================================  
    (Flask, docker, cloud deployment)       
    
