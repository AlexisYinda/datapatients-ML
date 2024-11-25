#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'
patient_id = 'pat-6654'

aPatient = {
  'State': 'Alabama',
  'Sex': 'Male',
  'GeneralHealth': 'Good',
  'AgeCategory': 'Age 40 to 44',
  'HeightInMeters': 1.90000004768372,
  'WeightInKilograms': 110.0199966430664,
  'BMI': 26.9400005340576,
  'HadAngina': 1,
  'HadStroke': 0,
  'HadAsthma': 1,
  'HadSkinCancer': 0,
  'HadCOPD': 0,
  'HadDepressiveDisorder': 0,
  'HadKidneyDisease': 0,
  'HadArthritis':0,
  'HadDiabetes': 'No',
  'DeafOrHardOfHearing': 1,
  'BlindOrVisionDifficulty': 0,
  'DifficultyConcentrating': 0,
  'DifficultyWalking': 1,
  'DifficultyDressingBathing': 0,
  'DifficultyErrands': 0,
  'SmokerStatus': 'Former smoker',
  'ECigaretteUsage': 'Never used e-cigarettes in my entire life',
  'ChestScan': 1,
  'RaceEthnicityCategory': 'White only, Non-Hispanic',
  'AlcoholDrinkers': 0,
  'HIVTesting': 1,
  'FluVaxLast12': 0,
  'PneumoVaxEver': 1,
  'TetanusLast10Tdap': 'No, did not receive any tetanus shot in the past 10 years',
  'HighRiskLastYear': 1,
  'CovidPos': 1}


response = requests.post(url, json=aPatient).json()
print(response)



if response['hadheartattack'] == True:
    print('The patient with patientID %s has Heart attack' % patient_id)
else:
    print("The patient with patientID %s don't have Heart attack" % patient_id)
