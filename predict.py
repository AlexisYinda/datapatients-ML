#!/usr/bin/env python
# coding: utf-8

import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in :
     dv, model = pickle.load(f_in)


#
app = Flask('hadheartattack')

@app.route('/predict', methods=['POST'])
def predict():
  aPatient = request.get_json() 

  X = dv.transform([aPatient])
  y_pred = model.predict_proba(X)[0,1]
  hadheartattack = y_pred >= 0.5

  # pyton dictionary
  result = {
      'hadheartattack_probability': float(y_pred),
      'hadheartattack': bool(hadheartattack)
  }

  return jsonify(result)


if __name__ == "__main__" :
    app.run(debug=True, host='0.0.0.0', port=9696)



