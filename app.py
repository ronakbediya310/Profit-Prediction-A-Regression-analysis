from flask import Flask, render_template, request, jsonify
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData
import os
import sys

app = Flask(__name__)
pipeline = PredictionPipeline()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('form.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            rd_spend = float(request.form.get('rd_spend'))
            administration = float(request.form.get('administration'))
            marketing_spend = float(request.form.get('marketing_spend'))
            state = request.form.get('state')
            
            data = CustomData(
                rd_spend=rd_spend,
                administration=administration,
                marketing_spend=marketing_spend,
                state=state
            )
            
            logging.info("New data received for prediction:")
            # logging.info(data.get_data_as_dataframe())
            
            pred = pipeline.predict(data.get_data_as_dataframe())
            result =round(pred[0], 2)
            logging.info("Profit Predicted")
            logging.info(result)
            return render_template('result.html', result=result)
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
