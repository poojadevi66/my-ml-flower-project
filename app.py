from flask import Flask,request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)


Model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = Model.predict(features)
    return render_template('index.html', prediction_text= 'the flower species is: {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)