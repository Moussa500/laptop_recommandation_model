from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__)
label_encoder_cpu = LabelEncoder()
label_encoder_gpu = LabelEncoder()
label_encoder_os = LabelEncoder()
label_encoder_company = LabelEncoder()
model = None
def initialize_model():
    global model, label_encoder_cpu, label_encoder_gpu, label_encoder_os, label_encoder_company
    data = pd.read_csv(r'C:\dev\ml\project\app\dataframe2.csv')
    label_encoder_cpu.fit(data['Cpu_brand'])
    label_encoder_gpu.fit(data['Gpu_brand'])
    label_encoder_os.fit(data['os'])
    label_encoder_company.fit(data['Company'])
    data['Cpu_brand'] = label_encoder_cpu.transform(data['Cpu_brand'])
    data['Gpu_brand'] = label_encoder_gpu.transform(data['Gpu_brand'])
    data['os'] = label_encoder_os.transform(data['os'])
    data['Company'] = label_encoder_company.transform(data['Company'])
    X = data[['G', 'MD', 'W', 'AI', 'S', 'P', 'os', 'Company', 'Touchscreen']]
    Y = data[['Ram', 'Gpu_brand', 'Cpu_brand']]
    forest = RandomForestClassifier(random_state=1, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    model = MultiOutputClassifier(forest, n_jobs=-1)
    model.fit(X, Y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_data = pd.DataFrame({
            'G': [data['G']],
            'MD': [data['MD']],
            'W': [data['W']],
            'AI': [data['AI']],
            'S': [data['S']],
            'P': [data['P']],
            'os': [label_encoder_os.transform([data['os']])[0]],
            'Company': [label_encoder_company.transform([data['company']])[0]],
            'Touchscreen': [data['touchscreen']],
        })
        
        predictions = model.predict(input_data)
        
        ram = predictions[0][0]
        gpu_brand = label_encoder_gpu.inverse_transform([predictions[0][1]])[0]
        cpu_brand = label_encoder_cpu.inverse_transform([predictions[0][2]])[0]
        
        return jsonify({
            'ram': int(ram),
            'gpu_brand': gpu_brand,
            'cpu_brand': cpu_brand
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)