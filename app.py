from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load the symptom severity data
csv_path = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser/Symptom-severity.csv')
symptom_severity_df = pd.read_csv(csv_path)

# Load the disease precaution data
precaution_path = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser/Disease_precaution.csv')
disease_precaution_df = pd.read_csv(precaution_path)

# Load the machine learning models
models_dir = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser')
clf_model = joblib.load(os.path.join(models_dir, 'CLF_model.sav'))
svc_model = joblib.load(os.path.join(models_dir, 'SVC_model.sav'))
xgb_model = joblib.load(os.path.join(models_dir, 'XGB_model.sav'))

@app.route('/')
def home():
    return render_template('index.html', active_page='home')

@app.route('/projects')
def projects():
    return render_template('projects.html', page_title="PROJECTS", active_page="projects")

@app.route('/contact')
def contact():
    return render_template('contact.html', page_title="CONTACT", active_page="contact")

@app.route('/caduceus')
def caduceus():
    return render_template('caduceus.html', page_title="CADUCEUS", active_page="caduceus")

@app.route('/autoassess')
def autoassess():
    return render_template('autoassess.html', page_title="AUTOASSESS", active_page="asutoassess")

@app.route('/gradescopev2')
def gradescopev2():
    return render_template('gradescopev2.html', page_title="GRADESCOPEV2", active_page="gradescopev2")

if __name__ == '__main__':
    app.run(debug=True)