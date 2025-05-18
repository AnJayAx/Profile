from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
import os
import joblib
import base64
import cv2
import json
from tensorflow.keras import models
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Define the Docker service URL
AI_SERVICE_URL = "http://127.0.0.1:8000"  # Use Docker service name if in same network

@app.route('/')
def home():
    return render_template('index.html', page_title="HOME", active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', page_title="ABOUT", active_page="about")

@app.route('/projects')
def projects():
    return render_template('projects.html', page_title="PROJECTS", active_page="projects")

@app.route('/contact')
def contact():
    return render_template('contact.html', page_title="CONTACT", active_page="contact")

@app.route('/caduceus', methods=['GET', 'POST'])
def caduceus():
    if request.method == 'POST':
        # Check if the request is AJAX (JSON) or form submission
        if request.is_json:
            # Get symptoms from the JSON request
            data = request.get_json()
            symptoms = data.get('symptoms', [])
            
            # Print the selected symptoms
            print("Selected symptoms via POST to /caduceus:", symptoms)
            
            return jsonify({
                'status': 'success',
                'message': 'Symptoms received',
                'symptoms_received': symptoms
            })
    
    return render_template('caduceus.html', page_title="CADUCEUS", active_page="caduceus")

@app.route('/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        print("Received symptoms for analysis:", request.json)
        response = requests.post(f"{AI_SERVICE_URL}/analyze_symptoms", json=request.json)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error connecting to AI service: {str(e)}'
        }), 500

@app.route('/autoassess')
def autoassess():
    return render_template('autoassess.html', page_title="AUTOASSESS", active_page="asutoassess")

@app.route('/gradescopev2')
def gradescopev2():
    return render_template('gradescopev2.html', page_title="GRADESCOPEV2", active_page="gradescopev2")

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Forward the image to the Docker service
    try:
        # Get the image file from the request
        image_file = request.files['image']
        
        # Forward the file to the Docker service
        files = {'image': (image_file.filename, image_file, image_file.content_type)}
        response = requests.post(f"{AI_SERVICE_URL}/analyze_image", files=files)
        
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error connecting to AI service: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)