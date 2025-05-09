from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)