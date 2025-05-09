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
    # Using layout.html indirectly through projects.html
    return render_template('projects.html', page_title="PROJECTS", active_page="projects")

@app.route('/contact')
def contact():
    # Using layout.html indirectly through contact.html
    return render_template('contact.html', page_title="CONTACT", active_page="contact")

if __name__ == '__main__':
    app.run(debug=True)