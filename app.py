from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)