from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib


app = Flask(__name__)

# Load the symptom severity data
csv_path = os.path.join(os.path.dirname(__file__), 'models/symptom_analyzer/Symptom-severity.csv')
symptom_severity_df = pd.read_csv(csv_path)

# Load the disease precaution data
precaution_path = os.path.join(os.path.dirname(__file__), 'models/symptom_analyzer/Disease_precaution.csv')
disease_precaution_df = pd.read_csv(precaution_path)

# Load the machine learning models
models_dir = os.path.join(os.path.dirname(__file__), 'models/symptom_analyzer')
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
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Invalid request format'}), 400
    
    # Extract symptom data from the request
    data = request.get_json()
    symptom_data = data.get('symptoms', [])
    
    # Extract just the symptom codes for processing
    symptoms = [item['code'] for item in symptom_data]
    
    # Print the selected symptoms
    print("Selected symptoms via /analyze_symptoms endpoint:", symptoms)
    
    # Create a symptom input array with the selected symptoms and padding with zeros
    max_symptoms = 17  # Maximum number of symptoms the model can handle
    symptom_input = symptoms + [0] * (max_symptoms - len(symptoms))
    symptom_input = [symptom_input]
    
    # Convert to DataFrame
    user_symptom_df = pd.DataFrame(symptom_input)
    symptom_vals = user_symptom_df.values
    
    print(f"Initial symptoms received: {symptoms}")
    print(f"Symptom input with padding: {symptom_input}")
    print(f"Initial symptom_vals shape: {symptom_vals.shape}")
    print(f"Initial symptom_vals content: {symptom_vals}")
    
    # Replace symptom codes with severity weights
    for symptom in symptoms:
        try:
            weight = symptom_severity_df[symptom_severity_df['Symptom'] == symptom]['weight'].values[0]
            print(f"Found symptom '{symptom}' with weight: {weight}")
            symptom_vals[symptom_vals == symptom] = weight
        except (IndexError, KeyError) as e:
            print(f"Error finding symptom '{symptom}': {e}")
            # If symptom not found in the severity data, assign default weight
            symptom_vals[symptom_vals == symptom] = 1
    
    print(f"Symptom_vals after weight assignment: {symptom_vals}")
    
    # Convert any remaining string values to 0
    symptom_vals = np.where(np.char.isdigit(np.array(symptom_vals).astype(str)), 
                           np.array(symptom_vals).astype(float), 0)
    
    print(f"Symptom_vals after conversion to numeric: {symptom_vals}")
    
    # Generate a response based on the symptom severity
    total_severity = np.sum(symptom_vals)
    print(f"Total severity score: {total_severity}")
    
    # Use machine learning models to predict disease
    try:
        # Make predictions using all three models
        clf_prediction = clf_model.predict(symptom_vals)[0]
        svc_prediction = svc_model.predict(symptom_vals)[0]
        xgb_prediction = xgb_model.predict(symptom_vals)[0]
        
        print(f"CLF Model Prediction: {clf_prediction}")
        print(f"SVC Model Prediction: {svc_prediction}")
        print(f"XGB Model Prediction: {xgb_prediction}")
        
        # Use a simple voting mechanism to determine the final disease
        predictions = [clf_prediction, svc_prediction, xgb_prediction]
        # Most common prediction (mode)
        from collections import Counter
        prediction_counts = Counter(predictions)
        disease = prediction_counts.most_common(1)[0][0]
        
        # Calculate confidence percentages based on model agreement
        if prediction_counts[disease] == 3:  # All models agree
            confidence = 75  # High confidence
        elif prediction_counts[disease] == 2:  # Two models agree
            confidence = 60  # Medium confidence
        else:  # All models disagree, just take the first one
            confidence = 45  # Low confidence
        
        # Get the top 3 predictions with confidence levels
        results = []
        for pred, count in prediction_counts.most_common(3):
            confidence_level = 75 if count == 3 else (60 if count == 2 else 45)
            results.append({
                "condition": pred,
                "probability": confidence_level
            })
        
        # Ensure we have at least 3 results
        while len(results) < 3:
            # Add fallback results if needed
            if len(results) == 2:
                results.append({"condition": "Other Condition", "probability": 35})
            elif len(results) == 1:
                results.append({"condition": "Secondary Condition", "probability": 40})
                results.append({"condition": "Other Condition", "probability": 35})
    
    except Exception as e:
        print(f"Error in model prediction: {e}")
        # Fallback to severity-based classification
        if total_severity > 15:
            disease = "High Severity Condition"
        elif total_severity > 10:
            disease = "Moderate Severity Condition"
        else:
            disease = "Mild Condition"
        
        results = [
            {"condition": disease, "probability": 75},
            {"condition": "Secondary Condition", "probability": 45},
            {"condition": "Other Condition", "probability": 30}
        ]
    
    # Get precautions for the predicted disease
    precautions = []
    try:
        # Extract the base disease name without any severity indicators in parentheses
        base_disease = disease.split('(')[0].strip()
        print(f"Looking for precautions for base disease: {base_disease}")
        
        # Find the row with the predicted disease
        disease_row = disease_precaution_df[disease_precaution_df['Disease'].str.contains(base_disease, case=False, na=False)]
        
        # If the disease is found in the precaution dataframe, extract precautions
        if not disease_row.empty:
            print(f"Found precautions for disease: {disease_row['Disease'].values[0]}")
            for i in range(1, 5):  # There are 4 precaution columns
                precaution_col = f'Precaution_{i}'
                if precaution_col in disease_row.columns:
                    precaution = disease_row.iloc[0][precaution_col]
                    if isinstance(precaution, str) and precaution.strip() and precaution.lower() != 'nan':
                        precautions.append(precaution.strip())
        
        if not precautions:
            precautions = [
                'Consult with a healthcare professional for specific precautions',
                'Follow general health guidelines',
                'Monitor your symptoms closely'
            ]
    except Exception as e:
        print(f"Error getting precautions: {e}")
        precautions = [
            'Consult with a healthcare professional for specific precautions',
            'Follow general health guidelines',
            'Monitor your symptoms closely'
        ]
    
    # Print detailed diagnosis information
    print("\n===== DIAGNOSIS SUMMARY =====")
    print(f"Selected symptoms: {[item['name'] for item in symptom_data]}")
    print(f"Total symptom severity: {total_severity}")
    print(f"Decision Tree (CLF) prediction: {clf_prediction if 'clf_prediction' in locals() else 'N/A'}")
    print(f"Support Vector (SVC) prediction: {svc_prediction if 'svc_prediction' in locals() else 'N/A'}")
    print(f"XGBoost prediction: {xgb_prediction if 'xgb_prediction' in locals() else 'N/A'}")
    print(f"Final diagnosis: {disease} (Confidence: {next((r['probability'] for r in results if r['condition'] == disease), 'N/A')}%)")
    print(f"Recommended precautions: {precautions}")
    print("============================\n")
    
    response = {
        'status': 'success',
        'message': 'Analysis complete',
        'disease': disease,
        'symptoms_received': [item['name'] for item in symptom_data],
        'total_severity': float(total_severity),
        'precautions': precautions,
        'results': results,
        'model_predictions': {
            'clf': clf_prediction if 'clf_prediction' in locals() else "N/A",
            'svc': svc_prediction if 'svc_prediction' in locals() else "N/A",
            'xgb': xgb_prediction if 'xgb_prediction' in locals() else "N/A"
        }
    }
    
    return jsonify(response)

@app.route('/autoassess')
def autoassess():
    return render_template('autoassess.html', page_title="AUTOASSESS", active_page="asutoassess")

@app.route('/gradescopev2')
def gradescopev2():
    return render_template('gradescopev2.html', page_title="GRADESCOPEV2", active_page="gradescopev2")

if __name__ == '__main__':
    app.run(debug=True)