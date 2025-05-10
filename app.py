from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
import base64
import cv2
import json
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tempfile

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

# Find skin condition model and class indices
def find_skin_model():
    models_dir = os.path.join(os.path.dirname(__file__), 'models/skin_condition')
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.h5')]
    if not model_files:
        print("No trained skin condition models found.")
        return None, None
    
    # Get the most recent model
    most_recent_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, most_recent_model)
    
    # Find matching class indices file
    model_prefix = most_recent_model.replace('_best.h5', '')
    class_indices_file = f"{model_prefix}_class_indices.json"
    class_indices_path = os.path.join(models_dir, class_indices_file)
    
    if not os.path.exists(class_indices_path):
        # Try to find any class indices file
        json_files = [f for f in os.listdir(models_dir) if f.endswith('_class_indices.json')]
        if json_files:
            class_indices_path = os.path.join(models_dir, json_files[0])
            print(f"Using class indices from: {json_files[0]}")
        else:
            print("No class indices file found.")
            return model_path, None
    
    print(f"Using skin condition model: {most_recent_model}")
    return model_path, class_indices_path

# Load skin condition model and class indices (global for efficiency)
SKIN_MODEL_PATH, CLASS_INDICES_PATH = find_skin_model()
if SKIN_MODEL_PATH and CLASS_INDICES_PATH:
    try:
        SKIN_MODEL = load_model(SKIN_MODEL_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            CLASS_INDICES = json.load(f)
        print("Skin condition model loaded successfully")
    except Exception as e:
        print(f"Error loading skin condition model: {e}")
        SKIN_MODEL = None
        CLASS_INDICES = None
else:
    SKIN_MODEL = None
    CLASS_INDICES = None

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

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files and 'image_data' not in request.form:
        print("No image found in request")
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = None
        
        # Log the request details
        print("\n===== IMAGE ANALYSIS REQUEST =====")
        print(f"Request method: {request.method}")
        print(f"Files in request: {list(request.files.keys()) if request.files else 'None'}")
        print(f"Form keys in request: {list(request.form.keys()) if request.form else 'None'}")
        
        if 'image' in request.files:
            # Handle actual file upload
            file = request.files['image']
            print(f"Received file: {file.filename}")
            
            if file.filename == '':
                print("Empty filename provided")
                return jsonify({'status': 'error', 'message': 'No file selected'}), 400
            
            # Save file to temp location
            temp_file_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_file_path)
            print(f"Saved file to: {temp_file_path}")
            
        elif 'image_data' in request.form:
            # Handle base64 encoded image
            print("Received base64 image data")
            image_data = request.form['image_data']
            
            try:
                if ',' in image_data:
                    # Remove the data URL prefix if present
                    print("Stripping data URL prefix")
                    image_data = image_data.split(',', 1)[1]
                
                # Decode the base64 string
                image_bytes = base64.b64decode(image_data)
                print(f"Decoded base64 data, size: {len(image_bytes)} bytes")
                
                # Save to temporary file
                temp_file_path = os.path.join(temp_dir, 'uploaded_image.jpg')
                with open(temp_file_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"Saved decoded image to: {temp_file_path}")
            except Exception as e:
                print(f"Error processing base64 image: {e}")
                return jsonify({
                    'status': 'error', 
                    'message': f'Error processing image data: {str(e)}'
                }), 400
        
        # Check if the file was successfully saved
        if temp_file_path is None or not os.path.exists(temp_file_path):
            print("Failed to save image file")
            return jsonify({
                'status': 'error',
                'message': 'Failed to process the uploaded image'
            }), 500
        
        # Check if model is available
        if SKIN_MODEL is None or CLASS_INDICES is None:
            print("Skin condition model not available")
            return jsonify({
                'status': 'error',
                'message': 'Skin condition model not available'
            }), 500
        
        # Verify the image file can be opened and is valid
        try:
            test_img = cv2.imread(temp_file_path)
            if test_img is None or test_img.size == 0:
                print(f"Invalid image file: Could not read image data from {temp_file_path}")
                return jsonify({
                    'status': 'error',
                    'message': 'The uploaded file is not a valid image'
                }), 400
        except Exception as e:
            print(f"Error validating image: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Error validating image: {str(e)}'
            }), 400
        
        # Predict using the model
        print("Starting skin condition prediction")
        result = predict_skin_condition(temp_file_path)
        print(f"Prediction result: {result}")
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
            print("Cleaned up temporary files")
        except Exception as e:
            print(f"Warning: Failed to clean up temp files: {e}")
        
        print("===== END IMAGE ANALYSIS =====\n")
        
        # Use a fallback if prediction failed
        if result.get('status') == 'error':
            print("Using fallback response due to prediction error")
            return jsonify({
                'status': 'success',
                'primary_condition': 'Unknown Skin Condition',
                'confidence': 45.0,
                'severity': 'moderate',
                'top_predictions': [
                    {"condition": "Unknown Skin Condition", "confidence": 45.0}
                ],
                'recommendations': [
                    "Please consult with a dermatologist for proper diagnosis",
                    "Keep the affected area clean and dry",
                    "Try uploading a clearer image for better results"
                ],
                'message': f"Analysis completed with limited confidence: {result.get('message')}"
            })
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Unhandled error in analyze_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f'Error analyzing image: {str(e)}'
        }), 500

def predict_skin_condition(image_path):
    """Predict skin condition from an image using the loaded model"""
    try:
        print(f"Starting prediction for image: {image_path}")
        
        # Verify the class indices
        if not CLASS_INDICES:
            print("Error: CLASS_INDICES is empty or None")
            return {'status': 'error', 'message': 'Class indices not available'}
            
        # Reverse the indices to get class names from indices
        class_names = {v: k for k, v in CLASS_INDICES.items()}
        print(f"Available classes: {list(class_names.values())}")
        
        # Read and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image from {image_path}")
            return {'status': 'error', 'message': 'Could not read image file'}
            
        print(f"Original image shape: {img.shape}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
        print(f"Resized image shape: {img_resized.shape}")
        
        img_normalized = img_resized / 255.0  # Normalize pixel values
        img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
        print(f"Input batch shape: {img_batch.shape}")
        
        # Make prediction
        print("Running model prediction...")
        predictions = SKIN_MODEL.predict(img_batch)
        print(f"Raw prediction output shape: {predictions.shape}")
        
        predicted_class_idx = np.argmax(predictions[0])
        print(f"Predicted class index: {predicted_class_idx}")
        
        # Check if the predicted index exists in our class names
        if predicted_class_idx not in class_names:
            print(f"Error: Predicted index {predicted_class_idx} not found in class names")
            return {
                'status': 'error', 
                'message': f'Model predicted an unknown class index: {predicted_class_idx}'
            }
        
        predicted_class = class_names[predicted_class_idx]
        print(f"Predicted class name: {predicted_class}")
        
        confidence = float(predictions[0][predicted_class_idx] * 100)
        print(f"Confidence: {confidence:.2f}%")
        
        # Print detailed prediction information
        print("\n===== SKIN CONDITION PREDICTION =====")
        print(f"Predicted class: {predicted_class} (Index: {predicted_class_idx})")
        print(f"Confidence: {confidence:.2f}%")
        
        # Get and print top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        
        top_predictions = []
        for i, idx in enumerate(top_indices, 1):
            if idx in class_names:
                class_name = class_names[idx]
                class_confidence = float(predictions[0][idx] * 100)
                print(f"  {i}. {class_name}: {class_confidence:.2f}%")
                top_predictions.append({
                    "condition": class_name, 
                    "confidence": class_confidence
                })
        
        # Determine severity and print it
        severity = determine_severity(predicted_class, confidence)
        print(f"\nDetermined severity: {severity}")
        
        # Get recommendations
        recommendations = get_recommendations(predicted_class)
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("=====================================\n")
        
        return {
            'status': 'success',
            'primary_condition': predicted_class,
            'confidence': confidence,
            'severity': severity,
            'top_predictions': top_predictions,
            'recommendations': recommendations
        }
    
    except Exception as e:
        print(f"Error in predict_skin_condition: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }

def determine_severity(condition, confidence):
    """Determine severity level based on condition and confidence"""
    # Default to moderate
    severity = "moderate"
    
    # Higher confidence often correlates with more severe cases
    if confidence > 85:
        severity = "severe"
    elif confidence < 60:
        severity = "mild"
    
    # Adjust based on specific conditions (simplified)
    condition_lower = condition.lower()
    if "acne" in condition_lower:
        # Acne might be classified by levels
        if "severe" in condition_lower:
            severity = "severe"
        elif "mild" in condition_lower:
            severity = "mild"
    
    return severity

def get_recommendations(condition):
    """Get recommendations based on the skin condition"""
    # Default recommendations
    default_recommendations = [
        "Consult a dermatologist for proper diagnosis and treatment",
        "Keep the affected area clean and dry",
        "Avoid scratching or picking at the affected area"
    ]
    
    # Condition-specific recommendations
    condition_lower = condition.lower()
    
    if "acne" in condition_lower:
        return [
            "Use non-comedogenic skincare products",
            "Wash face twice daily with a gentle cleanser",
            "Consider benzoyl peroxide or salicylic acid treatments",
            "Avoid touching your face and picking at spots",
            "Consult a dermatologist for prescription options if severe"
        ]
    elif "eczema" in condition_lower:
        return [
            "Use moisturizers regularly to prevent dry skin",
            "Take lukewarm (not hot) showers or baths",
            "Use mild, fragrance-free soaps and detergents",
            "Apply prescribed corticosteroid creams as directed",
            "Identify and avoid triggers that worsen symptoms"
        ]
    elif "psoriasis" in condition_lower:
        return [
            "Keep skin moisturized to reduce scaling",
            "Use medicated shampoos for scalp psoriasis",
            "Get moderate sun exposure (with doctor's approval)",
            "Avoid injuries to skin and stress when possible",
            "Follow treatment plan as directed by your doctor"
        ]
    elif "dermatitis" in condition_lower:
        return [
            "Identify and avoid allergens and irritants",
            "Apply cool compresses to reduce itching",
            "Use prescribed anti-inflammatory creams",
            "Keep affected areas clean and moisturized",
            "Wear cotton clothes and avoid harsh fabrics"
        ]
    
    return default_recommendations

if __name__ == '__main__':
    app.run(debug=True)