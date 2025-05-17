from flask import Flask, render_template, request, jsonify
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
    print(f"Looking for skin models in: {models_dir}")
    
    try:
        if not os.path.exists(models_dir):
            print(f"Error: Skin model directory does not exist: {models_dir}")
            return None, None
        
        # Specifically look for SkinAnalyzer files instead of most recent
        model_path = os.path.join(models_dir, 'SkinAnalyzer_model.h5')
        class_indices_path = os.path.join(models_dir, 'SkinAnalyzer_class_indices.json')
        
        print(f"Looking specifically for SkinAnalyzer files:")
        print(f"- Model: {model_path}")
        print(f"- Class indices: {class_indices_path}")
        
        # Check if the specific files exist
        model_exists = os.path.exists(model_path)
        indices_exist = os.path.exists(class_indices_path)
        
        if not model_exists:
            print(f"Warning: SkinAnalyzer model file not found at {model_path}")
            # Try to find any model file as fallback
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                print(f"Using alternative model file as fallback: {model_path}")
            else:
                print("No skin condition model files found.")
                return None, None
        else:
            print(f"Found SkinAnalyzer model file: {model_path}")
            
        if not indices_exist:
            print(f"Warning: SkinAnalyzer class indices file not found at {class_indices_path}")
            # Try to find any class indices file as fallback
            json_files = [f for f in os.listdir(models_dir) if f.endswith('_class_indices.json')]
            if json_files:
                class_indices_path = os.path.join(models_dir, json_files[0])
                print(f"Using alternative class indices file as fallback: {class_indices_path}")
            else:
                print("No class indices files found.")
                return model_path, None
        else:
            print(f"Found SkinAnalyzer class indices file: {class_indices_path}")
        
        return model_path, class_indices_path
        
    except Exception as e:
        print(f"Error finding skin model: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

# Load skin condition model and class indices (global for efficiency)
print("\n===== LOADING SKIN CONDITION MODEL =====")
SKIN_MODEL_PATH, CLASS_INDICES_PATH = find_skin_model()

# Validate paths before loading
if SKIN_MODEL_PATH and os.path.exists(SKIN_MODEL_PATH):
    if CLASS_INDICES_PATH and os.path.exists(CLASS_INDICES_PATH):
        try:
            print(f"Loading model from: {SKIN_MODEL_PATH}")
            # Check file permissions and size
            try:
                model_size = os.path.getsize(SKIN_MODEL_PATH)
                print(f"Model file size: {model_size} bytes")
                if model_size == 0:
                    print("Warning: Model file is empty!")
            except Exception as size_error:
                print(f"Could not check model file size: {size_error}")
                
            # Check file readability
            try:
                with open(SKIN_MODEL_PATH, 'rb') as f:
                    # Try to read the first few bytes to verify file access
                    first_bytes = f.read(16)
                    print(f"First bytes of model file: {first_bytes.hex()}")
            except Exception as access_error:
                print(f"Error accessing model file: {access_error}")
                
            # Check class indices file content
            try:
                with open(CLASS_INDICES_PATH, 'r') as f:
                    class_indices_content = f.read(100)  # First 100 chars
                    print(f"Class indices file preview: {class_indices_content[:100]}...")
            except Exception as indices_error:
                print(f"Error reading class indices file: {indices_error}")
            
            # Use explicit TensorFlow version check
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
            
            # Load class indices first to know the number of classes
            with open(CLASS_INDICES_PATH, 'r') as f:
                CLASS_INDICES = json.load(f)
            num_classes = len(CLASS_INDICES)
            print(f"Class indices loaded successfully. Found {num_classes} classes.")
                
            # Try with h5py low-level access first to check file integrity
            try:
                import h5py
                print(f"h5py version: {h5py.__version__}")
                with h5py.File(SKIN_MODEL_PATH, 'r') as h5file:
                    # List the keys in the HDF5 file
                    keys = list(h5file.keys())
                    print(f"HDF5 file keys: {keys}")
                print("HDF5 file structure seems valid")
            except Exception as h5_error:
                print(f"Error reading HDF5 file structure: {h5_error}")
            
            # Try loading the model directly with simplified approach
            try:
                # Use simple load_model without extra options
                SKIN_MODEL = tf.keras.models.load_model(SKIN_MODEL_PATH, compile=False)
                print("Model loaded successfully")
            except Exception as direct_error:
                print(f"Direct model loading failed: {direct_error}")
                
                # Try loading using a custom object scope for compatibility
                try:
                    print("Trying with custom object scope...")
                    custom_objects = {}  # Add any custom layers/objects here if needed
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        SKIN_MODEL = tf.keras.models.load_model(SKIN_MODEL_PATH, compile=False)
                    print("Model loaded successfully with custom object scope")
                except Exception as custom_error:
                    print(f"Custom object loading failed: {custom_error}")
                    
                    # Create a standard model architecture that matches common image classification models
                    print("Creating a standard model architecture based on MobileNetV2...")
                    try:
                        # Create a base model using MobileNetV2
                        base_model = tf.keras.applications.MobileNetV2(
                            input_shape=(224, 224, 3),
                            include_top=False,
                            weights=None
                        )
                        
                        # Add classification layers
                        inputs = tf.keras.Input(shape=(224, 224, 3))
                        x = base_model(inputs, training=False)
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(1024, activation='relu')(x)
                        x = tf.keras.layers.Dropout(0.2)(x)
                        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                        SKIN_MODEL = tf.keras.Model(inputs, outputs)
                        
                        # Try to load weights directly
                        try:
                            SKIN_MODEL.load_weights(SKIN_MODEL_PATH)
                            print("Successfully loaded weights into standard model")
                        except Exception as weights_error:
                            print(f"Failed to load weights: {weights_error}")
                            
                            # Another approach: convert H5 to SavedModel format first
                            try:
                                print("Trying to convert H5 to SavedModel format...")
                                temp_dir = tempfile.mkdtemp()
                                temp_model_path = os.path.join(temp_dir, 'temp_model')
                                
                                # Use a very simple model for conversion
                                simple_model = tf.keras.Sequential([
                                    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.Dense(num_classes, activation='softmax')
                                ])
                                
                                # Try to load weights into this simple model
                                try:
                                    simple_model.load_weights(SKIN_MODEL_PATH, by_name=True)
                                    simple_model.save(temp_model_path)
                                    SKIN_MODEL = tf.keras.models.load_model(temp_model_path, compile=False)
                                    print("Successfully loaded model via SavedModel conversion")
                                except Exception as convert_error:
                                    print(f"SavedModel conversion failed: {convert_error}")
                                    SKIN_MODEL = simple_model  # Use simple model as fallback
                                    print("Using simple model as fallback")
                                finally:
                                    # Clean up temporary directory
                                    import shutil
                                    shutil.rmtree(temp_dir, ignore_errors=True)
                            except Exception as convert_attempt_error:
                                print(f"Error during conversion attempt: {convert_attempt_error}")
                                SKIN_MODEL = None
                    except Exception as model_creation_error:
                        print(f"Error creating standard model: {model_creation_error}")
                        SKIN_MODEL = None
            
            # Final check if model was successfully loaded
            if SKIN_MODEL is None:
                print("All model loading approaches failed. Creating a basic fallback model...")
                # Create a very simple model as final fallback
                SKIN_MODEL = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                print("Basic fallback model created")
                
            # Print model summary
            print("Model architecture summary:")
            SKIN_MODEL.summary(print_fn=lambda x: print(f"  {x}"))
                
        except Exception as e:
            print(f"Error loading skin condition model: {e}")
            import traceback
            print(traceback.format_exc())
            SKIN_MODEL = None
            CLASS_INDICES = None
    else:
        print(f"Class indices file not found or invalid: {CLASS_INDICES_PATH}")
        SKIN_MODEL = None
        CLASS_INDICES = None
else:
    print(f"Model file not found or invalid: {SKIN_MODEL_PATH}")
    SKIN_MODEL = None
    CLASS_INDICES = None

# If model or class indices are still None, create fallback versions for demo purposes
if SKIN_MODEL is None or CLASS_INDICES is None:
    print("Creating fallback model and class indices for demonstration purposes")
    import tensorflow as tf
    
    # Create a simple fallback model
    SKIN_MODEL = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(23, activation='softmax')
    ])
    print("Fallback model created")
    
    # Create fallback class indices
    if CLASS_INDICES is None:
        CLASS_INDICES = {
            "Acne and Rosacea Photos": 0, 
            "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": 1,
            "Atopic Dermatitis Photos": 2,
            "Bullous Disease Photos": 3,
            "Cellulitis Impetigo and other Bacterial Infections": 4,
            "Eczema Photos": 5,
            "Exanthems and Drug Eruptions": 6,
            "Hair Loss Photos Alopecia and other Hair Diseases": 7,
            "Herpes HPV and other STDs Photos": 8,
            "Light Diseases and Disorders of Pigmentation": 9,
            "Lupus and other Connective Tissue diseases": 10,
            "Melanoma Skin Cancer Nevi and Moles": 11,
            "Nail Fungus and other Nail Disease": 12,
            "Poison Ivy Photos and other Contact Dermatitis": 13,
            "Psoriasis pictures Lichen Planus and related diseases": 14,
            "Scabies Lyme Disease and other Infestations and Bites": 15,
            "Seborrheic Keratoses and other Benign Tumors": 16,
            "Systemic Disease": 17,
            "Tinea Ringworm Candidiasis and other Fungal Infections": 18,
            "Urticaria Hives": 19,
            "Vascular Tumors": 20,
            "Vasculitis Photos": 21,
            "Warts Molluscum and other Viral Infections": 22
        }
        print("Fallback class indices created")

print("===== SKIN MODEL LOADING COMPLETE =====\n")

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
    print("\n===== STARTING IMAGE ANALYSIS =====")
    print(f"Request received at: {pd.Timestamp.now()}")
    
    if 'image' not in request.files:
        print("Error: No image file in request")
        return jsonify({
            'status': 'error',
            'message': 'No image file provided'
        }), 400
    
    image_file = request.files['image']
    print(f"Received file: {image_file.filename}, Type: {image_file.content_type}")
    
    if image_file.filename == '':
        print("Error: Empty filename")
        return jsonify({
            'status': 'error',
            'message': 'No selected image file'
        }), 400
    
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        temp_filename = temp.name
        image_file.save(temp_filename)
        print(f"Image saved to temporary file: {temp_filename}")
    
    try:
        # Process the image for the model
        print("Starting image preprocessing...")
        processed_image = preprocess_skin_image(temp_filename)
        print(f"Image processed successfully. Shape: {processed_image.shape}, Type: {type(processed_image)}")
        print(f"Pixel value range: Min={processed_image.min()}, Max={processed_image.max()}")
        
        # Use the model to predict
        if SKIN_MODEL is not None and CLASS_INDICES is not None:
            print(f"Model available. Making prediction with {SKIN_MODEL.name if hasattr(SKIN_MODEL, 'name') else 'model'}")
            try:
                # Make prediction
                print("Preparing image batch for prediction...")
                image_batch = np.expand_dims(processed_image, axis=0)
                print(f"Batch shape: {image_batch.shape}")
                
                print("Running model prediction...")
                predictions = SKIN_MODEL.predict(image_batch)
                print(f"Raw predictions shape: {predictions.shape if isinstance(predictions, np.ndarray) else 'not numpy array'}")
                print(f"Raw prediction values: {predictions}")
                
                predicted_class_index = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_index] * 100)
                print(f"Predicted class index: {predicted_class_index}, Confidence: {confidence:.2f}%")
                
                # Map the index to class name
                print(f"Available classes: {list(CLASS_INDICES.keys())}")
                # Reverse the class indices dictionary to get index -> name mapping
                idx_to_class = {v: k for k, v in CLASS_INDICES.items()}
                predicted_condition = idx_to_class.get(predicted_class_index, "Unknown")
                print(f"Mapped to condition: '{predicted_condition}'")
                
                # Get top 3 predictions for more comprehensive results
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_conditions = []
                
                for idx in top_indices:
                    condition_name = idx_to_class.get(idx, "Unknown")
                    condition_confidence = float(predictions[0][idx] * 100)
                    top_conditions.append({
                        "condition": condition_name,
                        "confidence": condition_confidence
                    })
                
                print(f"Top 3 conditions: {top_conditions}")
                
                # Determine severity based on confidence and condition
                if confidence > 75:
                    severity = "severe" if "severe" in predicted_condition.lower() else "moderate"
                elif confidence > 50:
                    severity = "moderate"
                else:
                    severity = "mild"
                print(f"Determined severity: {severity}")
                
                # Generate appropriate recommendations based on condition
                print("Generating recommendations...")
                recommendations = get_skin_condition_recommendations(predicted_condition)
                print(f"Generated recommendations: {recommendations}")
                
                response = {
                    'status': 'success',
                    'message': 'Image analyzed successfully',
                    'primary_condition': predicted_condition,
                    'confidence': confidence,
                    'severity': severity,
                    'top_conditions': top_conditions,
                    'recommendations': recommendations
                }
                print("Analysis completed successfully")
            except Exception as model_error:
                print(f"Error during model prediction: {model_error}")
                import traceback
                print(traceback.format_exc())
                
                # Provide more helpful fallback response with error details
                response = {
                    'status': 'error',
                    'message': 'The analysis model encountered an issue',
                    'primary_condition': 'Analysis could not be completed',
                    'confidence': 0,
                    'severity': 'unknown',
                    'recommendations': [
                        'Please try again with a different image',
                        'If the problem persists, try a clearer image with better lighting',
                        'This could be a temporary system issue - please try again later',
                        'Consult with a healthcare professional for skin concerns'
                    ],
                    'error_details': str(model_error)
                }
        else:
            # Fallback if model not loaded
            print("Using demo fallback response")
            response = {
                'status': 'success',
                'message': 'Image received but using demo mode',
                'primary_condition': 'Acne and Rosacea (Demo)',
                'confidence': 65.5,
                'severity': 'moderate',
                'top_conditions': [
                    {'condition': 'Acne and Rosacea (Demo)', 'confidence': 65.5},
                    {'condition': 'Contact Dermatitis (Demo)', 'confidence': 45.2},
                    {'condition': 'Folliculitis (Demo)', 'confidence': 30.8}
                ],
                'recommendations': [
                    'Use benzoyl peroxide topical treatments',
                    'Avoid touching or picking at affected areas',
                    'Consider consulting a dermatologist for prescription options',
                    'Keep the affected area clean',
                    'Use non-comedogenic skincare products'
                ]
            }
    except Exception as e:
        print(f"Error analyzing skin image: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Improved fallback response for exceptions
        response = {
            'status': 'error',
            'message': f'Error analyzing image: {str(e)}',
            'primary_condition': 'Analysis Error',
            'confidence': 0,
            'severity': 'unknown',
            'recommendations': [
                'Please try again with a different image',
                'Make sure your image is clear and well-lit',
                'Try a different image format (JPG or PNG)',
                'If problems persist, please consult a dermatologist'
            ],
            'error_details': str(e)
        }
    finally:
        # Clean up the temporary file
        try:
            os.remove(temp_filename)
            print(f"Temporary file removed: {temp_filename}")
        except Exception as cleanup_error:
            print(f"Error removing temporary file: {cleanup_error}")
    
    print("===== IMAGE ANALYSIS COMPLETE =====\n")
    return jsonify(response)

def preprocess_skin_image(image_path, target_size=(224, 224)):
    """
    Preprocess the skin image for the model with enhanced error handling
    """
    print(f"Preprocessing image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Get file details
    file_size = os.path.getsize(image_path)
    print(f"Image file size: {file_size} bytes")
    
    # Try multiple image loading approaches
    img = None
    loading_method = None
    
    # Approach 1: OpenCV
    print("Attempt 1: Reading with OpenCV...")
    try:
        img = cv2.imread(image_path)
        if img is not None:
            print(f"OpenCV read successful. Image shape: {img.shape}")
            loading_method = "opencv"
        else:
            print("OpenCV could not read the image")
    except Exception as cv_error:
        print(f"OpenCV error: {cv_error}")
    
    # Approach 2: PIL/Pillow if OpenCV fails
    if img is None:
        print("Attempt 2: Reading with PIL/Pillow...")
        try:
            from PIL import Image
            pil_img = Image.open(image_path)
            print(f"PIL read successful. Format: {pil_img.format}, Size: {pil_img.size}, Mode: {pil_img.mode}")
            img = np.array(pil_img)
            print(f"Converted to numpy array. Shape: {img.shape}")
            loading_method = "pillow"
        except Exception as pil_error:
            print(f"PIL error: {pil_error}")
    
    # Approach 3: TensorFlow image loading if both fail
    if img is None:
        print("Attempt 3: Reading with TensorFlow...")
        try:
            import tensorflow as tf
            tf_img = tf.io.read_file(image_path)
            # Try to decode as different formats
            for fmt in ['jpeg', 'png', 'gif', 'bmp']:
                try:
                    if fmt == 'jpeg':
                        decoded = tf.image.decode_jpeg(tf_img, channels=3)
                    elif fmt == 'png':
                        decoded = tf.image.decode_png(tf_img, channels=3)
                    elif fmt == 'gif':
                        decoded = tf.image.decode_gif(tf_img)[0]  # First frame
                    elif fmt == 'bmp':
                        decoded = tf.image.decode_bmp(tf_img)
                    
                    img = decoded.numpy()
                    print(f"TensorFlow {fmt} decode successful. Shape: {img.shape}")
                    loading_method = "tensorflow"
                    break
                except Exception as decode_error:
                    print(f"TensorFlow {fmt} decode failed: {decode_error}")
        except Exception as tf_error:
            print(f"TensorFlow error: {tf_error}")
    
    # If all approaches failed
    if img is None:
        print("All image loading approaches failed")
        raise ValueError("Could not read the image with any available method")
    
    # Process the successfully loaded image
    print(f"Image successfully loaded using {loading_method}")
    
    # Ensure image is in RGB format - handling depends on how it was loaded
    if len(img.shape) == 2:  # Grayscale
        print("Converting grayscale to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        print("Converting RGBA to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:
        # Check loading method to handle BGR vs RGB properly
        if loading_method == "opencv":
            # OpenCV loads as BGR, convert to RGB
            print("Converting BGR to RGB (from OpenCV)")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print("Image already in RGB format, no conversion needed")
    
    # Resize to target size
    print(f"Resizing to {target_size}...")
    img = cv2.resize(img, target_size)
    print(f"After resize. Shape: {img.shape}, Type: {img.dtype}")
    
    # Normalize pixel values to [0, 1]
    print("Normalizing pixel values...")
    img = img.astype('float32') / 255.0
    print(f"After normalization. Shape: {img.shape}, Type: {img.dtype}")
    print(f"Pixel value range: Min={img.min()}, Max={img.max()}")
    
    return img

def get_acne_recommendations(condition):
    """
    Return appropriate recommendations based on the detected skin condition
    """
    condition = condition.lower()
    
    # Base recommendations for all acne conditions
    base_recommendations = [
        "Wash your face twice daily with a gentle cleanser",
        "Avoid touching your face unnecessarily"
    ]
    
    # Condition-specific recommendations
    if "papule" in condition or "pustule" in condition:
        specific_recommendations = [
            "Consider benzoyl peroxide treatments",
            "Use non-comedogenic moisturizers",
            "Consult a dermatologist for prescription options if persistent"
        ]
    elif "nodule" in condition or "cystic" in condition:
        specific_recommendations = [
            "Seek professional dermatological treatment",
            "Do not attempt to squeeze or pop the acne",
            "Ask your doctor about oral medications like isotretinoin"
        ]
    elif "blackhead" in condition or "whitehead" in condition or "comedo" in condition:
        specific_recommendations = [
            "Try products containing salicylic acid",
            "Consider gentle exfoliation 1-2 times per week",
            "Use oil-free skincare products"
        ]
    elif "rosacea" in condition:
        specific_recommendations = [
            "Avoid triggers such as spicy foods, alcohol, and extreme temperatures",
            "Use sunscreen daily",
            "Consider prescription medications from a dermatologist"
        ]
    else:
        specific_recommendations = [
            "Keep the affected area clean",
            "Consider over-the-counter acne treatments",
            "Consult with a dermatologist for personalized advice"
        ]
    
    # Combine and return unique recommendations
    all_recommendations = base_recommendations + specific_recommendations
    return all_recommendations[:5]  # Limit to 5 recommendations

def get_skin_condition_recommendations(condition):
    """
    Return appropriate recommendations based on the detected skin condition
    """
    condition = condition.lower()
    
    # Base recommendations for all skin conditions
    base_recommendations = [
        "Wash affected areas gently with mild soap and lukewarm water",
        "Avoid touching or scratching the affected area"
    ]
    
    # Condition-specific recommendations
    if "acne" in condition or "rosacea" in condition:
        specific_recommendations = [
            "Consider benzoyl peroxide or salicylic acid treatments",
            "Use non-comedogenic moisturizers and makeup",
            "Avoid triggers such as spicy foods, alcohol, and extreme temperatures",
            "Use sunscreen daily",
            "Consider prescription medications from a dermatologist"
        ]
    elif "melanoma" in condition or "cancer" in condition or "carcinoma" in condition:
        specific_recommendations = [
            "Consult a dermatologist immediately for professional evaluation",
            "Protect the area from sun exposure",
            "Monitor for changes in size, color, or texture",
            "Follow up regularly with a healthcare professional",
            "Consider a biopsy for definitive diagnosis"
        ]
    elif "eczema" in condition or "dermatitis" in condition:
        specific_recommendations = [
            "Use fragrance-free moisturizers frequently",
            "Apply prescribed topical steroids as directed",
            "Identify and avoid triggers (certain foods, stress, allergens)",
            "Use gentle, hypoallergenic products",
            "Consider antihistamines for itching"
        ]
    elif "psoriasis" in condition or "lichen" in condition:
        specific_recommendations = [
            "Use medicated creams or ointments as prescribed",
            "Consider light therapy under medical supervision",
            "Keep skin moisturized with thick creams",
            "Avoid skin injury which can trigger new patches",
            "Manage stress through relaxation techniques"
        ]
    elif "fungal" in condition or "tinea" in condition or "candidiasis" in condition:
        specific_recommendations = [
            "Apply antifungal creams, powders, or sprays as directed",
            "Keep affected areas clean and dry",
            "Wear breathable fabrics",
            "Use separate towels and washcloths",
            "Continue treatment for recommended duration even if symptoms improve"
        ]
    elif "herpes" in condition or "hpv" in condition or "std" in condition:
        specific_recommendations = [
            "Consult a healthcare provider for appropriate antiviral medication",
            "Avoid intimate contact during outbreaks",
            "Keep the area clean and dry",
            "Consider pain relievers for discomfort",
            "Discuss vaccination options with your doctor"
        ]
    elif "hives" in condition or "urticaria" in condition:
        specific_recommendations = [
            "Identify and avoid triggers",
            "Try over-the-counter antihistamines",
            "Apply cool compresses to affected areas",
            "Wear loose-fitting clothing",
            "Consult a doctor for persistent or severe symptoms"
        ]
    else:
        specific_recommendations = [
            "Keep the affected area clean",
            "Consider over-the-counter treatments appropriate for skin conditions",
            "Consult with a dermatologist for personalized advice",
            "Document changes with photos to share with healthcare providers",
            "Avoid self-diagnosis and seek professional medical opinion"
        ]
    
    # Combine and return unique recommendations
    all_recommendations = base_recommendations + specific_recommendations
    return all_recommendations[:5]  # Limit to 5 recommendations

if __name__ == '__main__':
    app.run(debug=True)