from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = None
try:
    if os.path.exists('energy_forecast_model.h5'):
        model = load_model('energy_forecast_model.h5')
        print("✅ Model loaded successfully.")
    else:
        print("⚠️ Warning: energy_forecast_model.h5 not found. Using mock predictions.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Using mock predictions.")

# Initialize scaler
features = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6',
            'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
            'Visibility', 'Tdewpoint', 'hour', 'dayofday', 'month', 'is_weekend']

scaler = MinMaxScaler()

# Fit scaler on sample data ranges
sample_data = np.array([
    [0, 15, 20, 15, 20, 15, 20, 15, 20, 15, 20, -10, 0, 15, 20, 15, 20, 15, 20, -10, 700, 0, 0, 0, -10, 0, 0, 1, 0],
    [50, 30, 60, 30, 60, 30, 60, 30, 60, 30, 80, 30, 100, 30, 60, 30, 60, 30, 60, 30, 800, 100, 20, 100, 20, 23, 6, 12, 1]
])
scaler.fit(sample_data)

def mock_predict(input_data):
    """Mock prediction when model is not available"""
    # Simple rule-based prediction based on lights usage and temperature
    lights = input_data[0]
    indoor_temp = input_data[1]
    outdoor_temp = input_data[19]
    hour = input_data[25]
    
    # Calculate energy consumption score
    base_score = lights / 50.0
    temp_diff = abs(indoor_temp - outdoor_temp)
    time_factor = 1.5 if 18 <= hour <= 22 else 1.0
    
    total_score = base_score + (temp_diff / 20.0) * time_factor
    
    if total_score < 1.5:
        probs = [0.8, 0.15, 0.05]
    elif total_score < 3.0:
        probs = [0.2, 0.6, 0.2]
    else:
        probs = [0.1, 0.2, 0.7]
    
    return np.array([probs])

def get_prediction_details(raw_pred, class_index, input_data):
    """Generate detailed prediction information for frontend"""
    class_index = int(class_index)  # Ensure Python int
    class_labels = ['Low (<100 Wh)', 'Medium (100-200 Wh)', 'High (>200 Wh)']
    colors = ['#10b981', '#f59e0b', 'ef4444']  # Green, Orange, Red
    
    prediction = class_labels[class_index]
    confidence = float(np.max(raw_pred))
    
    # Probability breakdown
    probabilities = {
        'Low': float(raw_pred[0]),
        'Medium': float(raw_pred[1]),
        'High': float(raw_pred[2])
    }
    
    # Chart data
    chart_data = {
        'labels': class_labels,
        'probabilities': [float(p) for p in raw_pred],
        'colors': colors,
        'predicted_class': class_index  # Now Python int
    }
    
    # Energy level info
    energy_level = {
        'label': prediction,
        'color': colors[class_index],
        'confidence': f"{confidence * 100:.1f}%",
        'confidence_value': confidence,
        'emoji': '🟢' if class_index == 0 else '🟡' if class_index == 1 else '🔴'
    }
    
    # Recommendations
    recommendations = {
        0: "✅ Excellent energy efficiency! Continue current practices.",
        1: "⚠️ Moderate usage detected. Consider optimizing lighting and HVAC.",
        2: "🔴 High consumption! Immediate optimization recommended for lights and temperature control."
    }
    
    # Key insights
    insights = {
        'lights_usage': float(input_data[0]),
        'indoor_temp': float(input_data[1]),  # T1
        'outdoor_temp': float(input_data[19]),  # T_out
        'humidity': float(input_data[2]),  # RH_1
        'hour_of_day': int(input_data[25])  # hour
    }
    
    return {
        'prediction': prediction,
        'class_index': class_index,
        'confidence': confidence,
        'confidence_display': f"{confidence * 100:.1f}%",
        'probabilities': probabilities,
        'chart_data': chart_data,
        'energy_level': energy_level,
        'recommendation': recommendations[class_index],
        'insights': insights,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'success',
        'model_used': 'real' if model else 'mock',
        'input_data': {features[i]: float(input_data[i]) for i in range(len(features))}
    }

@app.route('/')
def index():
    try:
        if os.path.exists('index.html'):
            return send_file('index.html')
        else:
            print("index.html not found, creating basic version...")
            with open('index.html', 'w') as f:
                f.write('''<!DOCTYPE html>
<html><head><title>Energy Forecast</title></head>
<body><h1>Energy Forecasting App Ready!</h1>
<p>Server running. Create your index.html for the frontend.</p></body></html>''')
            return send_file('index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return jsonify({'error': f'Failed to serve index.html: {str(e)}'}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received', 'status': 'error'}), 400
        
        print(f"📥 Received prediction request with {len(data)} features")
        
        # Validate all required features
        input_data = []
        missing_features = []
        for feature in features:
            if feature not in data:
                missing_features.append(feature)
            else:
                try:
                    input_data.append(float(data[feature]))
                except (ValueError, TypeError):
                    return jsonify({
                        'error': f'Invalid value for {feature}: {data[feature]}',
                        'status': 'error'
                    }), 400
        
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features[:5]}{"..." if len(missing_features) > 5 else ""}',
                'status': 'error'
            }), 400
        
        print(f"✅ Valid input data prepared: {len(input_data)} features")
        
        # Prepare input for model
        input_data = np.array([input_data])
        scaled_input = scaler.transform(input_data)
        
        # Reshape for LSTM model [samples, timesteps, features]
        seq_length = 6
        scaled_input = np.repeat(scaled_input[:, np.newaxis, :], seq_length, axis=1)
        
        # Get prediction
        if model:
            raw_pred = model.predict(scaled_input, verbose=0)
        else:
            raw_pred = mock_predict(input_data[0])
        
        class_index = np.argmax(raw_pred, axis=1)[0]
        
        # Generate detailed response
        prediction_details = get_prediction_details(raw_pred[0], class_index, input_data[0])
        
        # NO CONSOLE LOGGING OF RESULTS - ALL TO FRONTEND
        return jsonify(prediction_details)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(features),
        'features': features[:5] + ['...'] if len(features) > 5 else features,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()
        print("📧 Contact message received:", data)
        return jsonify({'success': 'Message received successfully! We will get back to you soon.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Energy Consumption Forecasting API")
    print(f"📊 Model status: {'✅ LOADED' if model else '⚠️ MOCK MODE'}")
    print("🌐 Server available at: http://localhost:5000")
    print("📁 Place index.html in the same directory as app.py")
    print("📁 Place energy_forecast_model.h5 in the same directory (optional)")
    print("🎨 Predictions displayed in frontend - no console output")
    print("-" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)