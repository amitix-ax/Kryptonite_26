import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the HTML dashboard

@app.route('/api/gemini_summary', methods=['POST'])
def gemini_summary():
    data = request.json
    
    # Initialize Gemini Client with env key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"summary": "GEMINI_API_KEY not set"}), 500
    client = genai.Client(api_key=api_key)
    
    # Extract telemetry for the prompt
    wind = data.get('env_wind_spd', 0)
    ice = data.get('env_ice_conc', 0) * 100
    u = data.get('u', 0)
    v = data.get('v', 0)
    r = data.get('r', 0)
    res = data.get('r_ice', 0)
    
    prompt = f"""
    You are the 'Kryptonite' AI Decision Support System onboard a Polar Class vessel.
    Analyze this current digital twin telemetry:
    - Wind Speed: {wind:.1f} m/s
    - Ice Concentration: {ice:.1f}%
    - Surge Velocity (u): {u:.2f} m/s
    - Sway Velocity (v): {v:.3f} m/s
    - Yaw Rate (r): {r:.3f} rad/s
    - Ice Resistance: {res:.0f} kN

    Write a highly concise, professional 3-4 sentence tactical summary of the current physics state. 
    State if conditions are nominal or if the vessel is experiencing severe leeway (sway) or heavy ice resistance. 
    Use a cold, analytical tone. Do not use markdown formatting.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return jsonify({"summary": response.text})
    except Exception as e:
        print("API Error:", e)
        return jsonify({"summary": f"API Error: {str(e)}\n\nPlease ensure your GEMINI_API_KEY environment variable is set correctly."}), 500

@app.route('/api/ai/analyze-route', methods=['POST'])
def analyze_route():
    data = request.json
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"advisory": "GEMINI_API_KEY not set"}), 500
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are the 'Kryptonite' AI Decision Support System onboard a Polar Class vessel.
    Analyze this proposed route:
    - Total Distance: {data.get('total_distance_km', 0):.1f} km
    - Open Water: {data.get('open_water_km', 0):.1f} km
    - Marginal Ice: {data.get('marginal_ice_km', 0):.1f} km
    - Pack Ice: {data.get('pack_ice_km', 0):.1f} km
    - Max Ice Concentration: {data.get('max_ice_conc', 0):.1f} %
    - Max Crosswind: {data.get('max_crosswind', 0):.1f} knots
    - Max Drift Velocity: {data.get('avg_drift', 0):.2f} m/s

    Provide a highly concise, professional tactical route assessment (4-5 bullet points). 
    Identify the most critical leg of the journey and any severe risks. 
    Use a cold, analytical tone. Format with markdown bullet points.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return jsonify({"analysis_markdown": response.text})
    except Exception as e:
        print("API Error (Route Analysis):", e)
        return jsonify({"analysis_markdown": f"API Error: {str(e)}"}), 500


if __name__ == '__main__':
    print("Starting Gemini API Bridge Server on http://127.0.0.1:5000")
    print("Ensure you have set: setx GEMINI_API_KEY 'your-key' (Windows)")
    app.run(port=5000)
