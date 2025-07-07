from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import chainlit as cl
import asyncio
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = None

# Only initialize Supabase if credentials are properly configured
if supabase_url and supabase_key and supabase_url != "your_supabase_project_url":
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"⚠️  Supabase initialization failed: {e}")
        supabase = None
else:
    print("⚠️  Supabase credentials not configured. Authentication will be disabled.")
    supabase = None

@app.route('/')
def landing_page():
    """Serve the custom landing page"""
    return app.send_static_file('index.html')

@app.route('/chat')
def chat_page():
    """Serve the Chainlit chat interface"""
    # Check if user is authenticated
    if not session.get('user_id'):
        return redirect('/')
    
    # Redirect to Chainlit running on a different port
    # The user should start Chainlit separately with: chainlit run app.py --port 8001
    return redirect('http://localhost:8001')

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle user signup"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not supabase:
            # Demo mode - accept any signup
            session['user_id'] = f"demo_user_{email}"
            session['user_email'] = email
            return jsonify({'message': 'Demo account created successfully'}), 200
        
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user:
            session['user_id'] = response.user.id
            session['user_email'] = email
            return jsonify({'message': 'Account created successfully'}), 200
        else:
            return jsonify({'error': 'Failed to create account'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/signin', methods=['POST'])
def signin():
    """Handle user signin"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not supabase:
            # Demo mode - accept any signin
            session['user_id'] = f"demo_user_{email}"
            session['user_email'] = email
            return jsonify({'message': 'Demo signin successful'}), 200
        
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            session['user_id'] = response.user.id
            session['user_email'] = email
            return jsonify({'message': 'Signed in successfully'}), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/signout')
def signout():
    """Handle user signout"""
    session.clear()
    return jsonify({'message': 'Signed out successfully'}), 200

@app.route('/api/auth/user')
def get_user():
    """Get current user information"""
    user_id = session.get('user_id')
    if user_id:
        return jsonify({'user_id': user_id}), 200
    else:
        return jsonify({'error': 'Not authenticated'}), 401

if __name__ == '__main__':
    app.run(debug=True, port=8000) 