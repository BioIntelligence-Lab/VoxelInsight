from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import os
import chainlit as cl
import asyncio
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

@app.route('/')
def landing_page():
    """Serve the custom landing page"""
    return app.send_static_file('index.html')

@app.route('/chat')
def chat_page():
    """Serve the Chainlit chat interface"""
    # Redirect to Chainlit running on a different port
    # The user should start Chainlit separately with: chainlit run app.py --port 8001
    return redirect('http://localhost:8001')

if __name__ == '__main__':
    app.run(debug=True, port=8000) 