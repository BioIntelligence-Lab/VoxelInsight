# VoxelInsight Startup Guide

## Quick Start

1. **Activate the environment:**
   ```bash
   conda activate voxelinsight
   ```

2. **Start the Flask server (Landing page + Authentication):**
   ```bash
   python server.py
   ```
   - This runs on http://localhost:8000
   - Handles authentication and landing page

3. **Start Chainlit (Chat interface):**
   ```bash
   chainlit run app.py --port 8001
   ```
   - This runs on http://localhost:8001
   - Handles the AI chat interface

## Authentication Flow

1. **Sign Up/Sign In**: Users authenticate through the landing page
2. **Redirect**: After successful authentication, users are redirected to `/chat`
3. **Chat Interface**: The `/chat` route redirects to the Chainlit interface on port 8001

## Troubleshooting

- **Port conflicts**: Make sure ports 8000 and 8001 are available
- **Environment issues**: Always use `conda activate voxelinsight` before running commands
- **Authentication errors**: Check that your `.env` file has valid Supabase credentials

## Development

- Flask server: http://localhost:8000
- Chainlit interface: http://localhost:8001
- Both need to be running for full functionality 