# VoxelInsight Professional Setup Guide

This guide will help you set up VoxelInsight with a professional landing page and Supabase authentication.

## 🚀 Quick Start

### 1. Environment Setup

Make sure you're in the correct Python environment:
```bash
conda activate voxelinsight
```

### 2. Supabase Setup

1. **Create a Supabase Project**:
   - Go to [supabase.com](https://supabase.com)
   - Create a new project
   - Note your project URL and anon key

2. **Configure Environment Variables**:
   Edit the `.env` file with your credentials:
   ```bash
   # Supabase Configuration
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your_anon_key_here
   
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Application Configuration
   SECRET_KEY=your_secret_key_here
   ENVIRONMENT=development
   ```

### 3. Update JavaScript Configuration

Edit `static/landing.js` and replace the placeholder values:
```javascript
const SUPABASE_URL = 'https://your-project-id.supabase.co';
const SUPABASE_ANON_KEY = 'your_anon_key_here';
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

**Option 1: Run with Flask (Recommended for production)**
```bash
python server.py
```
This will serve the landing page at `http://localhost:8000`

**Option 2: Run Chainlit directly**
```bash
chainlit run app.py
```
This will serve the chat interface directly at `http://localhost:8000`

## 🎨 Customization

### Branding
- **Logo**: Replace `static/voxelinsight_logo.svg` with your logo
- **Colors**: Update the gradient colors in `static/landing.css`
- **Content**: Edit `static/index.html` to customize the content

### Features
- **Pricing**: Modify the pricing section in `static/index.html`
- **Features**: Update the features list to match your capabilities
- **Contact**: Change the contact email in the JavaScript functions

## 🔐 Authentication Flow

1. **Landing Page**: Users visit the professional landing page
2. **Sign Up/Sign In**: Users authenticate via Supabase
3. **Chat Interface**: Authenticated users are redirected to the Chainlit chat
4. **Session Management**: User sessions are maintained via Flask sessions

## 📁 File Structure

```
VoxelInsight-3/
├── static/
│   ├── index.html          # Professional landing page
│   ├── landing.css         # Landing page styles
│   ├── landing.js          # Authentication & interactions
│   ├── voxelinsight_logo.svg # Logo
│   └── style.css           # Original Chainlit styles
├── app.py                  # Main Chainlit application
├── server.py              # Flask server for landing page
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── .chainlit/
    └── config.toml        # Chainlit configuration
```

## 🌐 Deployment

### Local Development
```bash
python server.py
```

### Production Deployment
1. Set up a production server (AWS, Google Cloud, etc.)
2. Configure environment variables
3. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 server:app
   ```

## 🔧 Troubleshooting

### Common Issues

1. **Supabase Connection Error**:
   - Verify your Supabase URL and anon key
   - Check that your Supabase project is active

2. **Authentication Not Working**:
   - Ensure Supabase Auth is enabled in your project
   - Check browser console for JavaScript errors

3. **Styling Issues**:
   - Clear browser cache
   - Verify CSS files are being served correctly

4. **Chainlit Not Starting**:
   - Check that all dependencies are installed
   - Verify OpenAI API key is set

### Debug Mode

Enable debug mode in Flask:
```python
app.run(debug=True, port=8000)
```

## 📞 Support

For issues or questions:
- Check the browser console for errors
- Review the Flask server logs
- Ensure all environment variables are set correctly

## 🎯 Next Steps

1. **Customize the Design**: Update colors, fonts, and layout
2. **Add Analytics**: Integrate Google Analytics or similar
3. **Email Integration**: Set up email notifications for signups
4. **Database**: Add user profiles and usage tracking
5. **Payment Integration**: Add Stripe or similar for paid plans

## 🔒 Security Notes

- Never commit `.env` files to version control
- Use strong secret keys in production
- Enable HTTPS in production
- Regularly update dependencies
- Monitor for security vulnerabilities 