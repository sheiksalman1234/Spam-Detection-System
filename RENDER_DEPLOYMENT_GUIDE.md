# Deploy to Render - Step by Step Guide

## Prerequisites
- GitHub account with your repository pushed
- Render account (free tier available)
- Gemini API key (get from https://aistudio.google.com/app/apikeys)

## Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

## Step 2: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub
3. Authorize Render to access your repositories

## Step 3: Create New Web Service
1. Click "New +" → "Web Service"
2. Select your SpamCallDetection repository
3. Click "Connect"

## Step 4: Configure Service
- **Name**: `spamshield` (or your preferred name)
- **Environment**: Docker
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Build Command**: (leave empty - uses Dockerfile)
- **Start Command**: (leave empty - uses Dockerfile CMD)

## Step 5: Add Environment Variables
1. Scroll to "Environment" section
2. Click "Add Environment Variable"
3. Add:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `<your-gemini-api-key>`

## Step 6: Select Plan
- **Free Tier**: 750 hours/month (sufficient for testing)
- **Paid Tier**: Unlimited hours (recommended for production)

## Step 7: Deploy
1. Click "Create Web Service"
2. Render will start building (takes 5-10 minutes)
3. Wait for "Live" status

## Step 8: Verify Deployment
Once deployed, test these endpoints:

### Health Check
```bash
curl https://your-service-name.onrender.com/health
```

### Test Translation
```bash
curl -X POST https://your-service-name.onrender.com/test-translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Your bank account has been blocked"}'
```

### Test Spam Detection
```bash
curl -X POST https://your-service-name.onrender.com/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your bank account has been blocked. Share your OTP immediately."}'
```

## Important Notes

### First Request Delay
- **First request will take 2-3 minutes** as models download
- This is normal - models are cached after first download
- Subsequent requests will be fast (< 5 seconds)

### Model Download
The following models will be downloaded on first request:
- DistilBERT spam classifier (~300 MB)
- Whisper tiny speech-to-text (~140 MB)
- HuggingFace cache (~100 MB)
- **Total**: ~540 MB

### Render Limitations
- Free tier: 750 hours/month
- Paid tier: Unlimited
- Inactivity: Service spins down after 15 minutes of no requests
- Startup time: 30-60 seconds after spin-up

### Monitoring
1. Go to Render dashboard
2. Select your service
3. View logs in real-time
4. Check metrics (CPU, memory, requests)

## Troubleshooting

### Build Fails
- Check Dockerfile is in root directory
- Verify all dependencies in requirements.txt
- Check Render logs for specific errors

### Service Won't Start
- Check GEMINI_API_KEY is set correctly
- Verify port 8080 is exposed in Dockerfile
- Check logs for Python errors

### Slow Performance
- First request: Normal (model download)
- Subsequent requests: Should be < 5 seconds
- If consistently slow: Upgrade to paid tier

### Translation Not Working
- Verify GEMINI_API_KEY is set
- Check internet connectivity (deep-translator needs it)
- Fallback to Gemini API if deep-translator fails

## Custom Domain (Optional)
1. In Render dashboard, go to "Settings"
2. Add custom domain
3. Update DNS records as instructed
4. SSL certificate auto-generated

## Auto-Deploy (Optional)
1. In Render dashboard, go to "Settings"
2. Enable "Auto-Deploy"
3. Service will redeploy on every GitHub push

---

**Your Service URL**: `https://your-service-name.onrender.com`
**Status**: Ready to deploy ✅
