from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pickle
import re
import numpy as np
from urllib.parse import urlparse
import uvicorn

app = FastAPI(
    title="Spam Text Detector API",
    description="API for detecting spam texts and analyzing URLs",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Load the model (lazily on first request)
model = None

def get_model():
    global model
    if model is None:
        try:
            with open('text_classification.pkl', 'rb') as file:
                model = pickle.load(file)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model

class TextInput(BaseModel):
    text: str

class URLAnalysisResult(BaseModel):
    domain: str
    url: str
    trust_score: float
    classification: str
    risk_factors: List[str]
    security_features: List[str]
    error: Optional[str] = None

class SpamAnalysisResult(BaseModel):
    prediction: str
    confidence: float
    spam_probability: float
    not_spam_probability: float
    urls: List[URLAnalysisResult] = []

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze", response_model=SpamAnalysisResult)
async def analyze_text(text_input: TextInput):
    text = text_input.text
    
    # Check if model is loaded
    model = get_model()
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please check the model file."}
        )
    
    # Predict spam
    prediction_result, confidence, spam_prob, not_spam_prob = predict_spam(text, model)
    
    # Extract and analyze URLs
    urls = extract_urls(text)
    url_results = []
    for url in urls:
        url_results.append(check_website_trust(url))
    
    # Prepare response
    response = {
        "prediction": prediction_result,
        "confidence": confidence,
        "spam_probability": spam_prob,
        "not_spam_probability": not_spam_prob,
        "urls": url_results
    }
    
    return response

def predict_spam(text, model):
    try:
        prediction = model.predict([text])
        try:
            prob = model.predict_proba([text])[0]
            spam_prob = float(prob[0] * 100)
            not_spam_prob = float(prob[1] * 100)
            confidence = float(max(prob) * 100)
        except:
            spam_prob = 85.0 if prediction[0] == 0 else 15.0
            not_spam_prob = 15.0 if prediction[0] == 0 else 85.0
            confidence = max(spam_prob, not_spam_prob)
        result = "Spam" if prediction[0] == 0 else "Not Spam"
        return result, confidence, spam_prob, not_spam_prob
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0, 0, 0

def extract_urls(text):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[\w\d._~:/?#[\]@!$&\'()*+,;=]+'
    urls = re.findall(url_pattern, text)
    return urls

def check_website_trust(url):
    try:
        domain = urlparse(url).netloc
        
        trust_score = 0
        risk_factors = []
        security_features = []
        
        established_domains = ["google.com", "microsoft.com", "amazon.com", "github.com", 
                              "apple.com", "netflix.com", "yahoo.com", "linkedin.com"]
        suspicious_keywords = ["free", "win", "prize", "crypto", "urgent", "lucky", "discount", "act-now"]
        security_keywords = ["secure", "official", "verified", "authentic"]
        
        if any(domain.endswith(ed) or ed in domain for ed in established_domains):
            trust_score = 90
            security_features.append("Established domain")
        elif len(domain) > 30:
            trust_score = 30
            risk_factors.append("Unusually long domain name")
        elif any(keyword in domain.lower() for keyword in suspicious_keywords):
            trust_score = 40
            risk_factors.append("Contains suspicious keywords")
            
            for keyword in suspicious_keywords:
                if keyword in domain.lower():
                    risk_factors.append(f"Contains '{keyword}'")
        else:
            trust_score = 65
        
        if domain.count('.') > 2:
            trust_score -= 15
            risk_factors.append("Multiple subdomains")
        
        if any(keyword in domain.lower() for keyword in security_keywords):
            if trust_score < 70:
                trust_score += 10
                security_features.append("Security signaling terms")
        
        trust_score = max(0, min(100, trust_score))
            
        return {
            "domain": domain,
            "url": url,
            "trust_score": trust_score,
            "classification": get_trust_classification(trust_score),
            "risk_factors": risk_factors,
            "security_features": security_features
        }
    except Exception as e:
        return {
            "domain": url,
            "url": url,
            "trust_score": 0,
            "classification": "Error analyzing URL",
            "risk_factors": ["Error in analysis"],
            "security_features": [],
            "error": str(e)
        }

def get_trust_classification(score):
    if score >= 80:
        return "High Trust"
    elif score >= 60:
        return "Moderate Trust"
    elif score >= 40:
        return "Low Trust"
    else:
        return "Suspicious"

@app.get("/api/demo-text")
async def get_demo_text():
    demo_text = """URGENT: Your account has been compromised! 
    Click here to verify: http://secur1ty-verify.prize-winner.com
    Also check our legitimate site at https://google.com for more information.
    You've won $1000 - claim at https://free-prizes-winner.net now!"""
    
    return {"text": demo_text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)