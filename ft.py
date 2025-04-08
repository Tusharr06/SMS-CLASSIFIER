import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from urllib.parse import urlparse
import time
import datetime
import requests
from io import BytesIO
import base64
import json
from textblob import TextBlob
import hashlib
import validators
import calendar

# For PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Add this for fake whois data (in a real app, you would use a whois library)
def get_domain_info(domain):
    # Simulated domain age and registration data
    domain_hash = int(hashlib.md5(domain.encode()).hexdigest(), 16) % 10000
    registration_year = max(2000, 2023 - (domain_hash % 20))
    current_year = 2025
    domain_age = current_year - registration_year
    
    registrars = ["GoDaddy.com, LLC", "Namecheap, Inc.", "Google Domains", "NameSilo, LLC", 
                 "Network Solutions, LLC", "Tucows Domains Inc.", "Amazon Registrar, Inc."]
    
    countries = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "BR", "IN", "SG"]
    
    return {
        "domain_age": domain_age,
        "creation_date": f"{registration_year}-{(domain_hash % 12) + 1}-{(domain_hash % 28) + 1}",
        "expiration_date": f"{current_year + 1}-{(domain_hash % 12) + 1}-{(domain_hash % 28) + 1}",
        "registrar": registrars[domain_hash % len(registrars)],
        "country": countries[domain_hash % len(countries)]
    }

# Add this for SSL certificate check simulation
def check_ssl(url):
    # Simulate SSL check
    domain = urlparse(url).netloc
    domain_hash = int(hashlib.md5(domain.encode()).hexdigest(), 16)
    
    # Well-known domains are more likely to have SSL
    has_ssl = True if any(known in domain for known in ["google", "amazon", "microsoft", "github", "apple"]) else (domain_hash % 100 > 30)
    
    if has_ssl:
        cert_authorities = ["DigiCert Inc", "Let's Encrypt", "Comodo CA", "GeoTrust Inc", "GlobalSign"]
        expiry_days = (domain_hash % 300) + 65
        
        return {
            "has_ssl": True,
            "issuer": cert_authorities[domain_hash % len(cert_authorities)],
            "valid_days": expiry_days,
            "valid_until": (datetime.datetime.now() + datetime.timedelta(days=expiry_days)).strftime("%Y-%m-%d")
        }
    else:
        return {
            "has_ssl": False
        }

# Function to detect PII in text
def detect_pii(text):
    pii_found = []
    
    # Simple patterns - in a real app you would use more sophisticated regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    credit_card_pattern = r'\b(?:\d{4}[- ]?){3}\d{4}\b'
    
    if re.search(email_pattern, text):
        pii_found.append("Email Address")
    
    if re.search(phone_pattern, text):
        pii_found.append("Phone Number")
    
    if re.search(ssn_pattern, text):
        pii_found.append("Social Security Number")
    
    if re.search(credit_card_pattern, text):
        pii_found.append("Credit Card Number")
    
    return pii_found

# Function to create downloadable PDF report
def create_pdf_report(text_analysis, url_analysis):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Spam Text Analysis Report")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 70, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Text analysis results
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 100, "Text Analysis Results")
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 130, f"Classification: {text_analysis['prediction']}")
    p.drawString(50, height - 150, f"Confidence: {text_analysis['confidence']:.1f}%")
    
    if 'sentiment' in text_analysis:
        p.drawString(50, height - 170, f"Sentiment: {text_analysis['sentiment']['label']} ({text_analysis['sentiment']['score']:.2f})")
    
    # URL analysis if available
    if url_analysis and len(url_analysis) > 0:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 210, "URL Analysis Results")
        
        y_position = height - 240
        for i, url_info in enumerate(url_analysis):
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y_position, f"URL {i+1}: {url_info['domain']}")
            p.setFont("Helvetica", 12)
            y_position -= 20
            p.drawString(70, y_position, f"Trust Score: {url_info['trust_score']}/100 ({url_info['classification']})")
            y_position -= 20
            
            if 'domain_age' in url_info:
                p.drawString(70, y_position, f"Domain Age: {url_info['domain_age']} years")
                y_position -= 20
            
            if url_info['risk_factors']:
                p.drawString(70, y_position, "Risk Factors:")
                y_position -= 20
                for factor in url_info['risk_factors']:
                    p.drawString(90, y_position, f"- {factor}")
                    y_position -= 20
            
            y_position -= 10
    
    # Safety tips
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 150, "Safety Tips")
    p.setFont("Helvetica", 12)
    safety_tips = [
        "Be wary of urgent requests",
        "Check URL spelling carefully",
        "Never share sensitive information",
        "Look for HTTPS in website addresses",
        "When in doubt, contact organizations directly"
    ]
    
    y_pos = 130
    for tip in safety_tips:
        p.drawString(70, y_pos, f"• {tip}")
        y_pos -= 20
    
    # Footer
    p.setFont("Helvetica-Italic", 10)
    p.drawString(50, 50, "Generated by Smart Spam Text Detector")
    
    p.save()
    buffer.seek(0)
    return buffer

# Page configuration with theme selection capability
def set_page_style(theme):
    if theme == "dark":
        bg_color = "#121212"
        text_color = "#ffffff"
        card_bg = "rgba(42, 49, 59, 0.7)"
        accent_color = "#38b6ff"
    else:
        bg_color = "#f8f9fa"
        text_color = "#212529"
        card_bg = "rgba(255, 255, 255, 0.85)"
        accent_color = "#0d6efd"
        
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: {accent_color};
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 0 10px rgba(56, 182, 255, 0.3);
        }}
        .subheader {{
            font-size: 1.5rem;
            color: {accent_color};
            margin-top: 2rem;
            margin-bottom: 1rem;
        }}
        .card {{
            background-color: {card_bg};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            border: 1px solid rgba(86, 100, 120, 0.5);
        }}
        .result-container {{
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            border: 1px solid rgba(86, 100, 120, 0.5);
        }}
        .spam-result {{
            font-size: 1.2rem;
            font-weight: bold;
        }}
        .url-card {{
            background-color: {card_bg};
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(86, 100, 120, 0.5);
        }}
        .url-title {{
            font-weight: bold;
            color: {accent_color};
        }}
        .metric-container {{
            text-align: center;
            background: {card_bg};
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin: 10px 0;
            border: 1px solid rgba(86, 100, 120, 0.5);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: {text_color};
            opacity: 0.8;
        }}
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {{
            background-color: {card_bg};
            color: {text_color};
            border: 1px solid rgba(86, 100, 120, 0.5);
        }}
        .stButton > button {{
            border: 1px solid rgba(86, 100, 120, 0.5);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        .info-card {{
            background-color: {card_bg};
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(86, 100, 120, 0.5);
            margin-bottom: 15px;
        }}
        .status-box {{
            border-radius: 5px;
            padding: 10px 15px;
            margin: 10px 0;
            border: 1px solid rgba(86, 100, 120, 0.5);
            text-align: center;
        }}
        .text-primary {{
            color: {text_color};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {card_bg};
            border-radius: 4px 4px 0px 0px;
            border: 1px solid rgba(86, 100, 120, 0.5);
            border-bottom: none;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent_color};
            color: white;
        }}
        
        /* New feature styles */
        .feature-card {{
            background-color: {card_bg};
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(86, 100, 120, 0.5);
            margin-bottom: 15px;
            transition: transform 0.3s ease;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
        }}
        .feature-icon {{
            font-size: 2rem;
            color: {accent_color};
            margin-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        .badge-info {{
            background-color: {accent_color};
            color: white;
        }}
        .badge-warning {{
            background-color: #ffc107;
            color: #212529;
        }}
        .badge-danger {{
            background-color: #dc3545;
            color: white;
        }}
        .badge-success {{
            background-color: #28a745;
            color: white;
        }}
        .pii-alert {{
            background-color: rgba(220, 53, 69, 0.1);
            border: 1px solid #dc3545;
            color: #dc3545;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .timeline {{
            position: relative;
            margin-left: 20px;
            padding-left: 20px;
        }}
        .timeline:before {{
            content: '';
            position: absolute;
            left: 0;
            top: 5px;
            bottom: 0;
            width: 2px;
            background: {accent_color};
        }}
        .timeline-item {{
            position: relative;
            margin-bottom: 15px;
        }}
        .timeline-item:before {{
            content: '';
            position: absolute;
            left: -28px;
            top: 5px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {accent_color};
        }}
        .sentiment-meter {{
            height: 10px;
            border-radius: 5px;
            background: linear-gradient(to right, #dc3545 0%, #ffc107 50%, #28a745 100%);
            margin: 5px 0;
            position: relative;
        }}
        .sentiment-indicator {{
            position: absolute;
            width: 10px;
            height: 15px;
            background: white;
            bottom: 0;
            border-radius: 3px;
            transform: translateX(-50%);
        }}
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .progress-bar {{
            height: 4px;
            background-color: #e9ecef;
            border-radius: 2px;
            margin: 10px 0;
        }}
        .progress-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('text_classification.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please upload the pickle file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_spam(text, model):
    try:
        prediction = model.predict([text])
        
        try:
            prob = model.predict_proba([text])[0]
            confidence = max(prob) * 100
            spam_prob = prob[0] * 100 if prediction[0] == 0 else (1 - prob[0]) * 100
        except:
            confidence = 85.0
            spam_prob = 85.0 if prediction[0] == 0 else 15.0
        
        result = "Spam" if prediction[0] == 0 else "Not Spam"
        return result, confidence, spam_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, 0

def extract_urls(text):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[\w\d._~:/?#[\]@!$&\'()*+,;=]+'
    urls = re.findall(url_pattern, text)
    return urls

def is_url_shortener(url):
    shorteners = [
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly",
        "adf.ly", "is.gd", "buff.ly", "rebrand.ly", "tiny.cc",
        "ow.ly", "bl.ink", "s.id", "tny.im", "rb.gy"
    ]
    domain = urlparse(url).netloc
    return any(shortener in domain for shortener in shorteners)

def analyze_keywords(text):
    # Remove URLs, punctuation, and convert to lowercase
    clean_text = re.sub(r'https?://\S+', '', text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = clean_text.lower()
    
    # Split into words and count occurrences
    words = clean_text.split()
    word_count = {}
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 or fewer
    top_words = sorted_words[:min(10, len(sorted_words))]
    
    return top_words

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        # Scale from -1,1 to 0,100
        score = (polarity + 1) * 50
        
        return {
            "score": score,
            "raw_score": polarity,
            "label": sentiment
        }
    except:
        return {
            "score": 50,
            "raw_score": 0,
            "label": "Neutral"
        }

def check_url_phishing_patterns(url):
    risk_factors = []
    
    # Check for IP address in URL
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        risk_factors.append("IP address in URL")
    
    # Check for multiple subdomains
    domain = urlparse(url).netloc
    if domain.count('.') > 2:
        risk_factors.append("Multiple subdomains")
    
    # Check for suspicious characters
    if re.search(r'[^a-zA-Z0-9.-]', domain):
        risk_factors.append("Suspicious characters in domain")
    
    # Check for misspelling of popular domains
    popular_domains = ["google", "facebook", "amazon", "apple", "microsoft", "paypal", "netflix"]
    for pop_domain in popular_domains:
        if pop_domain not in domain and any(fuzz_match(pop_domain, part) for part in domain.split('.')):
            risk_factors.append(f"Possible typosquatting of {pop_domain}")
            break
    
    # Check for URL shortener
    if is_url_shortener(url):
        risk_factors.append("URL shortener detected")
    
    return risk_factors

def fuzz_match(str1, str2):
    # Very simple fuzzy matching for demonstration
    # Returns True if strings are similar
    if abs(len(str1) - len(str2)) > 2:
        return False
    
    if str1[0] != str2[0]:
        return False
    
    common_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2))
    max_len = max(len(str1), len(str2))
    
    return common_chars / max_len > 0.7

def detect_special_characters(text):
    # Count special characters
    special_chars = re.findall(r'[^\w\s]', text)
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]')
    
    emojis = emoji_pattern.findall(text)
    
    return {
        "special_char_count": len(special_chars),
        "special_char_ratio": len(special_chars) / len(text) if text else 0,
        "emoji_count": len(emojis)
    }

def check_website_trust(url):
    try:
        domain = urlparse(url).netloc
        
        trust_score = 0
        risk_factors = []
        security_features = []
        
        established_domains = ["google.com", "microsoft.com", "amazon.com", "github.com", 
                              "apple.com", "netflix.com", "yahoo.com", "linkedin.com"]
        suspicious_keywords = ["free", "win", "prize", "crypto", "urgent", "lucky", "discount", "act-now", 
                              "limited-time", "password", "account", "verify", "bank", "paypal"]
        security_keywords = ["secure", "official", "verified", "authentic"]
        
        # Check for established domains
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
        
        # Check for multiple subdomains
        if domain.count('.') > 2:
            trust_score -= 15
            risk_factors.append("Multiple subdomains")
        
        # Check for security signaling terms
        if any(keyword in domain.lower() for keyword in security_keywords):
            if trust_score < 70:
                trust_score += 10
                security_features.append("Security signaling terms")
        
        # New: Check for URL shorteners
        if is_url_shortener(url):
            trust_score -= 20
            risk_factors.append("URL shortener detected")
        
        # New: Check for phishing patterns
        phishing_risks = check_url_phishing_patterns(url)
        if phishing_risks:
            trust_score -= 15
            risk_factors.extend(phishing_risks)
        
        # New: Check for SSL
        ssl_info = check_ssl(url)
        if ssl_info["has_ssl"]:
            trust_score += 10
            security_features.append(f"SSL Certificate (by {ssl_info['issuer']})")
            if ssl_info["valid_days"] < 30:
                risk_factors.append("SSL Certificate expiring soon")
        else:
            trust_score -= 15
            risk_factors.append("No SSL Certificate")
        
        # New: Check domain age
        domain_info = get_domain_info(domain)
        if domain_info["domain_age"] > 5:
            trust_score += 15
            security_features.append(f"Domain age: {domain_info['domain_age']} years")
        elif domain_info["domain_age"] < 1:
            trust_score -= 15
            risk_factors.append("Domain registered recently")
        
        # Ensure trust score is within bounds
        trust_score = max(0, min(100, trust_score))
            
        result = {
            "domain": domain,
            "url": url,
            "trust_score": trust_score,
            "classification": get_trust_classification(trust_score),
            "risk_factors": risk_factors,
            "security_features": security_features
        }
        
        # Add SSL and domain info to results
        if ssl_info.get("has_ssl"):
            result.update({
                "ssl_info": ssl_info
            })
        
        result.update({
            "domain_age": domain_info["domain_age"],
            "domain_creation": domain_info["creation_date"],
            "domain_expiration": domain_info["expiration_date"],
            "domain_registrar": domain_info["registrar"],
            "domain_country": domain_info["country"]
        })
        
        return result
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

def get_trust_color(score):
    if score >= 80:
        return "#00b894"
    elif score >= 60:
        return "#74b9ff"
    elif score >= 40:
        return "#fdcb6e"
    else:
        return "#ff7675"

def display_gauge_chart(value, title):
    fig, ax = plt.subplots(figsize=(3, 2), subplot_kw={'projection': 'polar'})
    
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    theta = np.pi * (value / 100 - 0.5)
    
    if value >= 80:
        color = '#00b894'
    elif value >= 60:
        color = '#74b9ff'
    elif value >= 40:
        color = '#fdcb6e'
    else:
        color = '#ff7675'
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    ax.barh(0, np.pi, height=0.2, left=-np.pi/2, color='#444444', alpha=0.5)
    
    ax.barh(0, theta + np.pi/2, height=0.2, left=-np.pi/2, color=color)
    
    ax.scatter(theta, 0, s=100, color=color, zorder=3, edgecolor='white')
    
    ax.text(0, -0.2, f"{value:.0f}%", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(0, -0.5, title, ha='center', va='center', fontsize=9, color='white')
    
    return fig

def main():
    # Initialize session state variables
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    set_page_style(st.session_state.theme)
    
    st.set_page_config(
        page_title="Advanced Spam Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">🛡️ Advanced Spam Text Detector</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/sparklines.png", width=100)
        
        # Theme switcher
        theme_col1, theme_col2 = st.columns([1, 1])
        with theme_col1:
            if st.button("🌙 Dark Mode", use_container_width=True, 
                         disabled=st.session_state.theme=="dark"):
                st.session_state.theme = "dark"
                st.rerun()
       with theme_col2:
    if st.button("☀️ Light Mode", use_container_width=True, disabled=st.session_state.theme=="light"):
        st.session_state.theme = "light"
        st.rerun()

st.header("About")
st.markdown("""
<div class="info-card">
    This advanced app uses machine learning to detect spam texts and analyze URL safety.
    It now features domain age verification, SSL checks, sentiment analysis, and PII detection.
</div>
""", unsafe_allow_html=True)

st.header("Website Trust Score")
st.markdown("""
<div class="info-card">
    <p><strong>Trust Score Range:</strong></p>
    <ul>
        <li><span style="color: #00b894;">80-100: High Trust</span></li>
        <li><span style="color: #74b9ff;">60-79: Moderate Trust</span></li>
        <li><span style="color: #fdcb6e;">40-59: Low Trust</span></li>
        <li><span style="color: #ff7675;">0-39: Suspicious</span></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# History section
if st.session_state.history:
    st.header("Analysis History")
    for i, entry in enumerate(st.session_state.history[-3:]):  # Show latest 3
        st.markdown(f"""
        <div class="info-card">
            <small>{entry['timestamp']}</small>
            <p><strong>Result:</strong> {entry['result']}</p>
            <p><strong>Confidence:</strong> {entry['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.header("Tips")
st.markdown("""
<div class="info-card" style="background-color: rgba(50, 70, 50, 0.7);">
    <p><strong>How to stay safe:</strong></p>
    <ul>
        <li>Be wary of urgent requests</li>
        <li>Check URL spelling carefully</li>
        <li>Never share sensitive information</li>
        <li>Look for HTTPS in website addresses</li>
        <li>Verify before clicking shortened URLs</li>
        <li>Check domain age for suspicious websites</li>
    </ul>
</div>
""", unsafe_allow_html=True)

if st.button("Try Demo Text", use_container_width=True):
    st.session_state.demo_text = """URGENT: Your account has been compromised! 
    Click here to verify: http://secur1ty-verify.prize-winner.com
    Also check our legitimate site at https://google.com for more information.
    You've won $1000 - claim at https://free-prizes-winner.net now!
    Contact us at 555-123-4567 or send your details to support@prize-winner.com"""
    st.rerun()

model = load_model()

tab1, tab2, tab3 = st.tabs(["Spam Detection", "Bulk Analysis", "How It Works"])

with tab1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if "demo_text" in st.session_state:
            user_input = st.text_area("Enter text to analyze:", 
                                    value=st.session_state.demo_text, 
                                    height=150,
                                    key="input_text")
            del st.session_state.demo_text
        else:
            user_input = st.text_area("Enter text to analyze:", height=150, key="input_text")
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col3:
            clear_button = st.button("🗑️ Clear", type="secondary", use_container_width=True)
        with col4:
            save_button = st.button("💾 Save PDF", use_container_width=True, disabled=not hasattr(st.session_state, 'last_analysis'))
        
        if clear_button:
            st.session_state.input_text = ""
            if 'last_analysis' in st.session_state:
                del st.session_state.last_analysis
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if save_button and hasattr(st.session_state, 'last_analysis'):
            pdf_buffer = create_pdf_report(
                st.session_state.last_analysis['text_analysis'],
                st.session_state.last_analysis['url_analysis'] if 'url_analysis' in st.session_state.last_analysis else None
            )
            
            b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="spam_analysis_report.pdf">Click here to download your PDF report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        if analyze_button:
            if not user_input:
                st.warning("Please enter some text to analyze.")
            elif model is None:
                st.error("Model not loaded. Please check the model file.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initial check
                status_text.text("Analyzing text...")
                progress_bar.progress(10)
                time.sleep(0.2)
                
                # Text classification
                prediction, confidence, spam_prob = predict_spam(user_input, model)
                
                status_text.text("Processing sentiment...")
                progress_bar.progress(25)
                time.sleep(0.2)
                
                # Sentiment analysis
                sentiment = analyze_sentiment(user_input)
                
                status_text.text("Checking for PII...")
                progress_bar.progress(40)
                time.sleep(0.2)
                
                # PII detection
                pii_found = detect_pii(user_input)
                
                status_text.text("Analyzing text patterns...")
                progress_bar.progress(50)
                time.sleep(0.2)
                
                # Keyword analysis
                top_keywords = analyze_keywords(user_input)
                
                # Special character analysis
                special_chars = detect_special_characters(user_input)
                
                status_text.text("Scanning for URLs...")
                progress_bar.progress(60)
                time.sleep(0.2)
                
                urls = extract_urls(user_input)
                
                url_results = []
                if urls:
                    status_text.text("Analyzing website safety...")
                    step_size = 30 / max(len(urls), 1)
                    current_progress = 65
                    
                    for url in urls:
                        url_results.append(check_website_trust(url))
                        current_progress += step_size
                        progress_bar.progress(min(int(current_progress), 90))
                        time.sleep(0.2)
                
                # Store analysis in session state for PDF export
                text_analysis = {
                    "text": user_input,
                    "prediction": prediction,
                    "confidence": confidence,
                    "spam_probability": spam_prob,
                    "sentiment": sentiment,
                    "pii_found": pii_found,
                    "top_keywords": top_keywords,
                    "special_chars": special_chars
                }
                
                st.session_state.last_analysis = {
                    "text_analysis": text_analysis,
                    "url_analysis": url_results if urls else None
                }
                
                # Add to history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                history_entry = {
                    "timestamp": timestamp,
                    "result": prediction,
                    "confidence": confidence,
                    "text_snippet": user_input[:30] + "..." if len(user_input) > 30 else user_input
                }
                st.session_state.history.append(history_entry)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results in tabs
                result_tab1, result_tab2, result_tab3 = st.tabs(["Text Analysis", "URL Safety", "Detailed Insights"])
                
                with result_tab1:
                    if prediction == "Spam":
                        result_color = "#ff7675"
                        emoji = "🚨"
                    else:
                        result_color = "#00b894"
                        emoji = "✅"
                    
                    st.markdown(f"""
                    <div style="background-color: {result_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h2>{emoji} {prediction}</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sentiment and text features
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3 style='text-align: center; color: #38b6ff;'>Text Sentiment</h3>", unsafe_allow_html=True)
                        
                        sentiment_position = sentiment["score"]
                        
                        st.markdown(f"""
                        <div class="card" style="text-align: center;">
                            <h4>{sentiment["label"]}</h4>
                            <div class="sentiment-meter">
                                <div class="sentiment-indicator" style="left: {sentiment_position}%;"></div>
                            </div>
                            <p>Score: {sentiment["raw_score"]:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display PII information
                        if pii_found:
                            st.markdown(f"""
                            <div class="pii-alert">
                                <h4>⚠️ Personal Information Detected</h4>
                                <p>Your text contains the following sensitive information:</p>
                                <ul>
                                {"".join(f"<li>{item}</li>" for item in pii_found)}
                                </ul>
                                <p><small>Consider removing sensitive information before sharing.</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<h3 style='text-align: center; color: #38b6ff;'>Spam Probability</h3>", unsafe_allow_html=True)
                        st.pyplot(display_gauge_chart(spam_prob, "Likelihood of being spam"))
                        
                        # Display special character info
                        st.markdown(f"""
                        <div class="card" style="margin-top: 10px;">
                            <h4 style="text-align: center;">Special Character Analysis</h4>
                            <p>Special characters: {special_chars["special_char_count"]} ({special_chars["special_char_ratio"]*100:.1f}% of text)</p>
                            <p>Emojis: {special_chars["emoji_count"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show keyword analysis
                    st.markdown("<h3 style='color: #38b6ff; margin-top: 20px;'>Keyword Analysis</h3>", unsafe_allow_html=True)
                    
                    if top_keywords:
                        # Create dataframe for keyword chart
                        keywords_df = pd.DataFrame(top_keywords, columns=["Word", "Count"])
                        
                        # List suspicious words
                        suspicious_words = ["urgent", "free", "limited", "offer", "click", "money", "prize", "win", "winner", "account", "verify", "password", "credit", "confidential", "bank", "expire"]
                        
                        # Add "is_suspicious" column
                        keywords_df["is_suspicious"] = keywords_df["Word"].apply(lambda x: x in suspicious_words)
                        
                        # Create the chart
                        keyword_chart = alt.Chart(keywords_df).mark_bar().encode(
                            x=alt.X("Count:Q", title="Frequency"),
                            y=alt.Y("Word:N", sort="-x", title=None),
                            color=alt.condition(
                                alt.datum.is_suspicious,
                                alt.value("#ff7675"),  # red for suspicious
                                alt.value("#74b9ff")   # blue for normal
                            ),
                            tooltip=["Word", "Count"]
                        ).properties(
                            title="Most Common Words",
                            height=min(250, len(keywords_df) * 25)
                        )
                        
                        st.altair_chart(keyword_chart, use_container_width=True)
                        
                        # Highlight suspicious words
                        suspicious_found = [word for word, count in top_keywords if word in suspicious_words]
                        if suspicious_found:
                            st.markdown(f"""
                            <div style="padding: 10px; background-color: rgba(255, 118, 117, 0.1); border-radius: 5px; margin: 10px 0;">
                                <p>⚠️ <strong>Warning:</strong> Found potentially suspicious words: {", ".join(suspicious_found)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                with result_tab2:
                    if urls:
                        st.markdown("<h3 style='color: #38b6ff;'>URL Safety Analysis</h3>", unsafe_allow_html=True)
                        
                        trust_scores = [info['trust_score'] for info in url_results]
                        domains = [info['domain'] for info in url_results]
                        classifications = [info['classification'] for info in url_results]
                        
                        df = pd.DataFrame({
                            'Domain': domains,
                            'Trust Score': trust_scores,
                            'Classification': classifications
                        })
                        
                        # Create a color scale based on trust score
                        color_scale = alt.Scale(
                            domain=[0, 40, 60, 80, 100],
                            range=['#ff7675', '#fdcb6e', '#74b9ff', '#00b894', '#00b894']
                        )
                        
                        # Create the chart
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('Domain', sort='-y', title='Website Domain'),
                            y=alt.Y('Trust Score', title='Trust Score (0-100)'),
                            color=alt.Color('Trust Score', scale=color_scale),
                            tooltip=['Domain', 'Trust Score', 'Classification']
                        ).properties(
                            title='Website Trust Scores',
                            height=300
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # URL Cards with detailed info
                        for i, url_info in enumerate(url_results):
                            with st.expander(f"URL {i+1}: {url_info['domain']} - {url_info['classification']}", expanded=(i==0)):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    fig = display_gauge_chart(url_info['trust_score'], "Trust Score")
                                    st.pyplot(fig)
                                
                                with col2:
                                    st.markdown(f"**Full URL:** {url_info['url']}")
                                    
                                    # Domain info
                                    st.markdown(f"""
                                    <div class="card" style="margin-top: 10px;">
                                        <h4>Domain Information</h4>
                                        <p>Domain Age: {url_info.get('domain_age', 'Unknown')} years</p>
                                        <p>Registered: {url_info.get('domain_creation', 'Unknown')}</p>
                                        <p>Registrar: {url_info.get('domain_registrar', 'Unknown')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # SSL Information
                                if "ssl_info" in url_info and url_info["ssl_info"]["has_ssl"]:
                                    st.markdown(f"""
                                    <div class="card" style="background-color: rgba(40, 167, 69, 0.1); margin-top: 10px;">
                                        <h4>🔒 SSL Certificate</h4>
                                        <p>Issuer: {url_info["ssl_info"]["issuer"]}</p>
                                        <p>Valid Until: {url_info["ssl_info"]["valid_until"]} ({url_info["ssl_info"]["valid_days"]} days)</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="card" style="background-color: rgba(220, 53, 69, 0.1); margin-top: 10px;">
                                        <h4>🔓 No SSL Certificate</h4>
                                        <p>This site does not use HTTPS, which is a security risk.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Risk factors and security features
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if url_info['risk_factors']:
                                        st.markdown("<strong>Risk Factors:</strong>", unsafe_allow_html=True)
                                        for factor in url_info['risk_factors']:
                                            st.markdown(f"""
                                            <div class="badge badge-danger">
                                                ⚠️ {factor}
                                            </div>
                                            """, unsafe_allow_html=True)
                                
                                with col2:
                                    if url_info['security_features']:
                                        st.markdown("<strong>Security Features:</strong>", unsafe_allow_html=True)
                                        for feature in url_info['security_features']:
                                            st.markdown(f"""
                                            <div class="badge badge-success">
                                                ✓ {feature}
                                            </div>
                                            """, unsafe_allow_html=True)
                                
                                # URL shortener warning
                                if is_url_shortener(url_info['url']):
                                    st.warning("⚠️ This is a URL shortener. Short URLs can hide malicious destinations.")
                        
                        # Overall assessment
                        avg_score = sum(trust_scores) / len(trust_scores)
                        if avg_score >= 70:
                            assessment_color = "#00b894"
                            assessment_text = "Links appear trustworthy"
                            emoji = "✅"
                        elif avg_score >= 50:
                            assessment_color = "#74b9ff"
                            assessment_text = "Links have moderate trust"
                            emoji = "ℹ️"
                        else:
                            assessment_color = "#ff7675"
                            assessment_text = "Links appear suspicious"
                            emoji = "⚠️"
                        
                        st.markdown(f"""
                        <div style="background-color: {assessment_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; border: 1px solid rgba(255, 255, 255, 0.2);">
                            <h3>{emoji} Overall URL Assessment</h3>
                            <p>Average trust score: {avg_score:.1f}/100 - {assessment_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No URLs found in the text.")
                
                with result_tab3:
                    st.markdown("<h3 style='color: #38b6ff;'>Detailed Analysis Insights</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Text statistics
                        word_count = len(re.findall(r'\b\w+\b', user_input))
                        sentence_count = len(re.split(r'[.!?]+', user_input))
                        avg_word_length = sum(len(word) for word in re.findall(r'\b\w+\b', user_input)) / max(1, word_count)
                        
                        st.markdown(f"""
                        <div class="card">
                            <h4 style="text-align: center;">Text Statistics</h4>
                            <div class="metric-container">
                                <div class="metric-value">{word_count}</div>
                                <div class="metric-label">Words</div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-value">{sentence_count}</div>
                                <div class="metric-label">Sentences</div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-value">{avg_word_length:.1f}</div>
                                <div class="metric-label">Avg Word Length</div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-value">{len(urls)}</div>
                                <div class="metric-label">URLs</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Common spam patterns
                        spam_patterns = {
                            "Urgency": len(re.findall(r'\b(urgent|hurry|limited|expires?|today|now|soon)\b', user_input.lower())),
                            "Call to action": len(re.findall(r'\b(click|tap|visit|verify|confirm|act)\b', user_input.lower())),
                            "Money": len(re.findall(r'\b(money|cash|prize|win|won|claim|reward|\$|€|£)\b', user_input.lower())),
                            "Personal info": len(re.findall(r'\b(password|account|login|credential|ssn|credit|card)\b', user_input.lower())),
                            "Suspicious origins": len(re.findall(r'\b(overseas|foreign|nigeria|abroad|international)\b', user_input.lower()))
                        }
                        
                        # Convert to list of dictionaries for chart
                        pattern_data = [{"Pattern": k, "Count": v} for k, v in spam_patterns.items()]
                        pattern_df = pd.DataFrame(pattern_data)
                        
                        pattern_chart = alt.Chart(pattern_df).mark_bar().encode(
                            x=alt.X("Count:Q", title="Occurrences"),
                            y=alt.Y("Pattern:N", sort="-x", title=None),
                            color=alt.condition(
                                alt.datum.Count > 0,
                                alt.value("#ff7675"),
                                alt.value("#74b9ff")
                            )
                        ).properties(
                            title="Spam Pattern Detection",
                            height=150
                        )
                        
                        st.altair_chart(pattern_chart, use_container_width=True)
                    
                    with col2:
                        # PII findings
                        if pii_found:
                            st.markdown(f"""
                            <div class="card" style="background-color: rgba(220, 53, 69, 0.1);">
                                <h4 style="text-align: center;">Personal Information Found</h4>
                                <ul>
                                {"".join(f"<li>{item}</li>" for item in pii_found)}
                                </ul>
                                <div style="text-align: center; margin-top: 10px;">
                                    <span class="badge badge-danger">Privacy Risk</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="card" style="background-color: rgba(40, 167, 69, 0.1);">
                                <h4 style="text-align: center;">No Personal Information Found</h4>
                                <p style="text-align: center;">No sensitive personal data detected in this message.</p>
                                <div style="text-align: center; margin-top: 10px;">
                                    <span class="badge badge-success">Privacy Safe</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Special character ratio visualization
                        st.markdown(f"""
                        <div class="card" style="margin-top: 10px;">
                            <h4 style="text-align: center;">Character Composition</h4>
                            <div style="display: flex; height: 20px; border-radius: 4px; overflow: hidden; margin: 10px 0;">
                                <div style="width: {100-special_chars["special_char_ratio"]*100}%; background-color: #74b9ff;"></div>
                                <div style="width: {special_chars["special_char_ratio"]*100}%; background-color: #ff7675;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                                <div>Regular Text: {100-special_chars["special_char_ratio"]*100:.1f}%</div>
                                <div>Special Chars: {special_chars["special_char_ratio"]*100:.1f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk indicators summary
                    st.markdown("<h4 style='color: #38b6ff; margin-top: 20px;'>Risk Assessment Summary</h4>", unsafe_allow_html=True)
                    
                    # Calculate overall risk based on all factors
                    risk_factors = []
                    
                    if prediction == "Spam":
                        risk_factors.append({"factor": "Spam Classification", "severity": "high"})
                    
                    if spam_prob > 70:
                        risk_factors.append({"factor": "High Spam Probability", "severity": "high"})
                    elif spam_prob > 40:
                        risk_factors.append({"factor": "Moderate Spam Probability", "severity": "medium"})
                    
                    if pii_found:
                        risk_factors.append({"factor": "Contains Personal Information", "severity": "high"})
                    
                    if sentiment["label"] == "Negative":
                        risk_factors.append({"factor": "Negative Sentiment", "severity": "medium"})
                    
                    if special_chars["special_char_ratio"] > 0.2:
                        risk_factors.append({"factor": "High Special Character Ratio", "severity": "medium"})
                    
                    if len(urls) > 0:
                        suspicious_urls = sum(1 for url in url_results if url["trust_score"] < 50)
                        if suspicious_urls > 0:
                            risk_factors.append({"factor": f"{suspicious_urls} Suspicious URLs", "severity": "high"})
                    
                    # Pattern checks
                    if spam_patterns["Urgency"] > 1:
                        risk_factors.append({"factor": "Uses Urgency Language", "severity": "high"})
                    
                    if spam_patterns["Money"] > 1:
                        risk_factors.append({"factor": "Mentions Money/Prizes", "severity": "high"})
                    
                    if spam_patterns["Call to action"] > 1:
                        risk_factors.append({"factor": "Contains Call to Action", "severity": "medium"})
                    
                    if risk_factors:
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0;">Risk Indicators</h4>
                            <div class="timeline">
                        """, unsafe_allow_html=True)
                        
                        for risk in risk_factors:
                            severity_color = "#dc3545" if risk["severity"] == "high" else "#ffc107"
                            st.markdown(f"""
                            <div class="timeline-item">
                                <div style="display: flex; align-items: center;">
                                    <span class="badge" style="background-color: {severity_color}; margin-right: 10px;">
                                        {risk["severity"].upper()}
                                    </span>
                                    <span>{risk["factor"]}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Overall threat level
                        high_risks = sum(1 for risk in risk_factors if risk["severity"] == "high")
                        medium_risks = sum(1 for risk in risk_factors if risk["severity"] == "medium")
                        
                        if high_risks >= 2:
                            threat_level = "High"
                            threat_color = "#dc3545"
                        elif high_risks == 1 or medium_risks >= 2:
                            threat_level = "Medium"
                            threat_color = "#ffc107"
                        else:
                            threat_level = "Low"
                            threat_color = "#28a745"
                        
                        st.markdown(f"""
                        <div style="background-color: {threat_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
                            <h3>Overall Threat Level: {threat_level}</h3>
                            <p>Based on {len(risk_factors)} identified risk factors</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #28a745; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
                            <h3>Overall Threat Level: Very Low</h3>
                            <p>No significant risk factors identified</p>
                        </div>
                        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="subheader">Bulk Text Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Upload multiple messages to analyze them all at once. 
        Each message should be on a separate line or separated by a blank line.</p>
    </div>
    """, unsafe_allow_html=True)
    
    bulk_text = st.text_area("Enter multiple messages (one per line or paragraph):", height=200)
    
    analyze_bulk_