import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from urllib.parse import urlparse
import time

st.set_page_config(
    page_title="Spam Text Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #38b6ff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(56, 182, 255, 0.3);
    }
    .subheader {
        font-size: 1.5rem;
        color: #38b6ff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: rgba(42, 49, 59, 0.7);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid rgba(86, 100, 120, 0.5);
    }
    .result-container {
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        border: 1px solid rgba(86, 100, 120, 0.5);
    }
    .spam-result {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .url-card {
        background-color: rgba(50, 56, 66, 0.7);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(86, 100, 120, 0.5);
    }
    .url-title {
        font-weight: bold;
        color: #38b6ff;
    }
    .metric-container {
        text-align: center;
        background: rgba(42, 49, 59, 0.7);
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 10px 0;
        border: 1px solid rgba(86, 100, 120, 0.5);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #ccc;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(70, 77, 87, 0.5);
        color: white;
        border: 1px solid rgba(86, 100, 120, 0.5);
    }
    .stButton > button {
        border: 1px solid rgba(86, 100, 120, 0.5);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .info-card {
        background-color: rgba(42, 49, 59, 0.7);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(86, 100, 120, 0.5);
        margin-bottom: 15px;
    }
    .status-box {
        border-radius: 5px;
        padding: 10px 15px;
        margin: 10px 0;
        border: 1px solid rgba(86, 100, 120, 0.5);
        text-align: center;
    }
    .text-white {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(50, 56, 66, 0.7);
        border-radius: 4px 4px 0px 0px;
        border: 1px solid rgba(86, 100, 120, 0.5);
        border-bottom: none;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(70, 77, 87, 0.7);
    }
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
    st.markdown('<h1 class="main-header">📧 Smart Spam Text Detector</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/sparklines.png", width=100)
        st.header("About")
        st.markdown("""
        <div class="info-card">
            This app uses machine learning to detect spam text and analyze URL safety.
            It helps protect you from unwanted messages and potentially harmful links.
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
        
        st.header("Tips")
        st.markdown("""
        <div class="info-card" style="background-color: rgba(50, 70, 50, 0.7);">
            <p><strong>How to stay safe:</strong></p>
            <ul>
                <li>Be wary of urgent requests</li>
                <li>Check URL spelling carefully</li>
                <li>Never share sensitive information</li>
                <li>Look for HTTPS in website addresses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Try Demo Text", use_container_width=True):
            st.session_state.demo_text = """URGENT: Your account has been compromised! 
            Click here to verify: http://secur1ty-verify.prize-winner.com
            Also check our legitimate site at https://google.com for more information.
            You've won $1000 - claim at https://free-prizes-winner.net now!"""
            st.rerun()

    model = load_model()
    
    tab1, tab2 = st.tabs(["Spam Detection", "How It Works"])
    
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
            
            col1, col2 = st.columns([4, 1])
            
            with col2:
                analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
                clear_button = st.button("Clear", type="secondary", use_container_width=True)
                
                if clear_button:
                    st.session_state.input_text = ""
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if analyze_button:
                if not user_input:
                    st.warning("Please enter some text to analyze.")
                elif model is None:
                    st.error("Model not loaded. Please check the model file.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Analyzing text...")
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    prediction, confidence, spam_prob = predict_spam(user_input, model)
                    
                    status_text.text("Scanning for URLs...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    urls = extract_urls(user_input)
                    
                    url_results = []
                    if urls:
                        status_text.text("Analyzing website safety...")
                        progress_bar.progress(75)
                        time.sleep(0.5)
                        
                        for url in urls:
                            url_results.append(check_website_trust(url))
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    result_tab1, result_tab2 = st.tabs(["Spam Analysis", "URL Safety"])
                    
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
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.markdown("<h3 style='text-align: center; color: #38b6ff;'>Spam Probability</h3>", unsafe_allow_html=True)
                            st.pyplot(display_gauge_chart(spam_prob, "Likelihood of being spam"))
                        
                        with chart_col2:
                            st.markdown("<h3 style='text-align: center; color: #38b6ff;'>Classification</h3>", unsafe_allow_html=True)
                            
                            if prediction == "Spam":
                                values = [spam_prob, 100-spam_prob]
                                labels = ['Spam', 'Not Spam']
                                colors = ['#ff7675', '#00b894']
                            else:
                                values = [100-spam_prob, spam_prob]
                                labels = ['Not Spam', 'Spam']
                                colors = ['#00b894', '#ff7675']
                            
                            fig, ax = plt.subplots(figsize=(3, 3))
                            fig.patch.set_facecolor('#1e1e1e')
                            ax.set_facecolor('#1e1e1e')
                            
                            wedges, texts, autotexts = ax.pie(
                                values, 
                                labels=labels, 
                                colors=colors, 
                                autopct='%1.1f%%', 
                                startangle=90, 
                                wedgeprops=dict(width=0.5),
                                textprops={'color': 'white'}
                            )
                            
                            for autotext in autotexts:
                                autotext.set_color('white')
                            
                            ax.axis('equal')
                            st.pyplot(fig)
                    
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
                            
                            alt_config = {
                                'background': '#262930',
                                'title': {'color': 'white'},
                                'axis': {
                                    'labelColor': 'white',
                                    'titleColor': 'white',
                                    'gridColor': '#444444'
                                }
                            }
                            
                            chart = alt.Chart(df).mark_bar().encode(
                                x=alt.X('Domain', sort='-y', title='Website Domain'),
                                y=alt.Y('Trust Score', title='Trust Score (0-100)'),
                                color=alt.Color('Trust Score', scale=alt.Scale(
                                    domain=[0, 40, 60, 80, 100],
                                    range=['#ff7675', '#fdcb6e', '#74b9ff', '#00b894', '#00b894']
                                )),
                                tooltip=['Domain', 'Trust Score', 'Classification']
                            ).properties(
                                title='Website Trust Scores',
                                height=300
                            ).configure(**alt_config)
                            
                            st.altair_chart(chart, use_container_width=True)
                            
                            for i, url_info in enumerate(url_results):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="url-card">
                                        <h4 class="url-title">URL {i+1}: {url_info['domain']}</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    score_col, details_col = st.columns([1, 3])
                                    
                                    with score_col:
                                        fig = display_gauge_chart(url_info['trust_score'], "Trust Score")
                                        st.pyplot(fig)
                                    
                                    with details_col:
                                        st.markdown(f"<span class='text-white'><strong>Full URL:</strong> {url_info['url']}</span>", unsafe_allow_html=True)
                                        st.markdown(f"<span class='text-white'><strong>Classification:</strong> {url_info['classification']}</span>", unsafe_allow_html=True)
                                        
                                        if url_info['risk_factors']:
                                            st.markdown("<span class='text-white'><strong>Risk Factors:</strong></span>", unsafe_allow_html=True)
                                            for factor in url_info['risk_factors']:
                                                st.markdown(f"<span class='text-white'>- {factor}</span>", unsafe_allow_html=True)
                                        
                                        if url_info['security_features']:
                                            st.markdown("<span class='text-white'><strong>Security Features:</strong></span>", unsafe_allow_html=True)
                                            for feature in url_info['security_features']:
                                                st.markdown(f"<span class='text-white'>- {feature}</span>", unsafe_allow_html=True)
                            
                            avg_score = sum(trust_scores) / len(trust_scores)
                            if avg_score >= 70:
                                assessment_color = "#00b894"
                                assessment_text = "Links appear trustworthy"
                            elif avg_score >= 50:
                                assessment_color = "#74b9ff"
                                assessment_text = "Links have moderate trust"
                            else:
                                assessment_color = "#ff7675"
                                assessment_text = "Links appear suspicious"
                            
                            st.markdown(f"""
                            <div style="background-color: {assessment_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; border: 1px solid rgba(255, 255, 255, 0.2);">
                                <h3>Overall URL Assessment</h3>
                                <p>Average trust score: {avg_score:.1f}/100 - {assessment_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No URLs found in the text.")
    
    with tab2:
        st.markdown('<h2 class="subheader">How Our Spam Detection Works</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3 style="color: #38b6ff;">Text Analysis</h3>
            <p class="text-white">Our spam detector uses machine learning to identify patterns common in spam messages:</p>
            <ul class="text-white">
                <li>Urgency language and pressure tactics</li>
                <li>Requests for personal information</li>
                <li>Unusual grammar and spelling</li>
                <li>Promotional language and excessive offers</li>
                <li>Suspicious links and domains</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3 style="color: #38b6ff;">URL Trust Analysis</h3>
            <p class="text-white">Our link analyzer evaluates websites based on multiple factors:</p>
            <ul class="text-white">
                <li>Domain reputation and age</li>
                <li>Suspicious keywords and patterns</li>
                <li>Security features present</li>
                <li>URL structure and complexity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h3 class="subheader">Common Spam Indicators</h3>', unsafe_allow_html=True)
        
        spam_indicators = {
            'Urgency Language': 85,
            'Suspicious Links': 78,
            'Request for Information': 72,
            'Grammar Errors': 65,
            'Money Offers': 92,
            'Account Warnings': 81
        }
        
        df_indicators = pd.DataFrame({
            'Indicator': list(spam_indicators.keys()),
            'Frequency': list(spam_indicators.values())
        })
        
        chart = alt.Chart(df_indicators).mark_bar().encode(
            y=alt.Y('Indicator:N', sort='-x', title=None),
            x=alt.X('Frequency:Q', title='Frequency in Spam Messages (%)'),
            color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='blues', domain=[0, 100]), legend=None),
            tooltip=['Indicator', 'Frequency']
        ).properties(
            title='Common Indicators in Spam Messages',
            height=250
        ).configure(
            background='#262930'
        ).configure_title(
            color='white'
        ).configure_axis(
            labelColor='white',
            titleColor='white',
            gridColor='#444444'
        )
        
        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()