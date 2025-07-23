# app.py
 
import streamlit as st
import joblib
import pandas as pd
import re
from urllib.parse import urlparse
import requests
import tweepy
import time
from fpdf import FPDF
import altair as alt
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
 
# --- Load Models ---
phishing_model = joblib.load("best_phishing_model.pkl")
cyber_model = joblib.load("best_cyberbullying_model.pkl")  # SVM model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
 
# --- Twitter API ---
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKwe3AEAAAAAAzRn%2BpZZkVomWfzA90ZZINKE2js%3DqnUshoysqaiIlDpRVlzqFcKRzQFUsq6vY01MvRVdAtoue7dVMM"
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
 
# --- Streamlit Setup ---
st.set_page_config(page_title="Threat Intelligence", layout="wide")
st.title("🛡️ Threat Intelligence in Social Sphere")
 
# --- Helper Functions ---
def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return pd.DataFrame([{
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "num_dots": url.count('.'),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(bool(re.match(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', hostname))),
        "count_slash": url.count('/'),
        "count_question": url.count('?'),
        "count_equal": url.count('='),
        "count_at": url.count('@'),
        "count_percent": url.count('%'),
        "count_hyphen": url.count('-')
    }])
 
def expand_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url
 
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text.strip().lower()
 
def predict_tweet(text):
    return cyber_model.predict([text])[0]
 
def classify_url(url):
    features = extract_url_features(url)
    return "Phishing" if phishing_model.predict(features)[0] == 1 else "Safe"
 
def generate_summary_markdown(tweets_data, tweet_stats, phish_stats):
    cb_yes, cb_no, phish_yes, phish_no = [], [], [], []
    for i, tweet in enumerate(tweets_data, 1):
        label = tweet["result"]
        (cb_yes if label == "Cyberbullying" else cb_no).append(f"Tweet {i}")
        for url, verdict in tweet["urls"]:
            (phish_yes if verdict == "Phishing" else phish_no).append(f"Tweet {i}")
    return f"""
### 📝 Summary of the Tweets Analysis
 
**{tweet_stats['Cyberbullying']}** tweets flagged as 🚫 Cyberbullying, **{tweet_stats['Not Cyberbullying']}** tweets safe (👍 Not Cyberbullying).
- 🚫 Cyberbullying: {", ".join(cb_yes) if cb_yes else "None"}
- 👍 Not Cyberbullying: {", ".join(cb_no) if cb_no else "None"}
 
**{phish_stats['Phishing']}** URLs were ⚠️ Phishing, **{phish_stats['Safe']}** URLs 🔐 Safe.
- ⚠️ Phishing: {", ".join(phish_yes) if phish_yes else "None"}
- 🔐 Not Phishing: {", ".join(phish_no) if phish_no else "None"}
"""
 
def save_pdf_summary(tweets_data, tweet_stats, phish_stats, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln()
    for i, tweet in enumerate(tweets_data, 1):
        pdf.multi_cell(0, 8, f"Tweet {i}: {tweet['text']}")
        pdf.cell(0, 8, f"Result: {tweet['result']}", ln=True)
        for url, verdict in tweet["urls"]:
            pdf.cell(0, 8, f"URL: {url} → {verdict}", ln=True)
        pdf.ln(4)
    filename = "tweet_summary.pdf"
    pdf.output(filename)
    return filename
 
# --- Session State Init ---
st.session_state.setdefault("tweet_stats", {"Cyberbullying": 0, "Not Cyberbullying": 0})
st.session_state.setdefault("phish_stats", {"Phishing": 0, "Safe": 0})
st.session_state.setdefault("tweets_data", [])
st.session_state.setdefault("chat_history", [])
 
# --- UI Layout ---
left_col, right_col = st.columns([1, 2])
 
with right_col:
    st.header("🐦 Analyze Tweets from Twitter Account")
    with st.form("fetch_form"):
        username = st.text_input("Twitter username (without @)", "fyptiss2025")
        num_tweets = st.slider("Number of tweets", min_value=1, max_value=30, value=10)
        date_range = st.date_input("Select date range", (datetime.utcnow() - timedelta(days=7), datetime.utcnow()))
        run = st.form_submit_button("Fetch and Analyze")
 
    if run:
        start_time = time.time()
        with st.spinner("🧠 Fetching tweets... This might take a while due to Twitter rate limits."):
            st.session_state["tweet_stats"] = {"Cyberbullying": 0, "Not Cyberbullying": 0}
            st.session_state["phish_stats"] = {"Phishing": 0, "Safe": 0}
            st.session_state["tweets_data"] = []
            try:
                user = client.get_user(username=username)
                tweets = client.get_users_tweets(
                    id=user.data.id,
                    max_results=num_tweets,
                    exclude=["retweets", "replies"],
                    start_time=date_range[0].isoformat() + "T00:00:00Z",
                    end_time=date_range[1].isoformat() + "T23:59:59Z"
                )
                elapsed = int(time.time() - start_time)
                if elapsed > 30:
                    st.info(f"⏳ Fetching tweets took {elapsed} seconds due to Twitter rate limit.")
                elif elapsed > 10:
                    st.info(f"⏳ Fetched tweets in {elapsed} seconds.")
 
                if tweets.data:
                    for tweet in tweets.data:
                        text = tweet.text
                        label = predict_tweet(clean_tweet(text))
                        result = "Cyberbullying" if label == 1 else "Not Cyberbullying"
                        st.session_state["tweet_stats"][result] += 1
                        urls = re.findall(r"http\S+", text)
                        phishing_results = []
                        for url in urls:
                            full_url = expand_url(url)
                            verdict = classify_url(full_url)
                            phishing_results.append((full_url, verdict))
                            st.session_state["phish_stats"][verdict] += 1
                        st.session_state["tweets_data"].append({
                            "text": text,
                            "result": result,
                            "urls": phishing_results
                        })
                    st.success("✅ Tweets analyzed successfully.")
                else:
                    st.warning("⚠️ No tweets found.")
            except Exception as e:
                elapsed = int(time.time() - start_time)
                st.error(f"❌ Error fetching tweets: {e}")
                st.info(f"🕒 Total elapsed time: {elapsed} seconds")
 
    if st.session_state["tweets_data"]:
        st.subheader("📝 Tweet Classification Results")
        for i, tweet in enumerate(st.session_state["tweets_data"], 1):
            with st.expander(f"Tweet {i}"):
                st.write(tweet["text"])
                st.write("Cyberbullying:", "🚫 Yes" if tweet["result"] == "Cyberbullying" else "✅ No")
                for url, verdict in tweet["urls"]:
                    st.write(f"URL: {url} → {'⚠️' if verdict == 'Phishing' else '✅'} {verdict}")
        if st.button("📄 Generate PDF Report"):
            summary = generate_summary_markdown(
                st.session_state["tweets_data"],
                st.session_state["tweet_stats"],
                st.session_state["phish_stats"]
            )
            path = save_pdf_summary(
                st.session_state["tweets_data"],
                st.session_state["tweet_stats"],
                st.session_state["phish_stats"],
                summary
            )
            with open(path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="tweet_summary.pdf")
 
with left_col:
    st.header("📊 Threat Overview")
    st.altair_chart(
        alt.Chart(pd.DataFrame(st.session_state["tweet_stats"].items(), columns=["Type", "Count"]))
        .mark_bar().encode(x="Type", y="Count", color="Type"),
        use_container_width=True
    )
    st.altair_chart(
        alt.Chart(pd.DataFrame(st.session_state["phish_stats"].items(), columns=["Type", "Count"]))
        .mark_arc().encode(theta="Count", color="Type"),
        use_container_width=True
    )
    st.markdown(generate_summary_markdown(
        st.session_state["tweets_data"],
        st.session_state["tweet_stats"],
        st.session_state["phish_stats"]
    ))
 
# --- Sidebar Chatbot ---
with st.sidebar:
    st.header("💬 Manual Classification Chatbot")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask me anything...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        intents = {
            "greeting": ["hi", "hello", "hey"],
            "cyber_check": ["is this cyberbullying", "is this offensive", "is this bullying"],
            "url_check": ["is this link phishing", "check this url", "is this safe link"]
        }
        responses = {
            "greeting": "👋 Hello! How can I assist you today?",
            "cyber_safe": "✅ That input appears to be safe.",
            "cyber_bully": "🚫 That input may be cyberbullying.",
            "url_safe": "🔐 The link appears safe.",
            "url_phish": "⚠️ Warning: This link may be phishing.",
            "unknown": "🤖 I'm not sure. Try submitting a tweet or URL."
        }
        def get_intent(text):
            embed = embedding_model.encode(text, convert_to_tensor=True)
            scores = {
                intent: util.cos_sim(embed, embedding_model.encode(phrases, convert_to_tensor=True)).max().item()
                for intent, phrases in intents.items()
            }
            return max(scores, key=scores.get) if max(scores.values()) > 0.6 else None
        try:
            intent = get_intent(prompt.lower())
            if intent == "greeting":
                reply = responses["greeting"]
            elif intent == "url_check" or re.match(r'^https?://', prompt):
                verdict = classify_url(expand_url(prompt))
                reply = responses["url_safe"] if verdict == "Safe" else responses["url_phish"]
            elif intent == "cyber_check" or len(prompt.split()) <= 10:
                label = predict_tweet(prompt)
                reply = responses["cyber_bully"] if label == 1 else responses["cyber_safe"]
            else:
                label = predict_tweet(prompt)
                reply = responses["cyber_bully"] if label == 1 else responses["cyber_safe"]
        except:
            reply = "⚠️ I couldn't analyze that input."
        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
 