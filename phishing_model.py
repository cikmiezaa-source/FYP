#phishing_model.py
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
 
 # Data
labels = ['Legitimate (0)', 'Phishing (1)']
counts = [5486, 5569]
colors = ['#4CAF50', '#F44336']
 
# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts, color=colors)
 
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 50, str(height), ha='center', va='bottom', fontsize=11)
 
# Formatting
plt.title('Figure 4.2: Class Distribution of Web Page Phishing Detection Dataset', fontsize=13)
plt.ylabel('Number of Samples')
plt.ylim(0, 6000)
plt.grid(axis='y', linestyle='--', alpha=0.6)
 
# Display plot
plt.tight_layout()
plt.show()

# -------- Step 1: Feature Extraction -------- #
def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname if parsed.hostname else ""
 
    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "num_dots": url.count('.'),
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_ip": 1 if re.match(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', hostname) else 0,
        "count_slash": url.count('/'),
        "count_question": url.count('?'),
        "count_equal": url.count('='),
        "count_at": url.count('@'),
        "count_percent": url.count('%'),
        "count_hyphen": url.count('-')
    }
    return features
 
# -------- Step 2: Load & Prepare Dataset -------- #
df = pd.read_csv("dataset_phishing.csv")
df.dropna(inplace=True)
 
# Normalize labels
df['status'] = df['status'].astype(str).str.strip().str.lower()
label_map = {'legitimate': 0, 'phishing': 1, 'yes': 1, 'no': 0, '1': 1, '0': 0}
df['status'] = df['status'].map(label_map)
df.dropna(subset=['status'], inplace=True)
df['status'] = df['status'].astype(int)
 
# Extract features for all URLs
features_df = df['url'].apply(extract_url_features).apply(pd.Series)
X = features_df
y = df['status']
 
# -------- Step 3: Split Data -------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# -------- Step 4: Train and Compare Models -------- #
models = {
    "SVM (Linear)": LinearSVC(max_iter=5000, dual=False, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
 
best_acc = 0
best_model = None
best_model_name = ""
 
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-score (macro): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
 
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name
 
# -------- Step 5: Save Best Model -------- #
filename = "best_phishing_model.pkl"
joblib.dump(best_model, filename)
print(f"\n💾 Saved best model: {best_model_name} → `{filename}` with accuracy {best_acc:.4f}")

# -------- Step 6: Test a URL -------- #
def test_url(url, model_file="best_phishing_model.pkl"):
    print(f"\n🔍 Testing URL: {url}")
    
    # Load the model
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        print("❌ Error: Trained model not found. Please run the training process first.")
        return
    
    # Extract features from URL
    features = extract_url_features(url)
    features_df = pd.DataFrame([features])
    
    # Predict
    prediction = model.predict(features_df)[0]
    result = "Phishing 🚨" if prediction == 1 else "Legitimate ✅"
    print(f"🔎 Prediction: {result}")

test_url("https://study.uitm.edu.my/")
test_url("https://support-appleld.com.secureupdate.duilawyeryork.com/ap/c05f378d59c6ba8?cmd=_update&dispatch=c05f378d59c6ba805&locale=_US")
