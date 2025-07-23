import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

# Class distribution data
labels = ['Hate Speech (0)', 'Offensive Language (1)', 'Neither (2)']
counts = [1430, 19190, 4163]
 
# Plotting
plt.figure(figsize=(8,6))
bars = plt.bar(labels, counts, color=['#e74c3c','#f39c12','#2ecc71'])
 
# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 200, f'{yval}', ha='center', va='bottom', fontsize=11)
 
plt.title('Figure 4.1: Class Distribution of Hate Speech Dataset', fontsize=14)
plt.ylabel('Number of Tweets')
plt.xlabel('Class Label')
plt.ylim(0, max(counts) + 3000)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -------- Step 1: Load & Clean Dataset -------- #
df = pd.read_csv("cleaned_dataset_tweets.csv", encoding="utf-8-sig")
print("📋 Original Columns:", df.columns.tolist())
df.columns = df.columns.str.strip().str.lower()
print("✅ Cleaned Columns:", df.columns.tolist())

# Binary label
df["label"] = df["class"].apply(lambda x: 1 if x in [0, 1] else 0)

# Clean tweets
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["cleaned_tweet"] = df["cleaned_tweet"].astype(str).apply(clean_tweet)
df = df[["cleaned_tweet", "label"]].dropna()

# -------- Step 2: Use full imbalanced dataset (no manual balancing) -------- #
X = df["cleaned_tweet"]
y = df["label"]

# Split original imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(f"🔍 Support in test set: {Counter(y_test)}")
print(f"✔ y_train (before SMOTE): {Counter(y_train)}")


# -------- Step 3: TF-IDF + SMOTE -------- #
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

print("🔢 Training samples before SMOTE:", X_train_vec.shape[0])
print("🔢 Training samples after SMOTE:", X_train_res.shape[0])
print("✔ y_train (before SMOTE):", Counter(y_train))
print("✔ y_train_res (after SMOTE):", Counter(y_train_res))

# -------- Step 3B: Inspect SMOTE dataset -------- #
resampled_text = tfidf.inverse_transform(X_train_res)

df_smote = pd.DataFrame({
    "synthetic_text_approx": [" ".join(tokens) for tokens in resampled_text],
    "label": y_train_res
})

df_smote.to_csv("cyberbullying_smote_balanced_data.csv", index=False)
print("📁 Saved synthetic SMOTE dataset as: smote_balanced_data.csv")

# -------- Step 4: Visualize Class Distribution -------- #
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

before_counts = Counter(y_train)
after_counts = Counter(y_train_res)

axes[0].bar(["Safe", "Cyberbullying"], [before_counts[0], before_counts[1]], color=["green", "red"])
axes[0].set_title("🔍 Before SMOTE")
axes[0].set_ylabel("Samples")
axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

axes[1].bar(["Safe", "Cyberbullying"], [after_counts[0], after_counts[1]], color=["green", "red"])
axes[1].set_title("🔄 After SMOTE")
axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

plt.suptitle("Class Distribution Comparison")
plt.tight_layout()
plt.show()

# -------- Step 5: Train Models -------- #
models = {
    "SVM (Linear)": LinearSVC(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

best_f1 = 0
best_model = None
best_model_name = ""

for name, clf in models.items():
    print(f"\n🔍 Training {name}...")

    model = clone(clf)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_vec)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"✅ F1-score (macro): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Cyberbullying"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Cyberbullying"],
                yticklabels=["Safe", "Cyberbullying"])
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if f1 > best_f1:
        best_f1 = f1
        best_model = Pipeline([
            ('tfidf', tfidf),
            ('clf', model)
        ])
        best_model_name = name

# -------- Step 6: Save Best Model -------- #
filename = "best_cyberbullying_model.pkl"
joblib.dump(best_model, filename)
print(f"\n💾 Saved best model: {best_model_name} → `{filename}` with F1-score: {best_f1:.4f}")

# -------- Step 7: Test a Tweet -------- #
def test_tweet(tweet_text, model_file="best_cyberbullying_model.pkl"):
    print(f"\n🧪 Testing tweet: \"{tweet_text}\"")

    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        print("❌ Error: Trained model not found. Please train the model first.")
        return

    # Clean the input tweet
    def clean_input(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    cleaned = clean_input(tweet_text)
    prediction = model.predict([cleaned])[0]

    result = "Cyberbullying 🚨" if prediction == 1 else "Safe ✅"
    print(f"🔎 Prediction: {result}")

test_tweet("Hi pretty. You are lucky to have a beloved friends")
test_tweet("You're such a loser, your body is so fat and fucking smelly girl .")

