import pandas as pd
import re

# Load original dataset
df = pd.read_csv("dataset_tweets.csv", encoding="utf-8-sig")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Clean tweet text
def clean_tweet(text):
    text = re.sub(r"@\w+", "", text)  # remove @usernames
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)  # remove RT
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip().lower()

# Apply cleaning
df["cleaned_tweet"] = df["tweet"].astype(str).apply(clean_tweet)

# Save to new CSV
df.to_csv("cleaned_dataset_tweets.csv", index=False)
print("✅ Cleaned tweets saved to cleaned_dataset_tweets.csv")
