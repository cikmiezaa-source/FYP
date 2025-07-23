import pandas as pd

# Load cleaned tweets
df = pd.read_csv("cleaned_dataset_tweets.csv")

# Manually define extra toxic short phrases 
extra_samples = pd.DataFrame({
    "tweet": [
        "you are dumb", "you suck", "idiot", "stupid", "loser", "shut up",
        "you are an idiot", "go to hell", "you moron", "dumbass", "freak", "get lost"
    ],
    "class": [1]*12,  # 1 = Cyberbullying
})

# Clean them the same way
def clean_tweet(text):
    import re
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

extra_samples["cleaned_tweet"] = extra_samples["tweet"].apply(clean_tweet)

# Append to original dataset
df_aug = pd.concat([df, extra_samples], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Save to new CSV
df_aug.to_csv("augmented_dataset_tweets.csv", index=False)
print("✅ Augmented dataset saved to augmented_dataset_tweets.csv")
