import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEGMENTS_CSV = "data/agora/segments.csv"
DOCUMENTS_CSV = "data/agora/documents.csv"
FULLTEXT_DIR = "data/agora/fulltext"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")


# ---------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------
segments = pd.read_csv(SEGMENTS_CSV)
documents = pd.read_csv(DOCUMENTS_CSV)

print("Loaded:")
print(f"  {len(segments)} segment rows")
print(f"  {len(documents)} document rows")


# ---------------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------------
segments["Text"] = segments["Text"].fillna("")

# Parse tags
segments["Tags"] = (
    segments["Tags"]
    .fillna("")
    .astype(str)
    .apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
)

# Convert dates
if "Most recent activity date" in documents.columns:
    documents["Most recent activity date"] = pd.to_datetime(
        documents["Most recent activity date"], errors="coerce"
    )


# ---------------------------------------------------------------------
# 1. Segments per document
# ---------------------------------------------------------------------
seg_per_doc = segments.groupby("Document ID").size().reset_index(name="num_segments")

plt.figure(figsize=(10,5))
sns.histplot(seg_per_doc["num_segments"], bins=40)
plt.title("Segments per Document")
plt.xlabel("Number of Segments")
plt.ylabel("Document Count")
plt.savefig(f"{OUTPUT_DIR}/segments_per_document.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 2. Segment Length Distribution (WORDS)
# ---------------------------------------------------------------------
segments["word_length"] = segments["Text"].apply(lambda s: len(s.split()))

plt.figure(figsize=(10,5))
sns.histplot(segments["word_length"], bins=50)
plt.title("Segment Length Distribution (Words)")
plt.xlabel("Segment Length (words)")
plt.ylabel("Frequency")
plt.savefig(f"{OUTPUT_DIR}/segment_length_distribution_words.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 3. Tag Frequency (Top 20)
# ---------------------------------------------------------------------
all_tags = [tag for tags in segments["Tags"] for tag in tags]
tag_counts = pd.Series(all_tags).value_counts().head(20)

plt.figure(figsize=(12,6))
sns.barplot(x=tag_counts.index, y=tag_counts.values)
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Tags")
plt.ylabel("Count")
plt.xlabel("Tag")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_frequency_top20.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 4. Non-operative & Not AI-related distributions
# ---------------------------------------------------------------------
if "Non-operative" in segments.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Non-operative", data=segments)
    plt.title("Distribution of Non-operative Segments")
    plt.savefig(f"{OUTPUT_DIR}/non_operative_distribution.png", dpi=200)
    plt.close()

if "Not AI-related" in segments.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Not AI-related", data=segments)
    plt.title("Distribution of Not AI-related Segments")
    plt.savefig(f"{OUTPUT_DIR}/not_ai_related_distribution.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# 5. Authority Distribution (Top 20)
# ---------------------------------------------------------------------
if "Authority" in documents.columns:
    top_auth = documents["Authority"].value_counts().head(20)

    plt.figure(figsize=(12,6))
    sns.barplot(y=top_auth.index, x=top_auth.values)
    plt.title("Top 20 Authorities")
    plt.xlabel("Count")
    plt.ylabel("Authority")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/authority_top20.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# 6. Most Recent Activity Date
# ---------------------------------------------------------------------
if "Most recent activity date" in documents.columns:
    valid_dates = documents["Most recent activity date"].dropna()

    plt.figure(figsize=(10,5))
    sns.histplot(valid_dates, bins=40)
    plt.title("Most Recent Activity Date Distribution")
    plt.xlabel("Date")
    plt.ylabel("Document Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/activity_date_distribution.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# 7. Full Document Length (based on words)
# ---------------------------------------------------------------------
doc_lengths = []
missing_docs = 0

if "AGORA ID" in documents.columns:
    for doc_id in documents["AGORA ID"]:
        path = os.path.join(FULLTEXT_DIR, f"{doc_id}.txt")
        if not os.path.exists(path):
            missing_docs += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            doc_lengths.append(len(text.split()))

if doc_lengths:
    plt.figure(figsize=(10,5))
    sns.histplot(doc_lengths, bins=50)
    plt.title("Full Document Length Distribution (Words)")
    plt.xlabel("Document Length (words)")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUTPUT_DIR}/document_length_distribution.png", dpi=200)
    plt.close()

print(f"Missing fulltext files: {missing_docs}")


# ---------------------------------------------------------------------
# 8. Export Summary Statistics
# ---------------------------------------------------------------------
summary = {
    "num_documents": len(documents),
    "num_segments": len(segments),
    "avg_segments_per_doc": seg_per_doc["num_segments"].mean(),
    "avg_segment_length_words": segments["word_length"].mean(),
    "num_unique_tags": len(pd.Series([tag for tags in segments["Tags"] for tag in tags]).unique()),
    "top_20_tags": tag_counts.to_dict(),
}

summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv")

print("EDA complete. Figures saved to ./figures/")
