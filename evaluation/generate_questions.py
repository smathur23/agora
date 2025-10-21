import pandas as pd
from itertools import combinations
import random

# ================= CONFIG =================
DOCUMENTS_FILE = "data/agora/documents.csv"
SEGMENTS_FILE = "data/agora/segments.csv"
OUTPUT_FILE = "evaluation/generated_questions.csv"
TAG_SEPARATOR = ";"  # separator in your tags column
DATE_COLUMN = "most recent activity date"  # yyyy-mm-dd format
NUM_COMBOS = 800 # number of tag combinations to sample
# ================= LOAD DATA =================
docs = pd.read_csv(DOCUMENTS_FILE)
segments = pd.read_csv(SEGMENTS_FILE)
docs.columns = [c.strip().lower() for c in docs.columns]
segments.columns = [c.strip().lower() for c in segments.columns]

# ================= CLEAN TAGS =================
def clean_tag(tag):
    if ":" in tag:
        return tag.split(":", 1)[1].strip()
    return tag.strip()

docs["tags"] = docs["tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(TAG_SEPARATOR) if t.strip()]
)
segments["tags"] = segments["tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(TAG_SEPARATOR) if t.strip()]
)

# ================= EXTRACT YEAR =================
docs["year"] = pd.to_datetime(docs[DATE_COLUMN], errors="coerce").dt.year
years = sorted(docs["year"].dropna().unique().tolist())

# ================= COLLECT TAGS =================
all_tags = sorted({tag for tags in docs["tags"] for tag in tags})

# ================= TEMPLATES =================
tag_templates = [
    "What are the main {tag} policies?",
    "How does {tag} appear across different AI frameworks?",
    "Which authorities emphasize {tag} in their AI strategies?",
    "How do different regions approach {tag}?",
    "What recent trends are emerging around {tag}?"
]

time_templates = [
    "How has {tag} evolved over time?",
    "What changes in {tag} policies have occurred since {year}?",
    "Show recent {tag} policies published after {year}.",
    "Which authorities have released new policies on {tag} since {year}?",
    "Compare how {tag} was addressed before and after {year}?",
    "Identify new {tag} applications discussed in documents published after {year}."
]

# worthwhile to ensure that retriever gets documents named in question
name_templates = [
    "What AI related policies are in {name}"
]

templates = tag_templates + time_templates + name_templates

# ================= PRECOMPUTE DOCUMENTS =================
# Tag → documents (tag-only)
tag_to_ids = {}
tag_to_docs = {
    tag: docs[docs["tags"].apply(lambda tags: tag in tags)]["AGORA ID"].tolist()
    for tag in all_tags
}
for t in tag_to_docs:
    tag_to_ids[t] = []
    for i, id in enumerate(tag_to_docs[t]):
        tag_to_ids[t].append("document_" + id)
# add segment ids for tags
for tag in all_tags:
    has_tag = segments[segments["tags"].apply(lambda tags: tag in tags)]
    doc_ids = has_tag["Document ID"].tolist()
    segs = has_tag["Segment position"].tolist()
    segment_ids = [f"segment_{doc_ids[i]}_{segs[i]}" for i in range(len(has_tag))]
    tag_to_ids[tag] += segment_ids

# Tag+year → documents (for time-based templates)
tag_year_to_docs = {}
for tag in all_tags:
    # Randomly select 3 years per tag
    sample_years = random.sample(years, min(3, len(years)))
    for year in sample_years:
        docs_list = docs[
            docs["tags"].apply(lambda tags: tag in tags) & (docs["year"] >= year)
        ]["AGORA ID"].tolist()
        for i in range(len(docs_list)):
            docs_list[i] = "document_" + docs_list[i]
        tag_year_to_docs[(tag, year)] = docs_list

# Tag combinations (2-tag combos)

all_combos = list(combinations(all_tags, 2))
sampled_combos = random.sample(all_combos, min(NUM_COMBOS, len(all_combos)))

combo_to_docs = {}
for combo in sampled_combos:
    docs_list = docs[docs["tags"].apply(lambda tags: all(t in tags for t in combo))]["AGORA ID"].tolist()
    if docs_list:
        for i in range(len(docs_list)):
            docs_list[i] = "document_" + docs_list[i]
        combo_to_docs[combo] = docs_list

# ================= GENERATE QUESTIONS =================
rows = []

for template in templates:
    has_year = "{year}" in template
    for tag in all_tags:
        if has_year:
            # Only use the sampled years for this tag
            sampled_years = [y for t, y in tag_year_to_docs.keys() if t == tag]
            for year in sampled_years:
                doc_list = ", ".join(tag_year_to_docs[(tag, year)])
                question = template.format(tag=tag, year=year)
                rows.append({"template": template, "question": question, "relevant_documents": doc_list})
        else:
            doc_list = ", ".join(tag_to_docs.get(tag, []))
            question = template.format(tag=tag)
            rows.append({"template": template, "question": question, "relevant_documents": doc_list})

# ================= TAG COMBINATION QUESTIONS =================
for combo, docs_list in combo_to_docs.items():
    doc_list = ", ".join(docs_list)
    question = f"How do {combo[0]} and {combo[1]} interact in AI regulation?"
    rows.append({"template": "Tag Combination", "question": question, "relevant_documents": doc_list})

# ================= OUTPUT CSV =================
output_df = pd.DataFrame(rows)
output_df = output_df.sort_values("template").reset_index(drop=True)
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Generated {len(output_df)} questions and saved to {OUTPUT_FILE}")