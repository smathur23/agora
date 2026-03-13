from retriever_finetuning.prompts import comparison_of_authorities, compare_docs, specific_doc, status_of_tags, system_prompt, trends_over_tags
import pandas as pd
import os
import itertools
import random
from retriever_finetuning.llm import run_chat_completion

# ======================================================= #
# Get dataframes                                          #
# ======================================================= #
docs = pd.read_csv("data/agora/documents.csv")
short_docs = pd.read_csv("data/agora/documents.csv")
segs = pd.read_csv("data/agora/segments.csv")
MAX_DOC_LEN = 1000
# preprocess docs to remove docs that are too long, have no text, or have no name
short_docs = short_docs[short_docs["AGORA ID"].apply(lambda x: os.path.exists(f"data/agora/fulltext/{x}.txt"))]
doc_texts = {}
for idx, row in short_docs.iterrows():
    doc_id = row["AGORA ID"]
    with open(f"data/agora/fulltext/{doc_id}.txt", "r") as f:
        text = f.read()
    if len(text.split(" ")) <= MAX_DOC_LEN:
        doc_texts[row["AGORA ID"]] = text
short_docs = short_docs[short_docs["AGORA ID"].apply(lambda x: x in doc_texts)]
short_docs = short_docs[short_docs["Official name"] != ""]

# ======================================================= #
# Clean tags, dates, authorities                          #
# ======================================================= #

def clean_tag(tag):
    if ":" in tag:
        return tag.split(":", 1)[1].strip()
    return tag.strip()

docs["Tags"] = docs["Tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(";") if t.strip()]
)

segs["Tags"] = segs["Tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(";") if t.strip()]
)

# ======================================================== #
# Collect documents with tags, years, authorities          #
# ======================================================== #
# get tags and define generic tag types
all_tags = sorted({tag for tags in docs["Tags"] for tag in tags})
tag_types = ["Risk factors", "Applications", "Incentives", "Strategies", "Harms"]
# create lookup dicts to quickly get which docs and segments have a given tag
has_tag = {tag: docs[docs["Tags"].apply(lambda tags: tag in tags)]["AGORA ID"].tolist() for tag in all_tags}
has_tag_segs = {}
for tag in all_tags:
    segs_with_tag = segs[segs["Tags"].apply(lambda tags: tag in tags)]
    seg_doc_ids = segs_with_tag["Document ID"].tolist()
    seg_pos = segs_with_tag["Segment position"].tolist()
    has_tag_segs[tag] = [f"segment_{seg_doc_ids[i]}_{seg_pos[i]}" for i in range(len(segs_with_tag))]
# Get years of docs
docs["year"] = pd.to_datetime(docs["Most recent activity date"], errors="coerce").dt.year
years = sorted(docs["year"].dropna().unique().tolist())
has_year = {year: docs[docs["year"] == year]["AGORA ID"].tolist() for year in years}
all_specific_authorities = sorted({auth for auth in docs["Authority"]})
# For authorities, create broad authorities (like US federal govt)
# the key is the generic auth, the value is the specific values from the data that correspond to it 
general_authorities = {}
general_authorities["US local governments"] = ['Boise, ID', 'Boston, MA', 'Indianapolis, IN', 'Kendall County, IL', 'Long Beach, CA', 'Los Angeles, CA', 'Miami Dade County, FL', 'New York, NY', 'San Francisco, CA', 'San Jose, CA', 'Santa Cruz County, CA', 'Seattle, WA', 'St. Louis, MO', 'Tempe, AZ', 'Tulsa, OK']
general_authorities["US federal government"] = ['Consumer Financial Protection Bureau', 'Copyright Office, Library of Congress', 'Department of Agriculture', 'Department of Commerce', 'Department of Defense', 'Department of Education', 'Department of Health and Human Services', 'Department of Housing and Urban Development', 'Department of Transportation', 'Department of the Treasury',  'Executive Office of the President', 'Federal Election Commission', 'Federal government',  'Food and Drug Administration', 'National Institute of Standards and Technology', 'Office of Management and Budget', 'Office of Personnel Management', 'Office of Science and Technology Policy', 'United States Congress']
general_authorities["US state governments"] = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Louisiana', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',  'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island','South Carolina', 'South Dakota',  'State governments',  'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
general_authorities["United States"] = general_authorities["US local governments"] + general_authorities["US federal government"] + general_authorities["US state governments"]
general_authorities["China"] = ['Chinese provincial and local governments', 'Chinese central government']
general_authorities["Foreign countries"] = ['Chinese central government', 'Chinese provincial and local governments',  'European Union', 'Government of Australia', 'Government of Canada', 'Government of Israel', 'Government of New Zealand', 'Government of Singapore', 'Government of Turkey', 'Government of the United Arab Emirates', 'Government of the United Kingdom']
general_authorities["Multinational"] = ['European Union', 'OECD', 'Other multinational', 'United Nations']
# create authority lookup dicts
has_specific_auth = {auth: docs[docs["Authority"] == auth]["AGORA ID"].tolist() for auth in all_specific_authorities}
has_general_auth = {auth: docs[docs["Authority"].apply(lambda x: x in general_authorities[auth])]["AGORA ID"].tolist() for auth in general_authorities}

# ========================================================= #
# Prompt generation functions                               #
# ========================================================= #

# Types of questions and max number to be generated
SINGLE_DOC = 500                       
DOC_COMP = 500                                  
TAG_STATUS = 400
TAG_TRENDS = 1200 # 6 combinations of questions, 100 each
AUTHORITY_COMP = 400

# Specific policy doc questions
def generate_single_doc_prompts(num):
    sd_rows = short_docs.sample(n=min(num, len(short_docs)))
    sd_texts = []
    sd_titles = []
    sd_relevant_ids = []
    # Get texts, titles, and doc_ids
    for idx, row in sd_rows.iterrows():
        doc_id = row["AGORA ID"]
        title = row["Official name"]
        text = doc_texts[doc_id]
        sd_relevant_ids.append([f"document_{doc_id}"])
        sd_texts.append(text)
        sd_titles.append(title)
    count = len(sd_relevant_ids)
    # generate prompts for each doc
    sd_prompts = [specific_doc(sd_titles[i], sd_texts[i]) if i % 2 == 0 else specific_doc(sd_titles[i], sd_texts[i], name_provided=True) for i in range(count)]
    # add segments for each doc to set of relevant ids
    for i, doc_id in enumerate(sd_relevant_ids):
        agora_id = doc_id[0][9:]
        segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
        for s in segments:
            sd_relevant_ids[i].append(f"segment_{agora_id}_{s}")
    return sd_prompts, sd_relevant_ids


# document comps
def generate_doc_comparison_prompts(num):
    all_doc_pairs = list(itertools.combinations(short_docs.index, 2))
    dc_rows_pairs = random.sample(all_doc_pairs, min(num, len(all_doc_pairs)))
    dc_texts = []
    dc_titles = []
    dc_relevant_ids = []
    # Get texts, titles, doc_ids
    for i,j in dc_rows_pairs:
        row1 = docs.loc[i]
        row2 = docs.loc[j]
        doc1_id = row1["AGORA ID"]
        doc2_id = row2["AGORA ID"]
        title1 = row1["Official name"]
        title2 = row2["Official name"]
        text1 = doc_texts[doc1_id]
        text2 = doc_texts[doc2_id]
        dc_relevant_ids.append([f"document_{doc1_id}", f"document_{doc2_id}"])
        dc_texts.append([text1, text2])
        dc_titles.append([title1, title2])
    count = len(dc_relevant_ids)
    # Generate prompts
    dc_prompts = [compare_docs(dc_titles[i][0], dc_texts[i][0], dc_titles[i][1], dc_texts[i][1]) if i % 2 == 0 else compare_docs(dc_titles[i][0], dc_texts[i][0], dc_titles[i][1], dc_texts[i][1], name_provided=True) for i in range(count)]
    # Include segment ids for each doc as well
    for i, doc_ids in enumerate(dc_relevant_ids):
        agora_id1 = doc_ids[0][9:]
        agora_id2 = doc_ids[1][9:]
        segments1 = segs[segs["Document ID"] == int(agora_id1)]["Segment position"].tolist()
        segments2 = segs[segs["Document ID"] == int(agora_id2)]["Segment position"].tolist()
        for s in segments1:
            dc_relevant_ids[i].append(f"segment_{agora_id1}_{s}")
        for s in segments2:
            dc_relevant_ids[i].append(f"segment_{agora_id2}_{s}")
    return dc_prompts, dc_relevant_ids

# Tag status
def generate_tag_status_prompts(num):
    # First half: specific authorities
    pairs_1 = [[auth, tag] for auth in all_specific_authorities for tag in all_tags]
    random.shuffle(pairs_1)
    # Second half: generic authorities
    pairs_2 = [[auth, tag] for auth in general_authorities for tag in all_tags]
    random.shuffle(pairs_2)
    # get relevant doc ids
    ts_relevant_ids = []
    ts_pairs = []
    for auth, tag in pairs_1:
        ids = set(has_specific_auth[auth]).intersection(set(has_tag[tag]))
        if len(ids) != 0:
            ts_relevant_ids.append([f"document_{doc_id}" for doc_id in list(ids)])
            ts_pairs.append([auth, tag])
            if len(ts_relevant_ids) == num // 2:
                break
    for auth, tag in pairs_2:
        ids = set(has_general_auth[auth]).intersection(set(has_tag[tag]))
        if len(ids) != 0:
            ts_relevant_ids.append([f"document_{doc_id}" for doc_id in list(ids)])
            ts_pairs.append([auth, tag])
            if len(ts_relevant_ids) == num:
                break
    # get segment ids from those docs
    for i, doc_ids in enumerate(ts_relevant_ids):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[ts_pairs[i][1]]: segment_ids.append(seg_id)
        ts_relevant_ids[i] += segment_ids 
    # Generate prompts
    ts_prompts = [status_of_tags(tag, auth) for auth, tag in ts_pairs]
    return ts_prompts, ts_relevant_ids


# authority comparison
def generate_authority_comparison_prompts(num):
    ac_pairs_1 = [random.sample(all_specific_authorities, 2) for i in range(num // 2)]
    ac_pairs_2 = [random.sample(list(general_authorities.keys()), 2) for i in range(num // 2)]
    ac_pairs = ac_pairs_1 + ac_pairs_2
    # get relevant doc ids:
    ac_relevant_ids_1 = [[f"document_{doc_id}" for doc_id in has_specific_auth[auth1]] + [f"document_{doc_id}" for doc_id in has_specific_auth[auth2]] for auth1, auth2 in ac_pairs_1]
    ac_relevant_ids_2 = [[f"document_{doc_id}" for doc_id in has_general_auth[auth1]] + [f"document_{doc_id}" for doc_id in has_general_auth[auth2]] for auth1, auth2 in ac_pairs_2]
    ac_relevant_ids = ac_relevant_ids_1 + ac_relevant_ids_2
    # get segment ids from the doc ids
    for i, doc_ids in enumerate(ac_relevant_ids):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                segment_ids.append(seg_id)
        ac_relevant_ids[i] += segment_ids 
    # generate prompts
    ac_prompts = [comparison_of_authorities(auth1, auth2) for auth1, auth2 in ac_pairs]
    return ac_prompts, ac_relevant_ids 

# trends over tags
def generate_trends_prompts(num):
    tt_tags = random.sample(all_tags, min(num // 6, len(all_tags)))
    tt_authorities_1 = random.sample(all_specific_authorities, min(num // 12, len(all_specific_authorities)))
    tt_authorities_2 = [random.sample(list(general_authorities.keys()), 1)[0] for i in range(num // 12)]
    # For the pairs and triplets, generate more than the number of prompt because a lot of them aren't gonna have any matches
    tt_tags_years = [[random.sample(all_tags, 1)[0], random.sample(years, 1)[0]] for i in range(TAG_TRENDS * 2)]
    tt_tags_authorities_1 = [[random.sample(all_tags, 1)[0], random.sample(all_specific_authorities, 1)[0]] for i in range(TAG_TRENDS)]
    tt_tags_authorities_2 = [[random.sample(all_tags, 1)[0], random.sample(list(general_authorities.keys()), 1)[0]] for i in range(TAG_TRENDS)]
    tt_authorities_years_1 = [[random.sample(all_specific_authorities, 1)[0], random.sample(years, 1)[0]] for i in range(TAG_TRENDS)] 
    tt_authorities_years_2 = [[random.sample(list(general_authorities.keys()), 1)[0], random.sample(years, 1)[0]] for i in range(TAG_TRENDS)]
    tt_tay_1 = [[random.sample(all_tags, 1)[0], random.sample(all_specific_authorities, 1)[0], random.sample(years, 1)[0]] for i in range(TAG_TRENDS * 4)]
    tt_tay_2 = [[random.sample(all_tags, 1)[0], random.sample(list(general_authorities.keys()), 1)[0], random.sample(years, 1)[0]] for i in range(TAG_TRENDS * 4)]
    # get relevant ids for tt_tags
    tt_tags_relevant_ids = [[f"document_{doc_id}" for doc_id in has_tag[tag]] for tag in tt_tags]
    for i, doc_ids in enumerate(tt_tags_relevant_ids):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tags[i]]: segment_ids.append(seg_id)
        tt_tags_relevant_ids[i] += segment_ids 
    # get relevant ids for tt_authorities
    tt_authorities_relevant_ids = [[f"document_{doc_id}" for doc_id in has_specific_auth[auth]] for auth in tt_authorities_1]
    tt_authorities_relevant_ids += [[f"document_{doc_id}" for doc_id in has_general_auth[auth]] for auth in tt_authorities_2]
    for i, doc_ids in enumerate(tt_authorities_relevant_ids):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                segment_ids.append(seg_id)
        tt_authorities_relevant_ids[i] += segment_ids 
    # get relevant ids for tt_tags_years
    tt_tags_years_relevant_ids = []
    tt_tags_years_pairs = []
    i = 0
    while True:
        tag, year = tt_tags_years[i]
        ids = set(has_tag[tag]).intersection(set(has_year[year]))
        if len(ids) != 0:
            tt_tags_years_relevant_ids.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_tags_years_pairs.append([tag, year])
            if len(tt_tags_years_relevant_ids) == num // 6 or i + 1 == len(tt_tags_years):
                break
        i += 1
    for i, doc_ids in enumerate(tt_tags_years_relevant_ids):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tags_years_pairs[i][0]]: segment_ids.append(seg_id)
        tt_tags_years_relevant_ids[i] += segment_ids 
    # get relevant ids for tt_tags_authorities_1
    tt_tags_authorities_relevant_ids_1 = []
    tt_tags_authorities_pairs_1 = []
    i = 0
    while True:
        tag, auth = tt_tags_authorities_1[i]
        ids = set(has_tag[tag]).intersection(set(has_specific_auth[auth]))
        if len(ids) != 0:
            tt_tags_authorities_relevant_ids_1.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_tags_authorities_pairs_1.append([tag, auth])
            if len(tt_tags_authorities_relevant_ids_1) == num // 12 or i + 1 == len(tt_tags_authorities_1):
                break
        i += 1
    for i, doc_ids in enumerate(tt_tags_authorities_relevant_ids_1):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tags_authorities_pairs_1[i][0]]: segment_ids.append(seg_id)
        tt_tags_authorities_relevant_ids_1[i] += segment_ids 
    # get relevant ids for tt_tags_authorities_2
    tt_tags_authorities_relevant_ids_2 = []
    tt_tags_authorities_pairs_2 = []
    i = 0
    while True:
        tag, auth = tt_tags_authorities_2[i]
        ids = set(has_tag[tag]).intersection(set(has_general_auth[auth]))
        if len(ids) != 0:
            tt_tags_authorities_relevant_ids_2.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_tags_authorities_pairs_2.append([tag, auth])
            if len(tt_tags_authorities_relevant_ids_2) == num // 12 or i + 1 == len(tt_tags_authorities_2):
                break
        i += 1
    for i, doc_ids in enumerate(tt_tags_authorities_relevant_ids_2):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tags_authorities_pairs_2[i][0]]: segment_ids.append(seg_id)
        tt_tags_authorities_relevant_ids_2[i] += segment_ids
    # get relevant ids for tt_authorities_years_1
    tt_authorities_years_relevant_ids_1 = []
    tt_authorities_years_pairs_1 = []
    i = 0
    while True:
        auth, year = tt_authorities_years_1[i]
        ids = set(has_specific_auth[auth]).intersection(set(has_year[year]))
        if len(ids) != 0:
            tt_authorities_years_relevant_ids_1.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_authorities_years_pairs_1.append([auth, year])
            if len(tt_authorities_years_relevant_ids_1) == num // 12 or i + 1 == len(tt_authorities_years_1):
                break
        i += 1
    for i, doc_ids in enumerate(tt_authorities_years_relevant_ids_1):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                segment_ids.append(seg_id)
        tt_authorities_years_relevant_ids_1[i] += segment_ids
    # get relevant ids for tt_authorities_years_2
    tt_authorities_years_relevant_ids_2 = []
    tt_authorities_years_pairs_2 = []
    i = 0
    while True:
        auth, year = tt_authorities_years_2[i]
        ids = set(has_general_auth[auth]).intersection(set(has_year[year]))
        if len(ids) != 0:
            tt_authorities_years_relevant_ids_2.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_authorities_years_pairs_2.append([auth, year])
            if len(tt_authorities_years_relevant_ids_2) == num // 12 or i + 1 == len(tt_authorities_years_2):
                break
        i += 1
    for i, doc_ids in enumerate(tt_authorities_years_relevant_ids_2):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                segment_ids.append(seg_id)
        tt_authorities_years_relevant_ids_2[i] += segment_ids
    # Generate prompts for tt_tay_1
    tt_tay_relevant_ids_1 = []
    tt_tay_pairs_1 = []
    i = 0
    while True:
        tag, auth, year = tt_tay_1[i]
        ids = set(has_specific_auth[auth]).intersection(set(has_year[year])).intersection(set(has_tag[tag]))
        if len(ids) != 0:
            tt_tay_relevant_ids_1.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_tay_pairs_1.append([tag, auth, year])
            if len(tt_tay_relevant_ids_1) == num // 12 or i + 1 == len(tt_tay_1):
                break
        i += 1
    for i, doc_ids in enumerate(tt_tay_relevant_ids_1):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tay_pairs_1[i][0]]: segment_ids.append(seg_id)
        tt_tay_relevant_ids_1[i] += segment_ids
    # Generate prompts for tt_tay_2
    tt_tay_relevant_ids_2 = []
    tt_tay_pairs_2 = []
    i = 0
    while True:
        tag, auth, year = tt_tay_2[i]
        ids = set(has_general_auth[auth]).intersection(set(has_year[year])).intersection(set(has_tag[tag]))
        if len(ids) != 0:
            tt_tay_relevant_ids_2.append([f"document_{doc_id}" for doc_id in list(ids)])
            tt_tay_pairs_2.append([tag, auth, year])
            if len(tt_tay_relevant_ids_2) == num // 12 or i + 1 == len(tt_tay_2):
                break
        i += 1
    for i, doc_ids in enumerate(tt_tay_relevant_ids_2):
        segment_ids = []
        for doc_id in doc_ids:
            agora_id = doc_id[9:]
            segments = segs[segs["Document ID"] == int(agora_id)]["Segment position"].tolist()
            for s in segments:
                seg_id = f"segment_{agora_id}_{s}"
                if seg_id in has_tag_segs[tt_tay_pairs_2[i][0]]: segment_ids.append(seg_id)
        tt_tay_relevant_ids_2[i] += segment_ids

    # actually get the prompts
    tt_t_prompts = [trends_over_tags(tag=tag) for tag in tt_tags]
    tt_a1_prompts = [trends_over_tags(authority=auth) for auth in tt_authorities_1]
    tt_a2_prompts = [trends_over_tags(authority=auth) for auth in tt_authorities_2]
    tt_ty_prompts = [trends_over_tags(tag=tag, year=year) for auth, year in tt_tags_years_pairs]
    tt_ta1_prompts = [trends_over_tags(tag=tag, authority=auth) for tag, auth in tt_tags_authorities_pairs_1]
    tt_ta2_prompts = [trends_over_tags(tag=tag, authority=auth) for tag, auth in tt_tags_authorities_pairs_2]
    tt_ay1_prompts = [trends_over_tags(year=year, authority=auth) for auth, year in tt_authorities_years_pairs_1]
    tt_ay2_prompts = [trends_over_tags(year=year, authority=auth) for auth, year in tt_authorities_years_pairs_2]
    tt_tay1_prompts = [trends_over_tags(tag=tag,year=year, authority=auth) for tag, auth, year in tt_tay_pairs_1]
    tt_tay2_prompts = [trends_over_tags(tag=tag,year=year, authority=auth) for tag, auth, year in tt_tay_pairs_2]
    tt_prompts = tt_t_prompts + tt_a1_prompts + tt_a2_prompts + tt_ty_prompts + tt_ta1_prompts + tt_ta2_prompts + tt_ay1_prompts + tt_ay2_prompts + tt_tay1_prompts + tt_tay2_prompts
    tt_relevant_ids = tt_tags_relevant_ids + tt_authorities_relevant_ids + tt_tags_years_relevant_ids + tt_tags_authorities_relevant_ids_1 + tt_tags_authorities_relevant_ids_2 + tt_authorities_years_relevant_ids_1 + tt_authorities_years_relevant_ids_2 + tt_tay_relevant_ids_1 + tt_tay_relevant_ids_2
    if (len(tt_prompts) != len(tt_relevant_ids)):
        print(len(tt_prompts))
        print(len(tt_relevant_ids))
        exit(1)
    return tt_prompts, tt_relevant_ids
# ======================================================== #
# Generate prompts                                         #
# ======================================================== #

sd_prompts, sd_relevant_ids = generate_single_doc_prompts(SINGLE_DOC)
dc_prompts, dc_relevant_ids = generate_doc_comparison_prompts(DOC_COMP)
ts_prompts, ts_relevant_ids = generate_tag_status_prompts(TAG_STATUS)
ac_prompts, ac_relevant_ids = generate_authority_comparison_prompts(AUTHORITY_COMP)
tt_prompts, tt_relevant_ids = generate_trends_prompts(TAG_TRENDS)
prompts = sd_prompts + dc_prompts + ts_prompts + ac_prompts + tt_prompts
relevant_ids = sd_relevant_ids + dc_relevant_ids + ts_relevant_ids + ac_relevant_ids + tt_relevant_ids
if len(prompts) != len(relevant_ids):
    print("thats not right")
    exit(1)

prompts_df = pd.DataFrame({
    "prompt": prompts,
    "relevant_ids": relevant_ids
})
prompts_df.to_csv("retriever_finetuning/prompts.csv", index=False)
print("prompts saved")
