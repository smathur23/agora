import pandas as pd
import re
import random

documents = pd.read_csv("data/agora/documents.csv")
segments = pd.read_csv("data/agora/segments.csv")
tag_relations = pd.read_csv("retriever_finetuning/tag_relations.csv")

def clean_tag(tag):
    if ":" in tag:
        return tag.split(":", 1)[1].strip()
    return tag.strip()
documents["Tags"] = documents["Tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(";") if t.strip()]
)
segments["Tags"] = segments["Tags"].fillna("").apply(
    lambda x: [clean_tag(t) for t in x.split(";") if t.strip()]
)
documents["year"] = pd.to_datetime(documents["Most recent activity date"], errors="coerce").dt.year
all_years = sorted(documents["year"].dropna().unique().tolist())
segments = segments.merge(
    documents[['AGORA ID', 'year', 'Authority']], 
    left_on='Document ID',            
    right_on='AGORA ID',             
    how='left'                         
)
segments = segments.drop(columns=['AGORA ID'])
segments["year"] = segments["year"].astype("Int64").astype(str)

tag_relations["tag"] = tag_relations["tag"].apply(lambda x: clean_tag(x))
all_tags = sorted({tag for tags in documents["Tags"] for tag in tags})
all_specific_authorities = sorted({auth for auth in documents["Authority"]})
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
state_codes = {"Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT","Delaware": "DE","Florida": "FL","Georgia": "GA","Hawaii": "HI","Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","New Hampshire": "NH","New Jersey": "NJ","New Mexico": "NM","New York": "NY","North Carolina": "NC","North Dakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","Rhode Island": "RI","South Carolina": "SC","South Dakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA","West Virginia": "WV","Wisconsin": "WI","Wyoming": "WY","District of Columbia": "DC"}

def get_true_negatives(prompt_type, relevant_ids, tag=None, auth1=None, year=None, auth2=None, n=10):
    bad_tag = None
    bad_auths = None
    bad_years = None
    if prompt_type == "trends":
        if tag is not None:
            bad_tag = str(tag_relations[tag_relations["tag"] == tag]["furthest"])
        if auth1 is not None:
            bad_auths = get_furthest_auths(auth1)
        if year is not None:
            bad_years = all_years
            for i in range(11):
                if int(year) - 5 + i in bad_years:
                    bad_years.remove(int(year) - 5 + i)
    elif prompt_type == "tag_status":
        bad_tag = str(tag_relations[tag_relations["tag"] == tag]["furthest"])
        bad_auths = get_furthest_auths(auth1)
    elif prompt_type == "auth_comp":
        if tag is not None:
            bad_tag = str(tag_relations[tag_relations["tag"] == tag]["furthest"])
        bad_auths = get_furthest_auths(auth1, auth2)
    elif prompt_type == "doc_comp" or prompt_type == "single_doc":
        pass
    else:
        print("bad prompt type")
        return
    filter_df = segments.copy()
    filter_df = filter_df[filter_df.apply(lambda row: f"segment_{row['Document ID']}_{row['Segment position']}" not in relevant_ids, axis=1)]
    if bad_tag is not None:
        temp = filter_df[(filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and bad_tag in tags)) & (filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag not in tags))]
        if len(temp) >= n:
            filter_df = temp
        else:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag not in tags)]
    if bad_auths is not None:
        temp = filter_df[filter_df["Authority"].isin(bad_auths)]
        if len(temp) >= n:
            filter_df = temp
        else:
            filter_df = filter_df[(filter_df["Authority"] != auth1) & (filter_df["Authority"] != auth2)]
    if bad_years is not None:
        temp = filter_df[filter_df["year"].isin(bad_years)]
        if len(temp) >= n:
            filter_df = temp
        else:
            filter_df = filter_df[filter_df["year"] != year]
    negative_examples = filter_df.sample(n=min(len(filter_df), n))
    if len(negative_examples) != n:
        print("didn't find enough negative examples.")
        print(f"only found {len(negative_examples)}, looked for {n}")
        print(f"type: {prompt_type}, tag: {tag}, auth1: {auth1}, auth2: {auth2}, year: {year}")
    doc_id = negative_examples["Document ID"].tolist()
    seg_pos = negative_examples["Segment position"].tolist()
    return [f"segment_{doc_id[i]}_{seg_pos[i]}" for i in range(len(negative_examples))]

def get_furthest_auths(auth1, auth2=None):
    both = []
    for auth in [auth1, auth2]: 
        if auth is None:
            return both[0]
        bad_auths = []
        if auth in general_authorities["United States"] or auth == "United States" or auth == "US local governments" or auth == "US state governments" or auth == "US federal government":
            bad_auths = general_authorities["China"] + general_authorities["Foreign countries"]
        if auth in general_authorities["China"] or auth in general_authorities["Foreign countries"] or auth == "China" or auth == "Foreign countries": 
            bad_auths = general_authorities["United States"]
        if auth in general_authorities["Multinational"] or auth == "Multinational":
            bad_auths = general_authorities["US local governments"]
        both.append(bad_auths)
    common = list(set(both[0]) & set(both[1]))
    if len(common) > 0:
        return common
    if not (auth1 in general_authorities["United States"] or auth1 == "United States" or auth1 == "US local governments" or auth1 == "US state governments" or auth1 == "US federal government"):
        temp = auth1
        auth1 = auth2
        auth2 = temp 
    bad_auths = []
    if auth2 != "China" and auth2 not in general_authorities["China"]:
        bad_auths += general_authorities["China"]
    if auth2 == "China" or auth2 in general_authorities["China"]:
        bad_auths += general_authorities["Foreign countries"][2:]
    if auth1 != "US local governments":
        bad_auths += general_authorities["US local governments"]
        if auth1 in bad_auths: bad_auths.remove(auth1)
    else:
        bad_auths += general_authorities["US federal government"]
    if auth1 in general_authorities["US state governments"] and auth1 != "State governments":
        bad_auths = [a for a in bad_auths if state_codes[auth1] not in a]


def get_useful_negatives(prompt_type, relevant_ids, tag=None, auth1=None, year=None, auth2=None, n=10):
    bad_auths = None
    filter_df = segments.copy()
    filter_df = filter_df[filter_df.apply(lambda row: f"segment_{row['Document ID']}_{row['Segment position']}" not in relevant_ids, axis=1)]
    if prompt_type == "trends":
        if auth1 is not None:
            if auth1 in all_specific_authorities:
                bad_auths = all_specific_authorities.copy()
                bad_auths.remove(auth1)
            else:
                bad_auths = []
                for k in general_authorities:
                    if k != auth1:
                        bad_auths += general_authorities[k]
        if year is None and auth1 is None:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag not in tags)]
        elif year is None and tag is None:
            filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
        elif year is None and tag is not None and auth1 is not None:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag in tags)]
            filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
        elif year is not None and auth1 is None:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag not in tags)]
            temp =  filter_df[filter_df["year"] == year]
            if len(temp) > 0:
                filter_df = temp
        elif year is not None and tag is None:
            filter_df = filter_df[filter_df["year"] == year]
            filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
        elif year is not None and tag is not None and auth1 is not None:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag in tags)]
            filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
            temp = filter_df[filter_df["year"] == year]
            if len(temp) > 0:
                filter_df = temp
        else:
            print("This shouldn't happen")
            return
    elif prompt_type == "tag_status":
        if auth1 in all_specific_authorities:
            bad_auths = all_specific_authorities.copy()
            bad_auths.remove(auth1)
        else:
            bad_auths = []
            for k in general_authorities:
                if k != auth1:
                    bad_auths += general_authorities[k]
        filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag in tags)]
        filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
    elif prompt_type == "auth_comp":
        if auth1 in all_specific_authorities and auth2 in all_specific_authorities:
            bad_auths = all_specific_authorities.copy()
            bad_auths.remove(auth1)
            bad_auths.remove(auth2)
        elif auth1 in all_specific_authorities and auth2 in general_authorities:
            bad_auths = []
            for k in general_authorities:
                if k != auth2:
                    bad_auths += general_authorities[k]
            if auth1 in bad_auths:
                bad_auths.remove(auth1)
        elif auth2 in all_specific_authorities and auth1 in general_authorities:
            bad_auths = []
            for k in general_authorities:
                if k != auth1:
                    bad_auths += general_authorities[k]
            if auth2 in bad_auths:
                bad_auths.remove(auth2)
        elif auth1 in general_authorities and auth2 in general_authorities:
            bad_auths = []
            for k in general_authorities:
                if k != auth1 and k != auth2:
                    bad_auths += general_authorities[k]
        else: 
            print("This should not happen")
            return
        filter_df = filter_df[filter_df["Authority"].isin(bad_auths)]
        if tag is not None:
            filter_df = filter_df[filter_df["Tags"].apply(lambda tags: isinstance(tags, list) and tag in tags)]
    elif prompt_type == "doc_comp" or prompt_type == "single_doc":
        pass
    else:
        print("bad prompt type")
        return
    ids = [f"segment_{row['Document ID']}_{row['Segment position']}" for idx, row in filter_df.iterrows()]
    sample = random.sample(ids, min(n,len(ids)))
    if len(sample) < n:
        print(f"only {len(sample)} ids found, looked for {n}")
        print(f"type: {prompt_type}, tag: {tag}, auth1: {auth1}, auth2: {auth2}, year: {year}")
        if len(sample) == 0:
            row = segments.sample(1)
            return [f"segment_{row['Document ID']}_{row['Segment position']}"]
    return sample

def get_random_negatives(relevant_ids, n=10):
    candidates = []
    for idx, row in segments.iterrows():
        agora_id = row["Document ID"]
        seg_pos = row["Segment position"]
        if f"segment_{agora_id}_{seg_pos}" not in relevant_ids:
            candidates.append(f"segment_{agora_id}_{seg_pos}")
    return random.sample(candidates, 10)

def find_tag(text):
    tag = ""
    for t in all_tags:
        if t in text:
            tag = t
    if tag == "": 
        print("tag not identified in")
        print(text)
        return
    return tag

def find_authority(text, n=1):
    auths = []
    for a in all_specific_authorities:
        if a in text:
            auths.append(a)
    for a in general_authorities:
        if a in text:
            auths.append(a)
    if len(auths) != n:
        if len(auths) > 1 and auths[0] in auths[1]:
            auths.pop(0)
        if len(auths) > 1 and auths[1] in auths[0]:
            auths.pop(1)
        if len(auths) > 2 and auths[1] in auths[2]:
            auths.pop(1)
        if len(auths) > 2 and auths[2] in auths[1]:
            auths.pop(2)
        
        if len(auths) != n:
            print(f"{len(auths)} authorities identified, {n} should have been found in")
            print(text)
            return
    return auths if n > 1 else auths[0]

def find_year(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return match.group()
    else:
        print("No year identified in ")
        print(text)
        return

def main():
    questions_df = pd.read_csv("retriever_finetuning/questions_processed.csv")
    all_true_negatives = []
    all_useful_negatives = []
    all_random_negatives = []
    all_combo_negatives = []
    for idx, row in questions_df.iterrows():
        prompt = row["prompt"]
        question = row["question"]
        relevant_ids = row["relevant_ids"].split(";")
        tag = None
        year = None
        auth1 = None
        auth2 = None
        q_type = ""
        if "Ask one single question about AI policy trends " in prompt:
            q_type = "trends"
            if "tag: " in prompt:
                tag = find_tag(prompt)
                if tag is None: return
            if "authority: " in prompt:
                auth1 = find_authority(prompt)
                if auth1 is None: return
            if "since" in prompt:
                year = find_year(prompt)
                if year is None: return        
        elif "Ask a question about the status of AI policy relating to tag: " in prompt:
            q_type = "tag_status"
            tag = find_tag(prompt)
            if tag is None: return
            auth1 = find_authority(prompt)
            if auth1 is None: return            
        elif "Ask a question about the differences in AI policy " in prompt:
            q_type = "auth_comp"
            auths = find_authority(prompt, n=2)
            if auths is None: return
            auth1 = auths[0]
            auth2 = auths[1]
            if "relating to tag: " in prompt:
                tag = find_tag(prompt)
                if tag is None: return
        elif "(start of document)" in prompt:
            q_type = "single_doc"
        elif "(start of document 1)" in prompt:
            q_type = "doc_comp"
        else:
            print("unknown prompt")
            return

        true_negatives = get_true_negatives(q_type, relevant_ids, tag=tag, auth1=auth1, year=year, auth2=auth2)
        useful_negatives = get_useful_negatives(q_type, relevant_ids, tag=tag, auth1=auth1, year=year, auth2=auth2)
        random_negatives = get_random_negatives(relevant_ids)
        combo_negatives = random.sample(true_negatives, min(5, len(true_negatives))) + random.sample(useful_negatives, min(5, len(useful_negatives)))
        
        for i in true_negatives:
            if i in relevant_ids:
                print("true")
                print(true_negatives)
                print(relevant_ids)
                return
        for i in random_negatives:
            if i in relevant_ids:
                print("random")
                print(random_negatives)
                print(relevant_ids)
                return

        all_true_negatives.append(";".join(true_negatives))
        all_useful_negatives.append(";".join(useful_negatives))
        all_random_negatives.append(";".join(random_negatives))
        all_combo_negatives.append(";".join(combo_negatives))

        if idx % 20 == 0:
            print(idx)
            out = pd.DataFrame({
                "question": questions_df["question"][:idx + 1],
                "relevant_ids": questions_df["relevant_ids"][:idx + 1],
                "true_negatives": all_true_negatives,
                "useful_negatives": all_useful_negatives,
                "combo_negatives": all_combo_negatives,
                "random_negatives": all_random_negatives
            })
            out.to_csv("retriever_finetuning/questions_with_negatives.csv", index=False)


if __name__ == "__main__":
    main()