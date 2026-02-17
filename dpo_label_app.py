import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


BASE_DIR = Path(__file__).parent / "data" / "agora"
SPLIT_DIR = BASE_DIR / "split_dpo_answers"
FULLTEXT_DIR = BASE_DIR / "fulltext"
LABELS_DIR = SPLIT_DIR / "labels"


# ---------- Helpers ----------
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_split_file(index: int) -> Tuple[Path, List[Dict[str, Any]]]:
    file_path = SPLIT_DIR / f"dpo_answer_{index}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    data = read_json(file_path)
    if not isinstance(data, list):
        raise ValueError("Split file must contain a JSON array of objects.")
    return file_path, data


def get_doc_id(item: Dict[str, Any]) -> Optional[int]:
    doc_id = "document_id"
    if doc_id in item:
        try:
            return int(item[doc_id])
        except Exception:
            return None
    return None


def get_policy_name(item: Dict[str, Any]) -> Optional[str]:
    """Return the policy_name for an item if present."""
    if "policy_name" in item and isinstance(item["policy_name"], str):
        return item["policy_name"]
    return None


def get_question(item: Dict[str, Any]) -> str:
    q = "question"
    if q in item and isinstance(item[q], str):
        return item[q]
    return json.dumps(item, ensure_ascii=False)[:200]


def get_answers(item: Dict[str, Any]) -> Tuple[str, str]:
    a1, a2 = "answer_low_temp", "answer_high_temp"
    if a1 in item and a2 in item:
        return str(item[a1]), str(item[a2])
    return ("<missing answer 1>", "<missing answer 2>")


def load_document_context(doc_id: Optional[int]) -> str:
    if doc_id is None:
        return "(No document_id provided.)"
    doc_path = FULLTEXT_DIR / f"{doc_id}.txt"
    if not doc_path.exists():
        return f"(Context file not found: {doc_path.name})"
    try:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text
    except Exception as e:
        return f"(Error reading context: {e})"


def labels_path_for_index(index: int) -> Path:
    return LABELS_DIR / f"dpo_answer_{index}_labels.json"


def load_existing_labels(index: int) -> Dict[int, int]:
    """Return a mapping item_idx -> preferred_option (1 or 2)."""
    path = labels_path_for_index(index)
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        mapping = {}
        for row in data:
            item_idx = int(row.get("item_index"))
            pref = int(row.get("preferred"))
            if pref in (1, 2):
                mapping[item_idx] = pref
        return mapping
    except Exception:
        return {}


def save_labels(index: int, items: List[Dict[str, Any]], selections: Dict[int, int]) -> None:
    rows = []
    for i, item in enumerate(items):
        a1, a2 = get_answers(item)
        q = get_question(item)
        d = get_doc_id(item)
        pname = get_policy_name(item)
        global_idx = (index - 1) * 20 + i + 1
        rows.append(
            {
                "file_index": index,
                "item_index": i,
                "global_index": global_idx,
                "document_id": d,
                "policy_name": pname,
                "question": q,
                "answer_1": a1,
                "answer_2": a2,
                "preferred": selections.get(i),  # may be None if not selected
            }
        )
    write_json(labels_path_for_index(index), rows)


# ---------- UI ----------
st.set_page_config(page_title="DPO Preference Labeling", layout="wide")
st.title("DPO Preference Labeling")

with st.sidebar:
    st.header("Controls")
    st.number_input(
        "Split file index",
        min_value=1,
        max_value=200,
        value=st.session_state.get("file_index", 1),
        step=1,
        key="file_index",
    )

# Detect index change to reset transient state
prev_index = st.session_state.get("prev_file_index")
loaded_index = int(st.session_state.get("file_index", 1))
if prev_index != loaded_index:
    st.session_state["prev_file_index"] = loaded_index
    st.session_state.pop("selections", None)
    st.session_state.pop("items", None)
    st.session_state.pop("split_file_path", None)

try:
    split_file_path, items = load_split_file(loaded_index)
    st.session_state["items"] = items
    st.session_state["split_file_path"] = str(split_file_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.caption(f"Loaded: {st.session_state['split_file_path']}")

# Initialize / load selections
selections: Dict[int, int] = st.session_state.get("selections") or load_existing_labels(loaded_index)
st.session_state["selections"] = selections


def render_item(i: int, item: Dict[str, Any]):
    q = get_question(item)
    a1, a2 = get_answers(item)
    d = get_doc_id(item)
    pname = get_policy_name(item)
    context_text = load_document_context(d)
    header = f"Q{i+1}: {q}"
    if pname:
        header = f"{header}  \n\n*Document:* {pname}"

    st.markdown(f"### {header}")
    expander_label = f"Document Context (document_id={d}"
    if pname:
        expander_label += f", policy_name={pname}"
    expander_label += ")"

    with st.expander(expander_label, expanded=False):
        st.text_area(
            label=f"context_{i}",
            value=context_text,
            height=200,
            key=f"context_area_{loaded_index}_{i}",
        )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Answer 1**")
        st.write(a1)
    with cols[1]:
        st.markdown("**Answer 2**")
        st.write(a2)

    current = selections.get(i)
    choice = st.radio(
        "Preferred answer",
        options=[1, 2],
        format_func=lambda x: f"Answer {x}",
        index=0 if current == 1 else (1 if current == 2 else 0),
        key=f"choice_{loaded_index}_{i}",
        horizontal=True,
    )
    selections[i] = int(choice)
    st.divider()


for i, item in enumerate(st.session_state["items"]):
    render_item(i, item)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Save Preferences", type="primary"):
        save_labels(loaded_index, st.session_state["items"], selections)
        st.success(f"Saved to {labels_path_for_index(loaded_index)}")
with col2:
    if st.button("Reload Labels"):
        st.session_state["selections"] = load_existing_labels(loaded_index)
        st.rerun()

st.caption("Tip: Use the sidebar to switch between split files. Your choices are saved per file.")
