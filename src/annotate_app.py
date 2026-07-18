import sqlite3
import streamlit as st
from datetime import datetime

DB_PATH = "data/responses.db"

def init_labels_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER,
            prompt TEXT,
            chosen_response TEXT,
            rejected_response TEXT,
            annotator TEXT,
            rationale TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()

def get_prompt_pairs(conn):
    cursor = conn.execute("SELECT DISTINCT prompt_id, prompt FROM responses ORDER BY prompt_id")
    return cursor.fetchall()

def get_responses_for_prompt(conn, prompt_id):
    cursor = conn.execute(
        "SELECT id, response, temperature FROM responses WHERE prompt_id = ?",
        (prompt_id,)
    )
    return cursor.fetchall()

def save_label(conn, prompt_id, prompt, chosen, rejected, annotator, rationale):
    conn.execute(
        "INSERT INTO labels (prompt_id, prompt, chosen_response, rejected_response, annotator, rationale, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (prompt_id, prompt, chosen, rejected, annotator, rationale, datetime.now().isoformat())
    )
    conn.commit()

def main():
    st.set_page_config(page_title="LLM Response Annotator", layout="wide")
    st.title("LLM Response Evaluation Tool")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_labels_table(conn)

    annotator = st.sidebar.text_input("Your name (annotator ID)", value="annotator_1")

    prompt_pairs = get_prompt_pairs(conn)
    prompt_options = {f"Prompt {pid}: {ptext[:60]}...": pid for pid, ptext in prompt_pairs}

    selected = st.selectbox("Select a prompt to annotate", list(prompt_options.keys()))
    prompt_id = prompt_options[selected]

    responses = get_responses_for_prompt(conn, prompt_id)
    prompt_text = [p for pid, p in prompt_pairs if pid == prompt_id][0]

    st.markdown(f"### Prompt:\n{prompt_text}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Response A** (temp={responses[0][2]})")
        st.write(responses[0][1])
    with col2:
        st.markdown(f"**Response B** (temp={responses[1][2]})")
        st.write(responses[1][1])

    st.divider()
    choice = st.radio("Which response is better?", ["Response A", "Response B", "Tie / Both equally good"])
    rationale = st.text_area("Why? (rationale)", placeholder="e.g. more accurate, clearer, safer...")

    if st.button("Submit Label"):
        if choice == "Response A":
            chosen, rejected = responses[0][1], responses[1][1]
        elif choice == "Response B":
            chosen, rejected = responses[1][1], responses[0][1]
        else:
            chosen, rejected = "TIE", "TIE"

        save_label(conn, prompt_id, prompt_text, chosen, rejected, annotator, rationale)
        st.success("Label saved!")

    st.divider()
    st.subheader("Your Progress")
    count = conn.execute("SELECT COUNT(*) FROM labels WHERE annotator = ?", (annotator,)).fetchone()[0]
    st.metric("Labels submitted", count)

if __name__ == "__main__":
    main()
