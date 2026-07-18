import os
import sqlite3
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_PATH = "data/responses.db"
MODEL = "llama-3.1-8b-instant"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER,
            prompt TEXT,
            response TEXT,
            temperature REAL,
            model TEXT
        )
    """)
    conn.commit()
    return conn

def get_response(prompt, temperature):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return completion.choices[0].message.content

def main():
    conn = init_db()
    prompts_df = pd.read_csv("data/prompts.csv")

    for _, row in prompts_df.iterrows():
        prompt_id, prompt = row["id"], row["prompt"]
        print(f"Generating for prompt {prompt_id}: {prompt[:50]}...")

        for temp in [0.3, 0.9]:
            response = get_response(prompt, temp)
            conn.execute(
                "INSERT INTO responses (prompt_id, prompt, response, temperature, model) VALUES (?, ?, ?, ?, ?)",
                (prompt_id, prompt, response, temp, MODEL)
            )
            conn.commit()

    print("Done. Responses saved to", DB_PATH)
    conn.close()

if __name__ == "__main__":
    main()
