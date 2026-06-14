import sqlite3

DB_NAME = "travel_app.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            destination TEXT,
            budget TEXT,
            summary TEXT,
            schedule_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY,
            nickname TEXT,
            age TEXT,
            mbti TEXT,
            favorite_travel TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_plan(title, destination, budget, summary, schedule_text):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO saved_plans
        (title, destination, budget, summary, schedule_text)
        VALUES (?, ?, ?, ?, ?)
    """, (title, destination, budget, summary, schedule_text))

    conn.commit()
    conn.close()

def get_saved_plans():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM saved_plans
        ORDER BY created_at DESC
    """)

    plans = cur.fetchall()
    conn.close()

    return plans

def delete_plan(plan_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM saved_plans
        WHERE id = ?
    """, (plan_id,))

    conn.commit()
    conn.close()

def get_plan(plan_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM saved_plans
        WHERE id = ?
    """, (plan_id,))

    plan = cur.fetchone()
    conn.close()

    return plan

def save_profile(nickname, age, mbti, favorite_travel, image_path):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("DELETE FROM profile")

    cur.execute("""
        INSERT INTO profile
        (id, nickname, age, mbti, favorite_travel, image_path)
        VALUES (1, ?, ?, ?, ?, ?)
    """, (nickname, age, mbti, favorite_travel, image_path))

    conn.commit()
    conn.close()

def get_profile():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM profile WHERE id = 1")
    profile = cur.fetchone()

    conn.close()
    return profile