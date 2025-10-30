# app.py
"""
Streamlit Daily Nutrition and Mood Analytics Dashboard
- Single-file Streamlit app
- No paid APIs required (uses a mock nutrition DB and a simple mood analyzer)
- Stores entries locally in 'entries.json'
- Interactive charts using plotly
- Inline comments explain each section
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
import re
import os
import plotly.express as px
from collections import Counter, defaultdict

# Optional: try TextBlob for sentiment if available; otherwise we'll use a keyword-based fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# ---------------------------
# Configuration / constants
# ---------------------------
DATA_FILE = "entries.json"  # local JSON storage
st.set_page_config(page_title="Daily Nutrition & Mood Analytics", layout="wide", initial_sidebar_state="expanded")

# Mock nutrition DB: per-serving approximate macros (calories, protein(g), carbs(g), fat(g), fiber(g), sugar(g))
# Values are illustrative approximations for demo purposes
MOCK_NUTR_DB = {
    "egg": {"cal": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sugar": 0.0},
    "eggs": {"cal": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sugar": 0.0},
    "slice of bread": {"cal": 80, "protein": 3.0, "carbs": 15.0, "fat": 1.0, "fiber": 1.0, "sugar": 1.5},
    "bread": {"cal": 75, "protein": 3.0, "carbs": 14.0, "fat": 1.0, "fiber": 1.0, "sugar": 1.5},
    "milk (1 cup)": {"cal": 122, "protein": 8.0, "carbs": 12.0, "fat": 5.0, "fiber": 0.0, "sugar": 12.0},
    "glass of milk": {"cal": 122, "protein": 8.0, "carbs": 12.0, "fat": 5.0, "fiber": 0.0, "sugar": 12.0},
    "banana": {"cal": 105, "protein": 1.3, "carbs": 27.0, "fat": 0.4, "fiber": 3.1, "sugar": 14.4},
    "apple": {"cal": 95, "protein": 0.5, "carbs": 25.1, "fat": 0.3, "fiber": 4.4, "sugar": 18.9},
    "oatmeal (1 cup cooked)": {"cal": 166, "protein": 6.0, "carbs": 28.0, "fat": 4.0, "fiber": 4.0, "sugar": 1.0},
    "rice (1 cup cooked)": {"cal": 206, "protein": 4.3, "carbs": 45.0, "fat": 0.4, "fiber": 0.6, "sugar": 0.1},
    "chicken breast (100g)": {"cal": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sugar": 0.0},
    "salad (1 cup)": {"cal": 33, "protein": 1.6, "carbs": 6.0, "fat": 0.5, "fiber": 2.2, "sugar": 2.9},
    "yogurt (1 cup)": {"cal": 154, "protein": 12.9, "carbs": 17.4, "fat": 3.8, "fiber": 0.0, "sugar": 17.4},
    "coffee": {"cal": 2, "protein": 0.3, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sugar": 0.0},
    "tea": {"cal": 2, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sugar": 0.0},
    "butter (1 tbsp)": {"cal": 102, "protein": 0.1, "carbs": 0.0, "fat": 11.5, "fiber": 0.0, "sugar": 0.0},
    "cheese (1 slice ~28g)": {"cal": 113, "protein": 7.0, "carbs": 0.9, "fat": 9.3, "fiber": 0.0, "sugar": 0.2},
    # add more items as needed for better parsing
}

# List of known tokens to improve parsing: sort by length to catch multi-word items first
KNOWN_FOODS_SORTED = sorted(MOCK_NUTR_DB.keys(), key=lambda s: -len(s))


# ---------------------------
# Utility functions
# ---------------------------

def load_entries():
    """Load stored entries from DATA_FILE; if missing, create sample data."""
    if not os.path.exists(DATA_FILE):
        # create example mock data
        entries = [
            {
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().date().isoformat(),
                "food_text": "2 eggs, a slice of bread, and a glass of milk",
                "parsed_items": [
                    {"item": "eggs", "quantity": 2, "nutrition": multiply_nutrition(MOCK_NUTR_DB.get("eggs"))},
                    {"item": "slice of bread", "quantity": 1, "nutrition": multiply_nutrition(MOCK_NUTR_DB.get("slice of bread"))},
                    {"item": "glass of milk", "quantity": 1, "nutrition": multiply_nutrition(MOCK_NUTR_DB.get("glass of milk"))},
                ],
                "totals": sum_nutrition_list([
                    multiply_nutrition(MOCK_NUTR_DB.get("eggs"), 2),
                    multiply_nutrition(MOCK_NUTR_DB.get("slice of bread"),1),
                    multiply_nutrition(MOCK_NUTR_DB.get("glass of milk"),1),
                ]),
                "mood_text": "Feeling full and relaxed",
                "mood_label": "calm"
            }
        ]
        save_entries(entries)
        return entries
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []


def save_entries(entries):
    """Save entries list to JSON file."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def multiply_nutrition(nutr, qty=1):
    """Multiply single-serving nutrition by qty. nutr is a dict or None."""
    if not nutr:
        return {"cal": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0}
    return {k: round(float(v) * qty, 2) for k, v in nutr.items()}


def sum_nutrition_list(nlist):
    """Sum nutrition dictionaries from a list into totals."""
    totals = {"cal": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0}
    for n in nlist:
        for k in totals:
            totals[k] += n.get(k, 0)
    return {k: round(v, 2) for k, v in totals.items()}


def parse_food_text(food_text):
    """
    Very simple parser: tries to detect known food tokens and quantities.
    - Handles patterns like '2 eggs', 'a slice of bread', 'one apple', 'banana'
    - For unknown items, assumes 1 serving with zero nutrition (can be extended)
    """
    text = food_text.lower()
    # replace 'and' with commas for easier splitting
    text = text.replace(" and ", ", ")
    segments = [s.strip() for s in re.split("[,;]+", text) if s.strip()]
    parsed = []

    for seg in segments:
        # find quantity (number or words like one/two)
        qty = 1
        # numbers (e.g., '2 eggs' or '2.5 cups rice')
        num_match = re.match(r"^(\d+(\.\d+)?)\s+(.*)$", seg)
        if num_match:
            qty = float(num_match.group(1))
            rest = num_match.group(3)
        else:
            # words for numbers
            word_nums = {"one": 1, "two": 2, "three": 3, "four":4, "a":1, "an":1}
            first_word = seg.split()[0]
            if first_word in word_nums:
                qty = word_nums[first_word]
                rest = " ".join(seg.split()[1:])
            else:
                rest = seg

        # try to match known foods (longest first)
        matched = False
        for food_token in KNOWN_FOODS_SORTED:
            if food_token in rest:
                matched = True
                nutr = MOCK_NUTR_DB.get(food_token)
                parsed.append({
                    "item": food_token,
                    "quantity": qty,
                    "nutrition": multiply_nutrition(nutr, qty)
                })
                break

        if not matched:
            # fallback: assume the whole segment is an unknown item
            parsed.append({
                "item": rest,
                "quantity": qty,
                "nutrition": multiply_nutrition(None, 0)  # unknown nutrition, zeros
            })

    totals = sum_nutrition_list([p["nutrition"] for p in parsed])
    return parsed, totals


# ---------------------------
# Mood detection
# ---------------------------

MOOD_KEYWORDS = {
    "happy": ["happy", "joy", "glad", "delighted", "ecstatic", "cheerful", "pleased"],
    "sad": ["sad", "down", "depressed", "unhappy", "miserable", "tearful"],
    "tired": ["tired", "sleepy", "exhausted", "fatigued"],
    "angry": ["angry", "mad", "annoyed", "irritated", "furious"],
    "anxious": ["anxious", "nervous", "worried", "stressed"],
    "calm": ["calm", "relaxed", "peaceful", "chill"],
    "energetic": ["energetic", "energetic", "energetic", "lively", "energetic", "active"],
    "neutral": []
}

def detect_mood(mood_text):
    """Detect mood label. Prefer TextBlob polarity if available, otherwise keyword mapping."""
    if not mood_text or not mood_text.strip():
        return "neutral"
    text = mood_text.lower()

    # First try TextBlob if available (simple polarity -> mood)
    if TEXTBLOB_AVAILABLE:
        try:
            tb = TextBlob(text)
            polarity = tb.sentiment.polarity  # -1..1
            # map polarity ranges to rough moods
            if polarity >= 0.5:
                return "happy"
            elif 0.1 <= polarity < 0.5:
                return "calm"
            elif -0.1 <= polarity < 0.1:
                # check for energy words
                if any(w in text for w in ["tired", "exhausted", "sleep"]):
                    return "tired"
                return "neutral"
            elif polarity < -0.1:
                # negative polarity -> sad/angry/anxious check
                if any(w in text for w in MOOD_KEYWORDS["angry"]):
                    return "angry"
                if any(w in text for w in MOOD_KEYWORDS["anxious"]):
                    return "anxious"
                return "sad"
        except Exception:
            pass

    # Fallback: keyword counting
    scores = defaultdict(int)
    for mood, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[mood] += 1

    if scores:
        # pick mood with highest count >0
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] > 0:
            return best[0]

    # As a last resort, check for words indicating energy or sleepiness
    if any(w in text for w in ["sleep", "tired", "exhausted"]):
        return "tired"
    if any(w in text for w in ["happy", "great", "good", "fantastic"]):
        return "happy"
    if any(w in text for w in ["sad", "bad", "down"]):
        return "sad"

    return "neutral"


# ---------------------------
# App UI helpers
# ---------------------------

def pretty_nutrition_row(n):
    """Return string summarizing nutrition totals."""
    return f"Calories: {n['cal']} kcal — Protein: {n['protein']} g — Carbs: {n['carbs']} g — Fat: {n['fat']} g — Fiber: {n['fiber']} g — Sugar: {n['sugar']} g"


def build_dataframe_from_entries(entries):
    """Convert entries list to pandas DataFrame with totals flattened for analytics."""
    rows = []
    for e in entries:
        row = {
            "timestamp": e.get("timestamp"),
            "date": e.get("date"),
            "food_text": e.get("food_text"),
            "mood_text": e.get("mood_text"),
            "mood_label": e.get("mood_label"),
            "calories": e.get("totals", {}).get("cal", 0),
            "protein": e.get("totals", {}).get("protein", 0),
            "carbs": e.get("totals", {}).get("carbs", 0),
            "fat": e.get("totals", {}).get("fat", 0),
            "fiber": e.get("totals", {}).get("fiber", 0),
            "sugar": e.get("totals", {}).get("sugar", 0),
        }
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    else:
        return pd.DataFrame(columns=["timestamp", "date", "food_text", "mood_text", "mood_label", "calories", "protein", "carbs", "fat", "fiber", "sugar"])


# ---------------------------
# App layout
# ---------------------------

st.title("🍽️ Daily Nutrition & Mood Analytics")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Add Entry", "Analytics Dashboard", "History", "Settings"], index=0)

# small info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Built with a local nutrition DB (mock) • No API key needed")
if TEXTBLOB_AVAILABLE:
    st.sidebar.success("TextBlob available: improved sentiment detection")
else:
    st.sidebar.info("Using keyword-based mood detection (no external downloads)")

# Load entries at start
entries = load_entries()
df_entries = build_dataframe_from_entries(entries)

# ---------------------------
# Page: Add Entry
# ---------------------------
if page == "Add Entry":
    st.header("Add a new food + mood entry")
    st.markdown("Enter what you ate and (optionally) how you feel. The app will estimate nutrition and detect mood.")

    with st.form("entry_form"):
        food_input = st.text_area("What did you eat? (example: '2 eggs, a slice of bread, and a glass of milk')", value="")
        mood_input = st.text_input("How do you feel? (optional)", value="")
        submit_btn = st.form_submit_button("Add entry")

    if submit_btn:
        # parse food
        parsed_items, totals = parse_food_text(food_input)
        # detect mood
        mood_label = detect_mood(mood_input)
        # create entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().date().isoformat(),
            "food_text": food_input,
            "parsed_items": parsed_items,
            "totals": totals,
            "mood_text": mood_input,
            "mood_label": mood_label
        }
        entries.insert(0, entry)  # newest first
        save_entries(entries)
        st.success("Entry saved ✅")
        st.subheader("Nutrition summary")
        st.write(pretty_nutrition_row(totals))
        with st.expander("See parsed items"):
            for p in parsed_items:
                st.write(f"{p['quantity']} × {p['item']} — {pretty_nutrition_row(p['nutrition'])}")

        st.write(f"Mood detected: **{mood_label}**")
        # update dataframe
        df_entries = build_dataframe_from_entries(entries)

# ---------------------------
# Page: Analytics Dashboard
# ---------------------------
elif page == "Analytics Dashboard":
    st.header("Analytics Dashboard")
    if df_entries.empty:
        st.info("No entries yet — add an entry first from the 'Add Entry' tab.")
    else:
        # Date range filter
        min_date = df_entries["date"].min()
        max_date = df_entries["date"].max()
        st.sidebar.markdown("### Filters")
        date_range = st.sidebar.date_input("Select date range", value=(min_date, max_date))
        # ensure two dates
        if isinstance(date_range, tuple) or isinstance(date_range, list):
            start_d, end_d = date_range
        else:
            start_d = date_range
            end_d = date_range

        # filter df
        mask = (df_entries["date"] >= start_d) & (df_entries["date"] <= end_d)
        dff = df_entries.loc[mask].copy()

        # Key summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        total_cal_today = dff[dff["date"] == date.today()]["calories"].sum()
        col1.metric("Total calories (selected range)", f"{dff['calories'].sum():.0f} kcal")
        col2.metric("Avg protein per entry", f"{dff['protein'].mean():.1f} g" if not dff.empty else "0 g")
        col3.metric("Avg carbs per entry", f"{dff['carbs'].mean():.1f} g" if not dff.empty else "0 g")
        col4.metric("Avg fat per entry", f"{dff['fat'].mean():.1f} g" if not dff.empty else "0 g")

        # Daily calorie breakdown (bar chart)
        st.subheader("Daily Calories")
        daily = dff.groupby("date").agg({"calories": "sum"}).reset_index()
        if daily.empty:
            st.info("No daily data in the selected range.")
        else:
            fig_cal = px.bar(daily, x="date", y="calories", title="Calories per Day", labels={"calories":"kcal", "date":"Date"})
            st.plotly_chart(fig_cal, use_container_width=True)

        # Macro distribution for selected period (pie chart)
        st.subheader("Macro-nutrient Distribution (selected range)")
        macros_total = {
            "Protein": dff["protein"].sum(),
            "Carbs": dff["carbs"].sum(),
            "Fat": dff["fat"].sum()
        }
        # avoid empty pie
        if sum(macros_total.values()) <= 0:
            st.info("No macro data to display. Add entries with recognizable foods.")
        else:
            macro_df = pd.DataFrame({
                "macro": list(macros_total.keys()),
                "grams": list(macros_total.values())
            })
            fig_pie = px.pie(macro_df, names="macro", values="grams", title="Macro Distribution (g)")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Mood trend over time (line chart / area)
        st.subheader("Mood Trend")
        mood_counts = dff.groupby(["date", "mood_label"]).size().reset_index(name="count")
        if mood_counts.empty:
            st.info("No mood entries yet.")
        else:
            mood_pivot = mood_counts.pivot(index="date", columns="mood_label", values="count").fillna(0)
            mood_pivot = mood_pivot.sort_index()
            fig_mood = px.line(mood_pivot, x=mood_pivot.index, y=mood_pivot.columns, labels={"value":"count", "date":"Date"}, title="Daily Mood Counts")
            st.plotly_chart(fig_mood, use_container_width=True)

        # Most frequent mood in the last 7 days
        st.subheader("Top-level Insights")
        last_7_mask = df_entries["date"] >= (date.today() - pd.Timedelta(days=7))
        last7 = df_entries.loc[last_7_mask]
        if not last7.empty:
            most_freq_mood = last7["mood_label"].mode().iloc[0]
            st.write(f"Most frequent mood in the last 7 days: **{most_freq_mood}**")
        else:
            st.write("No entries in the last 7 days.")

        # Show a small table of totals per day
        with st.expander("Daily totals (table)"):
            daily_table = dff.groupby("date").agg({"calories":"sum", "protein":"sum", "carbs":"sum", "fat":"sum"}).reset_index()
            st.dataframe(daily_table.style.format({"calories":"{:.0f}", "protein":"{:.1f}", "carbs":"{:.1f}", "fat":"{:.1f}"}), use_container_width=True)


# ---------------------------
# Page: History
# ---------------------------
elif page == "History":
    st.header("History of Entries")
    if df_entries.empty:
        st.info("No saved entries yet.")
    else:
        st.markdown("You can search, inspect parsed nutrition per entry, or delete entries.")
        # search box
        q = st.text_input("Search food or mood text (substring search)", value="")
        df_filtered = df_entries.copy()
        if q:
            df_filtered = df_filtered[df_filtered["food_text"].str.contains(q, case=False) | df_filtered["mood_text"].str.contains(q, case=False)]

        st.dataframe(df_filtered.sort_values("timestamp", ascending=False), use_container_width=True)

        # inspect individual entry
        with st.expander("Inspect an entry"):
            idx = st.number_input("Select entry index (0 = newest)", min_value=0, max_value=max(0, len(entries)-1), value=0, step=1)
            if len(entries) > 0:
                selected = entries[idx]
                st.write(f"**Time:** {selected.get('timestamp')}")
                st.write(f"**Food:** {selected.get('food_text')}")
                st.write(f"**Mood text:** {selected.get('mood_text')}")
                st.write(f"**Mood label:** {selected.get('mood_label')}")
                st.write("**Parsed items & nutrition**")
                for p in selected.get("parsed_items", []):
                    st.write(f"- {p['quantity']} × {p['item']} — {pretty_nutrition_row(p['nutrition'])}")
                st.write("**Totals**")
                st.write(pretty_nutrition_row(selected.get("totals", {})))

        # delete functionality
        with st.expander("Manage data"):
            st.warning("Delete operations are permanent.")
            if st.button("Delete all entries"):
                if st.confirm("Are you sure you want to delete all entries? This cannot be undone."):
                    save_entries([])
                    entries = []
                    df_entries = build_dataframe_from_entries(entries)
                    st.success("All entries deleted.")
            # delete single by index
            del_idx = st.number_input("Delete entry index (0 = newest)", min_value=0, max_value=max(0, len(entries)-1) if entries else 0, value=0, step=1)
            if st.button("Delete selected entry"):
                if entries:
                    removed = entries.pop(int(del_idx))
                    save_entries(entries)
                    df_entries = build_dataframe_from_entries(entries)
                    st.success(f"Deleted entry from {removed.get('timestamp')}")
                else:
                    st.info("No entries to delete.")


# ---------------------------
# Page: Settings
# ---------------------------
elif page == "Settings":
    st.header("Settings & Info")
    st.markdown("""
    - Data file: `entries.json` (stored locally in the same folder as this app).  
    - Nutrition values are mock/approximate — meant for demonstrations. Replace `MOCK_NUTR_DB` with a better dataset or an API if you want accuracy.
    - Mood detection uses TextBlob when available, with a keyword fallback; no external keys required.
    """)
    st.subheader("Example mock food tokens recognized by the app")
    st.write(", ".join(list(MOCK_NUTR_DB.keys())[:30]))
    st.markdown("---")
    st.write("If you'd like, you can:")
    st.write("- Expand the `MOCK_NUTR_DB` dictionary with more foods and proper per-serving values.")
    st.write("- Replace the `parse_food_text` function with a more sophisticated parser or a small model for ingredient extraction.")
    st.write("- Hook a free nutrition API (USDA, Edamam) if you want real-world accuracy (requires API key).")


# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Tip: Add entries multiple times per day to track meals and mood changes. This app is a demo — tune nutrition DB and mood model for production use.")
