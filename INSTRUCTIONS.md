# INSTRUCTIONS.md
# AI Workflow Guide — Political Memory: Event Detection and Saliency Decay in the Congressional Record

This file is intended to help an LLM (Claude, GPT, Gemini, etc.) understand the structure, purpose, and execution of this project so it can assist with building, running, and extending it.

---

## Project Overview

This project analyzes 150+ years of US Congressional speech to detect the linguistic footprint of major external events and measure how long that footprint persists (saliency decay). The pipeline is fully unsupervised and NLP-driven.

**Core research questions:**
- When a major event occurs (e.g. 9/11, 2008 financial crisis), how does it register in congressional speech?
- How quickly does that linguistic signal decay over subsequent weeks?

**Key methods:** BERTopic, LDA, change-point detection (ruptures), TF-IDF, sentence embeddings.

---

## Repository Structure

```
congressional-records-events-saliency/
├── code/
│   ├── preprocessing.ipynb     # Data loading, cleaning, aggregation pipeline
│   └── analysis.ipynb          # Topic modeling, TF-IDF analysis, feature extraction
├── data/
│   └── hein-daily/             # Raw Gentzkow dataset (NOT included, must be downloaded)
│       ├── speeches_097.txt    # Raw speeches per congress session (097–114)
│       ├── descr_097.txt       # Speaker metadata per session
│       └── 097_SpeakerMap.txt  # Speaker ID to name/party mapping
├── my_data/
│   ├── df109.csv               # Cleaned speech dataframe for congress 109
│   └── test_weekly_109_110.csv # Weekly aggregated speeches for sessions 109–110
├── plots/                      # Output visualizations
└── README.md
```

---

## Data

The raw data is from the **Gentzkow, Shapiro & Taddy Congressional Speech Dataset**:
- Download: https://data.stanford.edu/congress_text
- Total size: ~31GB, not included in this repo
- Place downloaded files under `data/hein-daily/`

Each congress session has three files:
- `speeches_NNN.txt` — pipe-delimited, columns: `speech_id | speech`
- `descr_NNN.txt` — pipe-delimited, columns: `speech_id | chamber | date | ...`
- `NNN_SpeakerMap.txt` — maps speaker IDs to names and parties

The processed output files in `my_data/` are the cleaned, merged results ready for analysis.

---

## Environment Setup

**Python version:** 3.10

**Key dependencies:**
```
pandas
numpy
nltk
spacy
sentence-transformers
bertopic
scikit-learn
ruptures
matplotlib
seaborn
wordcloud
torch (CPU version recommended if no GPU)
```

**Install:**
```bash
pip install pandas numpy nltk spacy sentence-transformers bertopic scikit-learn ruptures matplotlib seaborn wordcloud
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_sm
```

---

## Running the Project

### Step 1: Preprocessing (`code/preprocessing.ipynb`)

Loads raw Gentzkow data, cleans it, and produces the weekly aggregated corpus.

**Cleaning pipeline in order:**
1. Load and merge `speeches`, `descr`, and `SpeakerMap` files
2. `removeProcedural(df)` — removes speeches by procedural speakers (The Speaker, The Clerk, etc.)
3. `filterWordCount(df, minWords=50)` — removes speeches under 50 words
4. `parseDate(df)` — parses date column to datetime
5. `filterCommonWordRatio(df, threshold=0.05)` — removes speeches where fewer than 5% of words are common English function words (catches OCR garbage and name-lists)
6. `stripSalutations(df)` — removes "Mr. Speaker.", "Madam President." etc. anywhere in speech text
7. `removeProceduralSentences(df, threshold=0.70)` — uses `all-MiniLM-L6-v2` sentence embeddings to drop sentences semantically similar to known procedural seed phrases

**Output:** `my_data/df109.csv` and `my_data/test_weekly_109_110.csv`

### Step 2: Analysis (`code/analysis.ipynb`)

Runs topic modeling and preliminary event detection on the cleaned weekly corpus.

**What it does:**
- TF-IDF vectorization over weekly windows
- Word trend plots and heatmaps
- Preliminary change-point detection using `ruptures`
- Wordcloud visualizations per congress session

---

## Key Design Decisions (for LLM context)

- **Weekly aggregation:** Speeches are grouped into weekly windows to create a time series. This is the unit of analysis for event detection.
- **Session scope:** Current cleaned data covers sessions 109–110 (approx. 2005–2008). Full corpus spans sessions 097–114.
- **Sentence-level cleaning:** Rather than dropping entire speeches, procedural sentences are removed individually to preserve substantive content while reducing token noise for BERTopic.
- **Threshold for procedural sentence removal:** 0.70 cosine similarity against 19 seed procedural sentences. Validated via spot-check (procedural sentences score 0.74+, substantive sentences score ~0.23–0.36).
- **Salutation pattern:** Regex removes titles like "Mr. President", "Madam Speaker", "Mr. Chairman" anywhere in text, not just at sentence start.

---

## Next Steps (Checkpoint 3)

- Fit BERTopic over full weekly corpus
- Extract topic distribution time series
- Apply `ruptures` change-point detection to topic signals
- Validate detected change-points against a manually curated ground truth list of ~20 major events

---

## Notes for LLM Assistants

- The column containing speech text is called `speech` in the raw data and `speech` or `speech_clean` in processed dataframes depending on the cleaning stage.
- All cleaning functions are designed to be chained inside `cleanCongress(df)`.
- The dataset uses periods instead of commas in many places due to OCR artifacts in older records.
- If asked to extend the cleaning pipeline, preserve the word count stats logging pattern used in existing functions.
- Do not use `df.copy()` unnecessarily — memory is constrained.
- For embedding tasks, always load the SentenceTransformer model once outside any loop or function.
