
# 🍽️ Restaurant Recommendation Chatbot (RAG)

A chatbot that recommends restaurants using **real Zomato reviews and menus**. It uses Retrieval-Augmented Generation (RAG): it first finds the most relevant reviews/menus, then asks an LLM to answer **only** from that retrieved text, so answers stay grounded in real data.

**Live app:** https://restaurant-rag-chatbot-wg8swsgakoxd5adaegxn9y.streamlit.app/

> **Current coverage:** Bangalore only. Support for more Indian cities is planned (see [Roadmap](#roadmap)).

---

## Features

- Chat interface built with Streamlit, with conversation history
- Answers grounded in Zomato reviews and menus (no invented restaurants or dishes)
- Automatic area detection: ask about "HSR" or "Jayanagar" and the search is limited to that area
- Recommendations include restaurant name, area, cuisine, price and the reason, based on reviews
- Fast, free LLM inference through Groq

## How it works

```
restaurants.json
      │
      ▼
 Build documents (one per menu, one per review)
      │
      ▼
 Split into chunks ──► Embed (all-MiniLM-L6-v2) ──► Chroma vector store
                                                          │
 User question ──► detect area ──► similarity search (top 8 chunks)
                                                          │
                                                          ▼
                              Prompt + retrieved context ──► Groq LLM ──► Answer
```

1. **Load data:** `restaurants.json` is read and turned into documents. Each menu and each review becomes its own document tagged with name, cuisine, area and price.
2. **Index:** documents are split into chunks (500 characters, 50 overlap), embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in an in-memory Chroma database.
3. **Retrieve:** the question is matched against the index. If it mentions a known area, results are filtered to that area.
4. **Generate:** the top 8 chunks are passed to the LLM, which is instructed to answer only from that context.

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | Chroma |
| LLM | Groq (`openai/gpt-oss-120b`) |

## Project structure

```
.
├── app.py              # Streamlit app (data loading, retrieval, chat UI)
├── restaurants.json    # Restaurant data (name, cuisine, area, price, menu, reviews)
├── requirements.txt    # Python dependencies
└── README.md
```

## Data format

Each entry in `restaurants.json` looks like this:

```json
{
  "name": "Restaurant Name",
  "cuisine": "North Indian, Chinese",
  "area": "HSR",
  "price": "₹600 for two",
  "menu": "Popular dishes: ...",
  "reviews": ["(Rated 4.0) Great food ...", "(Rated 2.0) Service was slow ..."]
}
```

If a restaurant has no menu, set `"menu": "Menu not listed"`. It is skipped during indexing.

## Run locally

**Requirements:** Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd restaurant-rag-chatbot

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# 5. Start the app
streamlit run app.py
```

The first start takes 1 to 2 minutes while the embedding model downloads and the index is built.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repo (`app.py`, `restaurants.json`, `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from the repo, with `app.py` as the main file.
3. Open **Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. Any later commit to the repo redeploys the app automatically.

> Never commit your API key. Keep it in Streamlit Secrets or in a local `.streamlit/secrets.toml` (add that file to `.gitignore`).

## Example questions

- "Best biryani in HSR"
- "Veg restaurants in Jayanagar"
- "Good ice cream places in Basavanagudi"
- "Cheap South Indian breakfast near Electronic City"
- "Where can I get good Kerala food?"

## Limitations

- Covers Bangalore only, and only the restaurants in the dataset.
- Review data is a snapshot. Prices, menus and opening status may be out of date.
- Answers can only be as good as the reviews retrieved. If the data has nothing relevant, the bot says so.
- Area detection is simple text matching on area names.
- The index is rebuilt in memory on each cold start.

## Roadmap

- [ ] Expand to restaurants across India (add a `city` field and city detection)
- [ ] Persist the vector index so startup is fast with larger data
- [ ] Filters for cuisine, budget and veg/non-veg
- [ ] Show source reviews alongside each answer

## Data and disclaimer

Review and menu text comes from Zomato listings and belongs to its respective owners. This project is for learning and demonstration only and is not affiliated with Zomato.
