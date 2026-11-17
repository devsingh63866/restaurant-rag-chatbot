import os, re, json
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

MODEL = "openai/gpt-oss-120b"   # same model that worked in Colab

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Restaurant Chatbot", page_icon="🍽️")
st.title("🍽️ Restaurant Recommendation Chatbot")
st.caption("Answers come only from real Zomato reviews and menus (Bangalore).")

@st.cache_resource(show_spinner="Loading restaurant data (first start takes ~1-2 min)...")
def load():
    restaurants = json.load(open("restaurants.json", encoding="utf-8"))
    docs = []
    for r in restaurants:
        meta = {"name": r["name"], "cuisine": r["cuisine"], "area": r["area"], "price": r["price"]}
        if r["menu"] != "Menu not listed":
            docs.append(Document(
                page_content=f"{r['name']} ({r['cuisine']}, {r['area']}, {r['price']}) MENU: {r['menu']}",
                metadata={**meta, "type": "menu"}))
        for rev in r["reviews"]:
            docs.append(Document(
                page_content=f"{r['name']} ({r['cuisine']}, {r['area']}, {r['price']}) REVIEW: {rev}",
                metadata={**meta, "type": "review"}))
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, emb)
    areas = sorted({r["area"] for r in restaurants})
    llm = ChatGroq(model=MODEL, temperature=0.2)
    return db, areas, llm

db, areas, llm = load()

# ---------- Chat UI ----------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Bangalore restaurant recommendation assistant. Answer ONLY using the "
     "context below (real Zomato reviews and menus). If the context does not contain "
     "the answer, say you don't have enough information. Recommend specific restaurants "
     "by name, mention area, cuisine and price, and briefly say why based on the reviews. "
     "Do not invent dishes, prices or facts.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def detect_area(question):
    q = question.lower()
    for a in areas:
        if a.lower() in q:
            return a
    return None

def answer(question):
    area = detect_area(question)
    kwargs = {"k": 8}
    if area:
        kwargs["filter"] = {"area": area}
    hits = db.similarity_search(question, **kwargs)
    if not hits and area:           # fall back to unfiltered search
        hits = db.similarity_search(question, k=8)
    context = "\n\n".join(h.page_content for h in hits)
    return (PROMPT | llm).invoke({"context": context, "question": question}).content

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask e.g. 'best biryani in HSR' or 'veg places in Jayanagar'"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching reviews..."):
            try:
                reply = answer(question)
            except Exception as e:
                reply = f"Sorry, something went wrong: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
