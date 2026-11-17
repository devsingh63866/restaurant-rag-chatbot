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
