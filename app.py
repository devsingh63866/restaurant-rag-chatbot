
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
