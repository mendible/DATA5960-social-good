# Paste your full Streamlit app code here
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pinecone
import os



#secrets

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["INDEX_NAME"]


# init
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# reag query-er
def create_rag_output(question):
  q_vec2 = model.encode(question, convert_to_numpy=True).tolist()
  res3 = index.query(vector=q_vec2, top_k=30, include_metadata=True, include_values=False)
  return res3["matches"]
#Question function
def do_alex_single_question(question: str):
    rag_context = create_rag_output(question)
    system_prompt = """
    You will receive a question. Please use all the JSON objects you are handed. 
  You will answer questions using the information you are handed. If you do not have sufficient information, give the links with the highest score
  Give your response in the form of an answer plus the relevant date and URL of the relevant article from the RAG output. Have the url as a seperate new line by itself
  IF your RAG score: is low (like lower than 0.1), then say that you are not sure about the question but still give the links with highest scores.
    """
    chat_model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=system_prompt)
    chat = chat_model.start_chat()
    response = chat.send_message(question)
    return response.text.strip()

# ui
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
st.title(" Police Report Database")

query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("###Response")
        st.write(response)

        st.markdown("### Retrieved Context")
        rag_data = create_rag_output(query)
        for item in rag_data["results"]:
            st.markdown(f"**{item['case']} — {item['date']}**")
            st.markdown(f"Tags: `{', '.join(item['tags'])}`")
            st.markdown(f"Excerpt: {item['excerpt']}...")
            st.markdown(f"[{item['link_text']}]({item['link_url']})")
            st.markdown("---")