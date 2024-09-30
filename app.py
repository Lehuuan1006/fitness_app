import os
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Streamlit page
st.set_page_config(
    page_title="Fitness AI Chatbot",
    page_icon="💪",
    initial_sidebar_state="collapsed",
)

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "fitness-chatbot-enhanced"
index = pc.Index(index_name)

# Initialize the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

embed_model = load_embedding_model()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def process_query(query, top_k=6):
    query_embedding = embed_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def prepare_context(results):
    context = ""
    for match in results['matches']:
        context += match['metadata'].get('content', '') + " "
    return context.strip()

def generate_response(query, context):
    combined_prompt = f"""
        As Jeff Cavaliere's AI fitness assistant, use the context to answer the query. Follow these guidelines:

        1. Style: Direct, informative, and encouraging. Use "we" for shared journey.
        2. Content: Focus on AthleanX principles:
            - Proper form and technique
            - Mind-muscle connection
            - Functional training
            - Injury prevention
        3. Explain: Briefly cover biomechanics and muscle activation.
        4. Tailor: Consider user's potential limitations, offer modifications if needed.
        5. Motivate: Include a brief encouragement or AthleanX catchphrase.
        6. Honesty: If unsure, say so. Don't speculate.

        Context: {context}

        User Query: {query}

        Response:
    """
    response = llm.invoke(combined_prompt)
    return response.content

def recommend_chunks(results, num_recommendations=3):
    recommendations = []
    
    for match in results['matches']:
        content = match['metadata'].get('content')
        title = match['metadata'].get('title')
        if content and title:
            recommendations.append({
                'title': title,
                'content': content
            })
        
        if len(recommendations) == num_recommendations:
            break
    
    return recommendations

def get_response_and_recommendations(user_query):
    start_time = time.time()
    search_results = process_query(user_query)
    context = prepare_context(search_results)
    response = generate_response(user_query, context)
    chunk_recommendations = recommend_chunks(search_results)
    end_time = time.time()
    response_time = end_time - start_time
    return response, chunk_recommendations, response_time

st.markdown("<h2 style='text-align: center;'>AthleanX AI Coach</h2>", unsafe_allow_html=True)
st.write("<h6 style='text-align: center;'> Your 24/7 fitness expert. Ask me anything about workouts, nutrition, or injury prevention!</h6>", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initial prompts
initial_prompts = [
    "How important is meal timing when it comes to muscle gain?",
    "I sit at a desk all day. What exercises can help with posture?",
    "How can I prevent wrist pain during push-ups and planks?",
    "What are some lifestyle changes I can make to prevent love handles?"
]

# Display initial prompts
st.write("<p style='text-align: center;'>Choose a question to get started or type your own</p>", unsafe_allow_html=True)
cols = st.columns(2)
for i, prompt in enumerate(initial_prompts):
    if cols[i % 2].button(prompt, key=f"prompt_{i}"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response, chunk_recommendations, response_time = get_response_and_recommendations(prompt)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "recommendations": chunk_recommendations,
            "response_time": response_time
        })
        st.rerun()

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "💬"):
        st.write(message["content"])
    if message["role"] == "assistant" and "recommendations" in message:
        st.markdown(f"<p style='color: grey; font-size: 0.8em;'>Response time: {message['response_time']:.2f} seconds</p>", unsafe_allow_html=True)
        st.subheader("Recommended Chunks:")
        for rec in message["recommendations"]:
            st.write(f"**{rec['title']}**")
            st.write(f"{rec['content'][:100]}...")  # Display first 100 characters of the content
            st.write("---")

user_input = st.chat_input("Type your fitness question here...")

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Crushing this query for you..."):
            response, chunk_recommendations, response_time = get_response_and_recommendations(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "recommendations": chunk_recommendations,
        "response_time": response_time
    })

    st.rerun()

# Sidebar
st.sidebar.title("About Fitness AI Coach")
st.sidebar.markdown("""### What's Unique?
✅ Direct access to Jeff Nippard's fitness philosophy  
✅ Personalized advice from Jeff Nippard's training program  
✅ AI responses with human-like understanding  
""")
st.sidebar.markdown("""### What can Fitness AI Coach do?
📊 Physical Therapist & Strength Coach  
🏆 Trained professional athletes  
🧠 Known for science-based fitness approach  
""")

st.sidebar.markdown("""---
### About the Creator
This AI coach was developed by *Le Huu An*  
*Exploring LLMs to solve real-world pain points in fitness.*  
[LinkedIn](https://www.linkedin.com/in/lehuuan/) | 
[GitHub](https://github.com/Lehuuan1006)
""")
