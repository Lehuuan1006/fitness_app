import os
import time
import streamlit as st
import speech_recognition as sr  # Thêm dòng này
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from deep_translator import GoogleTranslator
from gtts import gTTS  # Thêm thư viện này để đọc văn bản
from tempfile import NamedTemporaryFile
import pyttsx3
import re
import threading
# Configure Streamlit page
st.set_page_config(
    page_title="Fitness AI assistant",
    page_icon="💪",
    initial_sidebar_state="collapsed",
)

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-chunks"
index = pc.Index(index_name)

# Initialize the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

embed_model = load_embedding_model()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Initialize YouTube API
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

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
        As your AI fitness coach, I'll use the provided context to answer your question. Here's my approach:
        - Direct and encouraging tone
        - Focus on proper form, technique, and injury prevention
        - Briefly touch on biomechanics and muscle activation
        - Suggest modifications for limitations
        
        Context: {context}
        User Query: {query}

        Response:
    """
    response = llm.invoke(combined_prompt)
    return response.content

def recommend_videos(query, num_recommendations=3):
    search_response = youtube.search().list(
        q=query,
        part='id',
        maxResults=num_recommendations * 2,  # Tăng số lượng kết quả để lọc được đủ shorts
        type='video'
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

    # Lấy thông tin chi tiết video
    video_details = youtube.videos().list(
        part='contentDetails,snippet',
        id=','.join(video_ids)
    ).execute()

    recommendations = []
    for video in video_details.get('items', []):
        duration = video['contentDetails']['duration']
        if 'M' not in duration and 'H' not in duration:  # Chỉ chọn video < 1 phút (không chứa phút hoặc giờ)
            title = video['snippet']['title']
            video_id = video['id']
            thumbnail_url = video['snippet']['thumbnails']['high']['url']

            recommendations.append({
                'title': title,
                'video_id': video_id,
                'thumbnail_url': thumbnail_url
            })

        if len(recommendations) >= num_recommendations:
            break  # Đủ số lượng khuyến nghị

    return recommendations

def get_response_and_recommendations(user_query):
    start_time = time.time()
    search_results = process_query(user_query)
    context = prepare_context(search_results)
    response = generate_response(user_query, context)
    video_recommendations = recommend_videos(user_query)
    end_time = time.time()
    response_time = end_time - start_time
    return response, video_recommendations, response_time

def speak_response_with_gtts(response, lang_code="en"):
    try:
        # Làm sạch văn bản
        cleaned_response = clean_text(response)

        # Chuyển đổi mã ngôn ngữ cho gTTS
        if lang_code == "vi-VN":
            lang_code = "vi"  # gTTS chỉ hỗ trợ "vi" cho tiếng Việt

        # Generate speech từ văn bản đã làm sạch
        tts = gTTS(text=cleaned_response, lang=lang_code)
        
        # Tạo file tạm để lưu âm thanh
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Phát âm thanh bằng Streamlit
        st.audio(temp_audio_path)

        # Tùy chọn xóa file tạm sau khi phát
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as e:
        st.error(f"Error generating speech: {e}")

def clean_text(text):
    # Loại bỏ dấu '*' và khoảng trắng thừa
    cleaned_text = text.replace('*', '')  # Xóa tất cả dấu '*'
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Loại bỏ khoảng trắng thừa
    return cleaned_text


st.markdown("<h2 style='text-align: center;'>Fitness AI Coach</h2>", unsafe_allow_html=True)
st.write("<h6 style='text-align: center;'> Your 24/7 fitness expert. Ask me anything about workouts, nutrition, or injury prevention!</h6>", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chọn ngôn ngữ giao tiếp (Vietnamese hoặc English)
language = st.sidebar.selectbox(
    "Choose your language for voice input:",
    ("English", "Vietnamese"),  # Hiển thị giao diện chọn tiếng Việt và tiếng Anh
    index=0  # Mặc định là English
)

# Xác định mã ngôn ngữ dựa trên lựa chọn của người dùng
language_code = "vi-VN" if language == "Vietnamese" else "en-US"


# Hàm phản hồi ngôn ngữ linh hoạt

def translate_response_to_language(response, lang_code):
    if lang_code == "vi-VN":
        translated_response = GoogleTranslator(source='en', target='vi').translate(response)
        return translated_response
    return response  # Trả lời gốc (tiếng Anh)
# Initial prompts
# Danh sách câu hỏi khởi tạo
initial_prompts = {
    "en-US": [
        "How can I improve my form during squats?",
        "How should I warm up before a workout?",
        "Can you suggest a beginner workout plan for building muscle?",
        "What nutrition tips do you recommend for weight loss?"
    ],
    "vi-VN": [
        "Làm thế nào để cải thiện tư thế squat của tôi?",
        "Tôi nên khởi động thế nào trước khi tập luyện?",
        "Bạn có thể gợi ý kế hoạch tập luyện cơ bản để tăng cơ không?",
        "Bạn khuyến nghị những mẹo dinh dưỡng nào để giảm cân?"
    ]
}

# Chọn danh sách câu hỏi dựa trên ngôn ngữ
selected_prompts = initial_prompts[language_code]


# Display initial prompts
st.write("<p style='text-align: center;'>Chọn một câu hỏi để bắt đầu hoặc nhập câu hỏi của bạn</p>" if language_code == "vi-VN" else "<p style='text-align: center;'>Choose a question to get started or type your own</p>", unsafe_allow_html=True)
cols = st.columns(2)
for i, prompt in enumerate(selected_prompts):
    if cols[i % 2].button(prompt, key=f"prompt_{i}"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response, video_recommendations, response_time = get_response_and_recommendations(prompt)
        localized_response = translate_response_to_language(response, language_code)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": localized_response,
            "recommendations": video_recommendations,
            "response_time": response_time
        })
        st.rerun()

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "💬"):
        st.write(message["content"])
    if message["role"] == "assistant" and "recommendations" in message:
        st.markdown(f"<p style='color: grey; font-size: 0.8em;'>Response time: {message['response_time']:.2f} seconds</p>", unsafe_allow_html=True)
        st.subheader("Recommended Videos:")
        cols = st.columns(3)
        for idx, rec in enumerate(message["recommendations"]):
            with cols[idx]:
                st.image(rec['thumbnail_url'], use_column_width=True)
                st.write(f"**{rec['title']}**")
                video_url = f"https://www.youtube.com/watch?v={rec['video_id']}"
                st.markdown(f"[Watch Video]({video_url})")

    # Thêm nút phát câu trả lời
    if message["role"] == "assistant":
        if st.button("🔊 Play Response", key=f"play_{id(message)}"):
            speak_response_with_gtts(message["content"], language_code)


# Cập nhật thông báo phù hợp với ngôn ngữ
def get_localized_message(key):
    messages = {
        "listening": {
            "vi-VN": "🎤 Đang lắng nghe... Vui lòng nói rõ ràng.",
            "en-US": "🎤 Listening... Please speak clearly.",
        },
        "recognition_success": {
            "vi-VN": "🗣️ Bạn vừa nói: ",
            "en-US": "🗣️ You said: ",
        },
        "unknown_error": {
            "vi-VN": "Xin lỗi, tôi không thể hiểu được âm thanh.",
            "en-US": "Sorry, I couldn't understand the audio.",
        },
        "service_unavailable": {
            "vi-VN": "Dịch vụ nhận diện giọng nói không khả dụng.",
            "en-US": "Speech recognition service is not available.",
        },
        "error_occurred": {
            "vi-VN": "Đã xảy ra lỗi: ",
            "en-US": "An error occurred: ",
        },
    }
    return messages[key][language_code]

# Hàm xử lý giọng nói thành văn bản
def speech_to_text(lang_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(get_localized_message("listening"))
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language=lang_code)
            st.success(f"{get_localized_message('recognition_success')}{text}")
            return text
        except sr.UnknownValueError:
            st.error(get_localized_message("unknown_error"))
        except sr.RequestError:
            st.error(get_localized_message("service_unavailable"))
        except Exception as e:
            st.error(f"{get_localized_message('error_occurred')}{e}")
    return None



# Nút thu âm giọng nói
st.markdown("<h3 style='text-align: center;'>Voice Input</h3>", unsafe_allow_html=True)
if st.button("🎙️ Speak" if language_code == "en-US" else "🎙️ Nói"):
    user_speech = speech_to_text(language_code)
    if user_speech:
        with st.chat_message("user", avatar="🧑"):
            st.write(user_speech)
        with st.chat_message("assistant", avatar="💬"):
            with st.spinner(
                "Processing your query..." if language_code == "en-US" else "Đang xử lý câu hỏi của bạn..."
            ):
                response, video_recommendations, response_time = get_response_and_recommendations(user_speech)
                localized_response = translate_response_to_language(response, language_code)
        st.session_state.chat_history.append({"role": "user", "content": user_speech})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": localized_response,
            "recommendations": video_recommendations,
            "response_time": response_time,
        })
        st.rerun()

# Đầu vào văn bản
user_input = st.chat_input("Type your fitness question here..." if language_code == "en-US" else "Nhập câu hỏi của bạn về thể hình tại đây...")

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)
    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Processing your query..." if language_code == "en-US" else "Đang xử lý câu hỏi của bạn..."):
            response, video_recommendations, response_time = get_response_and_recommendations(user_input)
            localized_response = translate_response_to_language(response, language_code)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": localized_response,
        "recommendations": video_recommendations,
        "response_time": response_time,
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