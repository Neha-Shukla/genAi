import streamlit as st
import pdfplumber
import re
import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'sla_details' not in st.session_state:
    st.session_state.sla_details = {}

# Helper function: Extract SLA name
def extract_sla_name(text):
    sla_keywords = ["Service Level Agreement", "SLA"]
    lines = text.split("\n")
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in sla_keywords):
            return line.strip()
    return "Unnamed Service Level Agreement"

# Helper function: Extract text with page numbers
def extract_text_with_pages(pdf_path):
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_data.append({"page_number": i + 1, "text": text})
    return text_data

# Helper function: Analyze text for summarization
def analyze_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Helper function: Extract uptime dynamically
def extract_uptime_value(text):
    uptime_pattern = r"uptime\s*(\d+\.?\d*)%\s*"  # Regex pattern to capture uptime percentage
    match = re.search(uptime_pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))  # Return uptime percentage as a float
    return None  # If no uptime is found, return None

# Helper function: Extract entities dynamically
def extract_entities(text):
    entities = {"parties_involved": [], "metrics": [], "systems": [], "uptime": None, "response_time": None}
    keywords = ["uptime", "response time", "latency", "recovery time"]
    
    uptime_value = extract_uptime_value(text)
    if uptime_value:
        entities["uptime"] = uptime_value  # Store the dynamically extracted uptime value
    
    for line in text.split("\n"):
        if "party" in line.lower():
            entities["parties_involved"].append(line.strip())
        if any(keyword in line.lower() for keyword in keywords):
            entities["metrics"].append(line.strip())
        if "system" in line.lower():
            entities["systems"].append(line.strip())
        if "response time" in line.lower():
            entities["response_time"] = line.strip()
    
    entities["parties_involved"] = ", ".join(entities["parties_involved"])
    entities["metrics"] = ", ".join(entities["metrics"])
    entities["systems"] = ", ".join(entities["systems"])
    
    return entities

# Query intent classification
vectorizer = CountVectorizer()
classifier = MultinomialNB()

# Training data for intent classification
training_data = [
    ("What is SLA name", "sla"),
    ("Who are the parties involved?", "parties"),
    ("What are the metrics?", "metrics"),
    ("What system does this concern?", "system"),
    ("Describe the SLA.", "description"),
    ("What is the response time?", "metrics"),
]
texts, labels = zip(*training_data)
X_train = vectorizer.fit_transform(texts)
classifier.fit(X_train, labels)

def classify_query(query):
    query = query.lower()

    if "uptime" in query:
        return "uptime"
    elif "parties" in query or "who" in query:
        return "parties"
    elif "metrics" in query or "response time" in query:
        return "metrics"
    elif "system" in query:
        return "system"
    elif "description" in query or "overview" in query:
        return "description"
    else:
        return "other"

# Semantic search using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, context_data):
    query_embedding = model.encode(query)
    context_embeddings = [
        {"text": page['text'], "embedding": model.encode(page['text']), "page_number": page['page_number']}
        for page in context_data
    ]
    
    scores = [
        util.cos_sim(query_embedding, context["embedding"]).item()
        for context in context_embeddings
    ]
    
    best_match = context_embeddings[scores.index(max(scores))]
    return best_match['text'], best_match['page_number']

# Function to answer the query
def answer_query(query, sla_details, text_data):
    intent = classify_query(query)
    response = ""

    if intent == "uptime":
        if sla_details.get("uptime"):
            response = f"Uptime: {sla_details['uptime']}%"
        else:
            response = "Uptime information is not available in the document."
    elif intent == "parties":
        if sla_details.get("parties_involved"):
            response = f"Parties involved: {sla_details['parties_involved']}"
        else:
            response = "Parties involved information is not available in the document."
    elif intent == "metrics":
        if sla_details.get("metrics"):
            response = f"Metrics: {sla_details['metrics']}"
        else:
            response = "Metrics information is not available in the document."
    elif intent == "system":
        if sla_details.get("systems"):
            response = f"System Concerned: {sla_details['systems']}"
        else:
            response = "System concerned information is not available in the document."
    elif intent == "description":
        if sla_details.get("description"):
            response = f"Description: {sla_details['description']}"
        else:
            response = "Description information is not available in the document."
    else:
        matched_text, page_number = semantic_search(query, text_data)
        response = f"Found this information on page {page_number}: {matched_text}"

    return response

# Streamlit UI
st.set_page_config(page_title="SLA & KPI Analyzer", page_icon="📊", layout="wide")

# Add a custom CSS style for better look
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50;
            padding: 10px;
        }
        .stTextInput>div>div>input:focus {
            border-color: #FF5722;
        }
        .stJson {
            background-color: #f1f1f1;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
        .conversation-history {
            margin-top: 20px;
            padding: 10px;
            background-color: #f4f4f4;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .user-query {
            font-weight: bold;
            color: #3b82f6;
        }
        .bot-response {
            font-style: italic;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Add a banner image or title for extra design
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">SLA & KPI Analyzer</h1>
    <p style="text-align: center; color: #555;">Upload SLA documents to extract, analyze, and interact with key details.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Text extraction
    text_data = extract_text_with_pages(pdf_path)
    if not text_data:
        st.error("No text found in the document.")
    else:
        st.success("Text extraction completed!")

        # Summarization and entity extraction
        combined_text = " ".join([page['text'] for page in text_data if page['text'].strip()])
        entities = extract_entities(combined_text)

        st.session_state.sla_details = {
            "sla_name": extract_sla_name(combined_text),
            "parties_involved": entities["parties_involved"],
            "system_concerned": entities["systems"],
            "description": analyze_text(combined_text),
            "associated_metrics": entities["metrics"],
            "uptime": entities["uptime"],
            "response_time": entities["response_time"],
            "page_number": [page['page_number'] for page in text_data if page['text'].strip()]
        }

        # Display extracted SLA details
        st.subheader("Extracted SLA Details")
        st.json(st.session_state.sla_details)

        # Conversational Chat
        st.subheader("Ask about the SLA")
        user_query = st.text_input("Enter your query:")

        if user_query:
            response = answer_query(user_query, st.session_state.sla_details, text_data)

            # Add query and response to conversation history
            st.session_state.conversation_history.append(f"User: {user_query}")
            st.session_state.conversation_history.append(f"Bot: {response}")

            # Display conversation history
            for message in st.session_state.conversation_history:
                if message.startswith("User:"):
                    st.markdown(f"<div class='conversation-history user-query'>{message}</div>", unsafe_allow_html=True)
                elif message.startswith("Bot:"):
                    st.markdown(f"<div class='conversation-history bot-response'>{message}</div>", unsafe_allow_html=True)

        # Download SLA details
        st.download_button(
            label="Download SLA Details as JSON",
            data=json.dumps(st.session_state.sla_details, indent=4),
            file_name="sla_details.json",
            mime="application/json"
        )
