import streamlit as st
import pdfplumber
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sentence_transformers import SentenceTransformer, util
import json


# Helper function: Extract SLA name
def extract_sla_name(text):
    """Extract SLA name dynamically from the text."""
    sla_keywords = ["Service Level Agreement", "SLA"]
    lines = text.split("\n")
    
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in sla_keywords):
            return line.strip()
    return "Unnamed Service Level Agreement"

# Helper function: Extract text with page numbers
def extract_text_with_pages(pdf_path):
    """Extract text from a PDF file with page numbers."""
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_data.append({"page_number": i + 1, "text": text})
    return text_data

# Helper function: Analyze text for summarization
def analyze_text(text):
    """Analyze text using a summarization model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Helper function: Extract entities
def extract_entities(text):
    """Extract SLA-specific entities."""
    entities = {"parties_involved": [], "metrics": [], "systems": []}
    keywords = ["uptime", "response time", "latency", "recovery time"]
    
    for line in text.split("\n"):
        if "party" in line.lower():
            entities["parties_involved"].append(line.strip())
        if any(keyword in line.lower() for keyword in keywords):
            entities["metrics"].append(line.strip())
        if "system" in line.lower():
            entities["systems"].append(line.strip())
    
    entities["parties_involved"] = ", ".join(entities["parties_involved"])
    entities["metrics"] = ", ".join(entities["metrics"])
    entities["systems"] = ", ".join(entities["systems"])
    return entities

# Semantic search using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, context_data):
    """Perform semantic search using Sentence Transformers."""
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
    """Classify the user's query intent."""
    query_vector = vectorizer.transform([query])
    return classifier.predict(query_vector)[0]

# Streamlit UI
st.title("Enhanced SLA & KPI Analyzer")
st.write("Upload SLA documents to extract, analyze, and interact with key details.")

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
        description = analyze_text(combined_text)
        entities = extract_entities(combined_text)

        sla_details = {
            "sla_name": extract_sla_name(combined_text),
            "parties_involved": entities["parties_involved"],
            "system_concerned": entities["systems"],
            "description": description,
            "associated_metrics": entities["metrics"],
            "page_number": [page['page_number'] for page in text_data if page['text'].strip()]
        }

        # Display extracted SLA details
        st.subheader("Extracted SLA Details")
        st.json(sla_details)

        # Conversational Chat
        st.subheader("Ask about the SLA")
        user_query = st.text_input("Enter your query:")

        if user_query:
            intent = classify_query(user_query)
            response = ""

            if intent == "sla":
                response = f"SLA Name: {sla_details['sla_name']}"
            elif intent == "parties":
                response = f"Parties involved: {sla_details['parties_involved']}"
            elif intent == "metrics":
                response = f"Metrics: {sla_details['associated_metrics']}"
            elif intent == "system":
                response = f"System Concerned: {sla_details['system_concerned']}"
            elif intent == "description":
                response = f"Description: {sla_details['description']}"
            else:
                # Semantic search fallback
                matched_text, page_number = semantic_search(user_query, text_data)
                response = f"Found this information on page {page_number}: {matched_text}"

            st.write(response)

        # Download SLA details
        st.download_button(
            label="Download SLA Details as JSON",
            data=json.dumps(sla_details, indent=4),
            file_name="sla_details.json",
            mime="application/json"
        )
