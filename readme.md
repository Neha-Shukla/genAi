python3 -m venv venv
source ./venv/bin/activate -- ubuntu
pip install streamlit transformers pdfplumber torch torchvision sentence_transformers
streamlit run app.py