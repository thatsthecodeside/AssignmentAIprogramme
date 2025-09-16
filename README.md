# AssignmentAIprogramme
import streamlit as st
import pandas as pd 
import pdfplumber
import os
import json
from typing import List, Dict

# Placeholder for Ollama interaction
def query_ollama(prompt: str, chat_history: List[Dict]) -> str:
    """
    Send prompt and chat history to Ollama local SLM and return response.
    Implement this with your local Ollama API or CLI calls.
    """
    # Example: call subprocess or use Ollama python SDK if available
    # For now, return dummy response
    return "This is a placeholder response from the local SLM."

def extract_pdf_data(file_path: str) -> Dict:
    """
    Extract financial data from PDF.
    Returns a dictionary summarizing key financial metrics.
    """
    data = {}
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        # Simple heuristic extraction example:
        # You can improve with regex or table extraction
        # For demo, just return raw text
        data['raw_text'] = text
    return data

def extract_excel_data(file_path: str) -> Dict:
    """
    Extract financial data from Excel.
    Returns a dictionary summarizing key financial metrics.
    """
    data = {}
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    data['sheets'] = sheets
    sheet_data = {}
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        sheet_data[sheet] = df
    data['sheet_data'] = sheet_data
    return data

def main():
    st.title("Financial Document QA System")
    uploaded_file = st.file_uploader("Upload a financial document (PDF or Excel)", type=['pdf', 'xls', 'xlsx'])
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)
        # Save uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info("Processing document...")
        if uploaded_file.type == "application/pdf":
            extracted_data = extract_pdf_data(temp_file_path)
            st.success("PDF data extracted.")
            st.text_area("Extracted Text", extracted_data.get('raw_text', ''), height=300)
        else:
            extracted_data = extract_excel_data(temp_file_path)
            st.success("Excel data extracted.")
            for sheet, df in extracted_data['sheet_data'].items():
                st.subheader(f"Sheet: {sheet}")
                st.dataframe(df)
         # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.subheader("Ask questions about the financial data")
        user_question = st.text_input("Your question:")
        if st.button("Ask") and user_question.strip() != "":
            # Prepare prompt with extracted data summary + chat history + user question
            # For demo, just send user question
            prompt = f"Financial data summary: {json.dumps(extracted_data)[:1000]}...\nUser  question: {user_question}"
            # Append user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            # Query Ollama local SLM
            response = query_ollama(prompt, st.session_state.chat_history)
            # Append model response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            # Display conversation
            for chat in st.session_state.chat_history:
                if chat['role'] == 'user':
                    st.markdown(f"**You:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant:** {chat['content']}")

if __name__ == "__main__":
    main()
