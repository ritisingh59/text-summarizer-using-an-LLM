import streamlit as st
import torch
import base64
import os
import tempfile

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader



checkpoint = "./LaMini-Flan-T5-248M"

offload_dir = "./offload"
os.makedirs(offload_dir, exist_ok=True)

st.set_page_config(layout="wide", page_title="PDF Summarizer")



@st.cache_resource
def load_model():

    tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        offload_folder=offload_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return tokenizer, model


tokenizer, base_model = load_model()



def file_preprocessing(file_path):

    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.split_documents(pages)

        return [doc.page_content for doc in docs]

    except Exception as e:
        st.error(f"PDF Reading Error: {e}")
        return []



def llm_pipeline(file_path):

    summarizer = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=120,
        min_length=30,
        do_sample=False
    )

    chunks = file_preprocessing(file_path)

    chunk_summaries = []


    for chunk in chunks:

        if len(chunk.strip()) < 50:
            continue

        try:
            result = summarizer(chunk)
            summary_text = result[0]["summary_text"].strip()
            chunk_summaries.append(summary_text)

        except Exception:
            continue

    if not chunk_summaries:
        return ""

    
    combined_summary = " ".join(chunk_summaries)

    
    try:
        final_summary = summarizer(
            combined_summary,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        return final_summary[0]["summary_text"]

    except Exception:
        return combined_summary



def displayPDF(file_path):

    with open(file_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{pdf_base64}"
        width="100%" height="600"
        type="application/pdf"></iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)



def main():

    st.title("AI-Powered PDF Research Summarizer(Local LLM)")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.button("Summarize"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Summary")

            with st.spinner("Summarizing..."):

                summary = llm_pipeline(temp_path)

                if summary:
                    st.write(summary)
                else:
                    st.warning(" No summary generated.")

        with col2:

            st.subheader(" Uploaded PDF")
            displayPDF(temp_path)

        os.remove(temp_path)



if __name__ == "__main__":
    main()