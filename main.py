import streamlit as st
from text_split import DocumentP
from RAG_sys import RAGSystem
import os

# CRITICAL CONFIG - MUST BE FIRST
os.environ["GOOGLE_API_USE_GRPC"] = "false"
os.environ["GOOGLE_API_USE_REST"] = "true"
os.environ["GRPC_DNS_RESOLVER"] = "native"

@st.cache_resource
def init_processor():
    return DocumentP()

@st.cache_resource
def init_rag():
    return RAGSystem()

def main():
    st.title("📄 Document QA with Gemini")
    uploaded_file = st.file_uploader("Upload PDF/DOCX/EML", type=["pdf", "docx", "eml"])
    query = st.text_input("Ask about the document:")

    if uploaded_file and query:
        temp_path = f"./temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            processor = init_processor()
            processor.process_document(temp_path)
            
            rag = init_rag()
            relevant_chunks = processor.retrieve(query, k=3)
            answer = rag.generate_answer(query, relevant_chunks)
            
            st.subheader("Answer:")
            st.markdown(answer)
            
            with st.expander("See relevant chunks"):
                for chunk in relevant_chunks:
                    st.write(f"**Source:** {chunk.metadata['source']}")
                    st.write(chunk.page_content)
                    st.divider()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
