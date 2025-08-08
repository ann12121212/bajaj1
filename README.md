# Document QA with Gemini <img src="https://em-content.zobj.net/source/google/387/rocket_1f680.png" width="30">

A production-grade RAG system powered by Google's Gemini for advanced document question answering. Supports PDF, DOCX, and EML files with semantic search.


## ✨ Features
- **Multi-Format Support**: PDFs, Word docs, and emails
- **Production Architecture**: Robust processing pipeline
- **Hybrid Retrieval**: Semantic + keyword search
- **Gemini 1.5 Flash**: Cutting-edge LLM technology
- **Streamlit UI**: Intuitive web interface

## 🛠 Tech Stack
| Component           | Technology               |
|---------------------|--------------------------|
| LLM Framework       | LangChain                |
| Foundation Model    | Gemini 1.5 Flash         |
| Vector DB           | ChromaDB                 |
| Embeddings          | Google Embedding-001     |
| Document Processors | PDFPlumber, Unstructured |

## ⚡ Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourusername/document-qa-gemini.git
cd document-qa-gemini

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add .env file
echo "GOOGLE_API_KEY=your_key_here" > .env

# 5. Run!
streamlit run app.py
