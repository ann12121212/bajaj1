from loading_utils import loading # Your custom loader
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb  # For direct ChromaDB client access
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Takes care of all API keys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
class DocumentP:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # Clear old collection if exists
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            client.delete_collection("insurance_docs")
        except:
            pass
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        # Configure synchronous embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            transport="rest",  # Force REST instead of gRPC
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.client = client
        self.vectorstore = None
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document through the pipeline"""
        raw_docs = self._load_all_docs(file_path)
        chunks = self._chunk_documents(raw_docs)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            client=self.client,
            collection_name="insurance_docs",
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        return chunks

    def _load_all_docs(self, file_path: str) -> List[Document]:
        """Load documents and flatten the list"""
        docs = loading(file_path)  # Returns List[Document]
        # Ensure each doc has basic metadata
        for doc in docs:
            doc.metadata.setdefault("source", os.path.basename(file_path))
        return docs

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents with metadata preservation"""
        chunks = []
        for doc in docs:
            # Create a unique chunk ID
            doc.metadata["chunk_id"] = f"{doc.metadata['source']}_{len(chunks)}"
            # Split and extend
            chunks.extend(self.text_splitter.split_documents([doc]))
        return chunks

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Semantic search with ChromaDB"""
        if self.vectorstore is None:
            raise ValueError("Process documents first using process_document()")
        return self.vectorstore.similarity_search(query, k=k)

# Usage example
if __name__ == "__main__":
    processor = DocumentP()
    
    try:
        chunks = processor.process_document("https://arxiv.org/pdf/1706.03762.pdf")
        results = processor.retrieve( "What is a Transformer and what are its components?")
        
        for doc in results:
            print(f"📄 From {doc.metadata.get('source', 'unknown')}:")
            print(f"ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"Content:\n{doc.page_content}\n{'―'*40}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    try:
        # Example usage
        test_file = "example.pdf"  # Change to your test file
        if os.path.exists(test_file):
            chunks = processor.process_document(test_file)
            query = "What is the main topic of this document?"
            results = processor.retrieve(query, k=3)
            
            print(f"Results for: '{query}'")
            for doc in results:
                print(f"\n📄 Source: {doc.metadata.get('source', 'unknown')}")
                print(f"🔒 Security: {doc.metadata.get('security', 'unknown')}")
                print(f"📝 Content:\n{doc.page_content[:500]}...")
                
                # Show related documents
                related = processor.get_related_documents(doc.metadata["chunk_id"])
                if related:
                    print(f"\n🔗 Related documents: {len(related)}")
        else:
            print(f"Test file {test_file} not found")
                
    except Exception as e:
        print(f"Error: {str(e)}")
