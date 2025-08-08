from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on this context:
            {context}
            
            Question: {question}
        """)

    def generate_answer(self, query: str, context: List[Document]) -> str:
        """Generate answer using retrieved context (preferred method)"""
        context_text = "\n\n".join(doc.page_content for doc in context)
        chain = self.prompt | self.llm
        return chain.invoke({
            "question": query,
            "context": context_text
        }).content

    # Optional: Alias for backward compatibility
    generate_ans = generate_answer
