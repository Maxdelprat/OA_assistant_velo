import os
from dotenv import load_dotenv
import glob
import time
from typing import List, Tuple, Optional

# LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# --- Configuration et Initialisation ---

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

PDF_FOLDER = "./document_pdfs"
PDF_FILES_PATHS = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# --- Pydantic Models for Structured Output ---

class ChunkRelevance(BaseModel):
    """Represents the relevance judgment for a single chunk."""
    chunk_index: int = Field(description="The index of the chunk in the provided list.")
    is_relevant: bool = Field(description="True if chunk contains complete and sufficient information for the question.")
    relevance_score: float = Field(description="Relevance score from 0.0 (not relevant) to 1.0 (highly relevant).")
    reasoning: str = Field(description="Brief explanation of the relevance judgment.")

class BatchRerankResult(BaseModel):
    """Contains relevance judgments for all chunks in a batch."""
    evaluations: List[ChunkRelevance] = Field(description="List of relevance evaluations for each chunk.")

# --- Utility Functions ---

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single text string with metadata."""
    formatted_content = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        formatted_content.append(f"Source: {source} (Page {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_content)

# --- LLM Configuration ---

output_parser = StrOutputParser()

llm_main = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=1000  # Increased for more detailed technical responses
)

llm_judge = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0.1,
    max_tokens=800  # Increased for batch processing
)

# --- Batch Reranking Agent (Optimized) ---

def rerank_chunks_batch(question: str, retrieved_docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    Uses an LLM to evaluate relevance of all chunks in a single batch call,
    then returns the top_k most relevant documents.
    """
    if not retrieved_docs:
        return []
    
    start_time = time.time()
    print(f"   -> Starting batch reranking on {len(retrieved_docs)} chunks...")
    
    # Prepare chunks for batch evaluation
    chunks_text = ""
    for i, doc in enumerate(retrieved_docs):
        chunks_text += f"\n--- CHUNK {i} ---\n{doc.page_content}\n"
    
    # Batch reranking prompt
    rerank_prompt = f"""You are a relevance judge. Evaluate each chunk below for its relevance to the user's question.

Question: {question}

Chunks to evaluate:
{chunks_text}

For each chunk (0 to {len(retrieved_docs)-1}), provide:
- chunk_index: the chunk number
- is_relevant: true if it helps answer the question
- relevance_score: 0.0 to 1.0
- reasoning: brief explanation

Evaluate ALL chunks."""

    try:
        # Single API call for all chunks
        rerank_chain = (
            ChatPromptTemplate.from_messages([("user", "{prompt}")])
            | llm_judge.with_structured_output(BatchRerankResult, method="function_calling")
        )
        
        result = rerank_chain.invoke({"prompt": rerank_prompt})
        
        # Match evaluations with original documents
        scored_docs = []
        for evaluation in result.evaluations:
            if 0 <= evaluation.chunk_index < len(retrieved_docs):
                scored_docs.append({
                    "score": evaluation.relevance_score,
                    "document": retrieved_docs[evaluation.chunk_index],
                    "reasoning": evaluation.reasoning
                })
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k documents (no arbitrary threshold, always return best available)
        final_docs = [item['document'] for item in scored_docs[:top_k]]
        
        elapsed_time = time.time() - start_time
        print(f"   -> Batch reranking completed in {elapsed_time:.2f}s. {len(final_docs)} chunks retained.")
        
        # Optional: Log top scores for debugging
        if scored_docs:
            print(f"   -> Top 3 scores: {[f'{d['score']:.2f}' for d in scored_docs[:3]]}")
        
        return final_docs
    
    except Exception as e:
        print(f"   -> Error during batch reranking: {e}. Returning top {top_k} docs by retrieval order.")
        return retrieved_docs[:top_k]

# --- Vector Store Initialization ---

def initialize_vectorstore(pdf_paths: List[str]) -> Optional[object]:
    """Loads multiple PDFs, splits documents, and creates a single vector store."""
    all_documents = []
    
    print("\n=== VECTOR STORE INITIALIZATION ===")
    for path in pdf_paths:
        print(f"Loading: {path}")
        try:
            loader = PyMuPDFLoader(path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = os.path.basename(path)
                
            all_documents.extend(documents)
            print(f"   -> {len(documents)} pages loaded from {os.path.basename(path)}")
            
        except Exception as e:
            print(f"   -> Error loading '{path}': {e}")
    
    if not all_documents:
        print("No documents loaded. Creating fallback document.")
        all_documents = [Document(
            page_content="No source documents available. Answers will be based on general LLM knowledge.",
            metadata={'source': 'Fallback', 'page': 'N/A'}
        )]

    print(f"\nTotal pages loaded: {len(all_documents)}")
    
    # Optimized chunking parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Larger chunks for better context
        chunk_overlap=50     # Reduced overlap to minimize redundancy
    )
    doc_chunks = text_splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(doc_chunks)}")

    # Build vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings
    )

    # Retriever with optimal initial candidate count
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}  # Retrieve 6 candidates for reranking to top 5
    )
    
    print("Vector Store and Retriever initialized.\n")
    return retriever

# Initialize retriever
doc_retriever = initialize_vectorstore(PDF_FILES_PATHS)

# --- Prompt Configuration ---

try:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template_file("prompt_system_2.txt", input_variables=[]),
        SystemMessagePromptTemplate.from_template_file("prompt_context.txt", input_variables=["context"]),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    print("Prompt templates loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load prompt files. Using default prompt. Error: {e}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert bicycle mechanic. Answer the question based on the provided context. Be precise and technical."),
        ("system", "CONTEXT:\n{context}"),
        ("user", "{question}")
    ])

# --- RAG Chain with Optimized Reranking ---

def ask_mechanic(question: str) -> Tuple[str, List[Document]]:
    """
    Executes the RAG chain with optimized batch reranking.
    Returns the answer and the final documents used.
    """
    print(f"\n=== QUERY EXECUTION ===")
    print(f"Question: {question}")
    
    start_time = time.time()
    
    # Step 1: Retrieve initial candidate documents
    print("\nStep 1: Retrieving candidate chunks...")
    initial_docs = doc_retriever.invoke(question)
    print(f"   -> {len(initial_docs)} candidates retrieved")

    # Step 2: Batch reranking (single API call for all chunks)
    print("\nStep 2: Batch reranking...")
    final_docs = rerank_chunks_batch(question, initial_docs, top_k=5)

    # Step 3: Format context for main LLM
    context_for_llm = format_docs(final_docs)
    
    # Step 4: Execute main RAG chain
    print("\nStep 3: Generating answer...")
    final_rag_chain = (
        prompt 
        | llm_main
        | output_parser
    )
    
    try:
        response = final_rag_chain.invoke({"context": context_for_llm, "question": question})
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Query completed in {elapsed_time:.2f}s")
        
        return response, final_docs
    
    except OutputParserException as e:
        return f"Output parsing error: {e}", []
    except Exception as e:
        return f"Unexpected error during LLM call: {e}", []

# --- Testing ---

if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("BIKE MECHANIC ASSISTANT - RAG SYSTEM")
    print("="*50)
    
    test_questions = [
        "What is the ideal tire pressure for road cycling in wet conditions? I weigh 65kg",
        "How do I adjust my front derailleur?",
        "What tools do I need for a complete bike overhaul?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"TEST QUESTION {i}")
        print(f"{'='*50}")
        
        response, sources = ask_mechanic(question)
        
        print("\n--- LLM RESPONSE ---")
        print(response)
        
        print("\n--- SOURCES USED (AFTER RERANKING) ---")
        if sources:
            for j, doc in enumerate(sources, 1):
                source_info = doc.metadata.get('source', 'Unknown')
                page_info = doc.metadata.get('page', 'Unknown')
                print(f"\nChunk {j} (Source: {source_info}, Page: {page_info})")
                print(f"{doc.page_content[:200]}...")
                print("-" * 30)
        else:
            print("No sources used.")
        
        if i < len(test_questions):
            print("\n" + "="*50 + "\n")
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)