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

# Tavily Import
from tavily import TavilyClient

# --- Configuration et Initialisation ---

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found. Please set it in your .env file.")

PDF_FOLDER = "./document_pdfs"
PDF_FILES_PATHS = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

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
    temperature=0.7,
    max_tokens=800
)

llm_judge = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0.1,
    max_tokens=500
)

# --- Tavily Web Search Agent ---

def search_web_with_tavily(question: str, max_results: int = 3) -> List[Document]:
    """
    Performs web search using Tavily API and converts results to Document objects.
    Returns a list of Documents with web-sourced content.
    """
    print(f"   -> Performing Tavily web search for: '{question}'")
    
    try:
        response = tavily_client.search(
            query=question,
            search_depth="advanced",
            max_results=max_results,
            include_images=False,
            include_raw_content=False
        )
        
        web_docs = []
        
        # Convert Tavily results to LangChain Documents
        if response.get("results"):
            for i, result in enumerate(response['results']):
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': f"Web: {result.get('title', 'Unknown')}",
                        'page': 'Web Search',
                        'url': result.get('url', ''),
                        'score': 0.8  # Assign high score to web results
                    }
                )
                web_docs.append(doc)
            
            print(f"   -> Tavily returned {len(web_docs)} web results")
        else:
            print("   -> No web results found from Tavily")
        
        return web_docs
    
    except Exception as e:
        print(f"   -> Error during Tavily search: {e}")
        return []

# --- Batch Reranking Agent with Tavily Integration ---

def rerank_chunks_batch(question: str, retrieved_docs: List[Document], top_k: int = 5) -> Tuple[List[Document], List[float]]:
    """
    Uses an LLM to evaluate relevance of all chunks in a single batch call,
    then returns the top_k most relevant documents and their scores.
    Returns: (list of documents, list of scores)
    """
    if not retrieved_docs:
        return [], []
    
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
- relevance_score:  between 0.0 and 1.0
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
        
        # Extract documents and scores
        final_docs = [item['document'] for item in scored_docs[:top_k]]
        final_scores = [item['score'] for item in scored_docs[:top_k]]
        
        elapsed_time = time.time() - start_time
        print(f"   -> Batch reranking completed in {elapsed_time:.2f}s. {len(final_docs)} chunks retained.")
        
        # Log top scores for debugging
        if scored_docs:
            print(f"   -> Top scores: {[f'{score:.2f}' for score in final_scores]}")
        
        return final_docs, final_scores
    
    except Exception as e:
        print(f"   -> Error during batch reranking: {e}. Returning top {top_k} docs by retrieval order.")
        return retrieved_docs[:top_k], [0.5] * min(top_k, len(retrieved_docs))

# --- Enhanced RAG with Tavily Fallback ---

def enhance_context_with_tavily(question: str, docs: List[Document], scores: List[float], threshold: float = 0.5) -> List[Document]:
    """
    Checks if any chunk scores are below threshold. If so, performs Tavily search
    and replaces low-scoring chunks with web results.
    """
    # Check if we need web search
    low_score_count = sum(1 for score in scores if score < threshold)
    
    if low_score_count == 0:
        print(f"   -> All chunks above threshold ({threshold}). No web search needed.")
        return docs
    
    print(f"\n   -> {low_score_count} chunk(s) below threshold {threshold}. Triggering Tavily search...")
    
    # Perform web search
    web_docs = search_web_with_tavily(question, max_results=low_score_count)
    
    if not web_docs:
        print("   -> No web results available. Using original chunks.")
        return docs
    
    # Replace low-scoring chunks with web results
    enhanced_docs = []
    web_doc_index = 0
    
    for doc, score in zip(docs, scores):
        if score < threshold and web_doc_index < len(web_docs):
            print(f"   -> Replacing chunk (score: {score:.2f}) with web result")
            enhanced_docs.append(web_docs[web_doc_index])
            web_doc_index += 1
        else:
            enhanced_docs.append(doc)
    
    return enhanced_docs

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
        chunk_size=500,
        chunk_overlap=50
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
        search_kwargs={"k": 6}
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

# --- RAG Chain with Tavily Integration ---

def ask_mechanic(question: str, relevance_threshold: float = 0.5) -> Tuple[str, List[Document]]:
    """
    Executes the RAG chain with Tavily fallback for low-relevance chunks.
    Returns the answer and the final documents used.
    """
    print(f"\n=== QUERY EXECUTION ===")
    print(f"Question: {question}")
    
    start_time = time.time()
    
    # Step 1: Retrieve initial candidate documents
    print("\nStep 1: Retrieving candidate chunks...")
    initial_docs = doc_retriever.invoke(question)
    print(f"   -> {len(initial_docs)} candidates retrieved")

    # Step 2: Batch reranking (returns docs and scores)
    print("\nStep 2: Batch reranking...")
    ranked_docs, scores = rerank_chunks_batch(question, initial_docs, top_k=5)

    # Step 3: Enhance context with Tavily if needed
    print("\nStep 3: Checking relevance scores...")
    final_docs = enhance_context_with_tavily(question, ranked_docs, scores, threshold=relevance_threshold)

    # Step 4: Format context for main LLM
    context_for_llm = format_docs(final_docs)
    
    # Step 5: Execute main RAG chain
    print("\nStep 4: Generating answer...")
    final_rag_chain = (
        prompt 
        | llm_main
        | output_parser
    )
    
    try:
        response = final_rag_chain.invoke({"context": context_for_llm, "question": question})
        
        elapsed_time = time.time() - start_time
        print(f"\nQuery completed in {elapsed_time:.2f}s")
        
        return response, final_docs
    
    except OutputParserException as e:
        return f"Output parsing error: {e}", []
    except Exception as e:
        return f"Unexpected error during LLM call: {e}", []

# --- Testing ---

if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("BIKE MECHANIC ASSISTANT - RAG SYSTEM WITH TAVILY")
    print("="*50)
    
    test_questions = [
        "comment monter et démonter des pédales?",
        "Comment dévoiler une roue à rayons?",
        "Quand a été créée l'entreprise Shimano ?"  # Question likely to trigger web search
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTEST QUESTION {i}")
        
        response, sources = ask_mechanic(question, relevance_threshold=0.5)
        
        print("\n--- LLM RESPONSE ---")
        print(response)
        
        print("\n--- SOURCES USED ---")
        if sources:
            for j, doc in enumerate(sources, 1):
                source_info = doc.metadata.get('source', 'Unknown')
                page_info = doc.metadata.get('page', 'Unknown')
                url_info = doc.metadata.get('url', '')
                
                print(f"\nChunk {j} (Source: {source_info}, Page: {page_info})")
                if url_info:
                    print(f"URL: {url_info}")
                print(f"{doc.page_content[:200]}...")
                print("-" * 30)
        else:
            print("No sources used.")
        
        if i < len(test_questions):
            print("\n" + "="*50 + "\n")
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)