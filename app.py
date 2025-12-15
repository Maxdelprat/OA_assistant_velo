import streamlit as st
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document 
from langchain_core.exceptions import OutputParserException 
from pydantic import BaseModel, Field

# Tavily Import
from tavily import TavilyClient

# --- Configuration et Initialisation ---

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY non trouvée. Veuillez la définir dans votre fichier .env.")
    st.stop() 

if not os.getenv("TAVILY_API_KEY"):
    st.warning("TAVILY_API_KEY non trouvée. La recherche Web ne sera pas fonctionnelle. Veuillez la définir dans votre fichier .env.")

PDF_FOLDER = "./document_pdfs"
if os.path.isdir(PDF_FOLDER):
    PDF_FILES_PATHS = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
else:
    PDF_FILES_PATHS = ["doc_shimano.pdf"]

try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception:
    tavily_client = None

output_parser = StrOutputParser()
RELEVANCE_THRESHOLD = 0.5

# LLM Configuration
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


# --- Fonction Utilitaires ---

def format_docs(docs: List[Document]) -> str:
    """Aplatit la liste de documents récupérés en une seule chaîne de texte, en incluant la source."""
    formatted_content = []
    for doc in docs:
        page_info = doc.metadata.get('page')
        page_display = page_info + 1 if isinstance(page_info, int) else 'N/A'
        
        source_base = os.path.basename(doc.metadata.get('source', 'Inconnu'))
        
        # Ajout de l'URL pour les sources Web (inchangé par rapport à la dernière version)
        url_info = doc.metadata.get('url', '')
        url_display = f" (URL: {url_info})" if url_info else ""
        
        formatted_content.append(f"--- Source: {source_base} (Page {page_display}){url_display} ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_content)

# --- Tavily Web Search Agent ---

@st.cache_data(show_spinner=False)
def search_web_with_tavily(question: str, max_results: int = 3) -> List[Document]:
    """Performs web search using Tavily API and converts results to Document objects."""
    if not tavily_client:
        return []
        
    st.info(f"Déclenchement de la recherche Web (Tavily) pour : '{question}'")
    
    try:
        response = tavily_client.search(
            query=question,
            search_depth="advanced",
            max_results=max_results,
            include_images=False,
            include_raw_content=False
        )
        
        web_docs = []
        
        if response.get("results"):
            for i, result in enumerate(response['results']):
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': f"Web: {result.get('title', 'Unknown')}",
                        'page': 'Web Search',
                        'url': result.get('url', ''),
                        'score': 0.8  
                    }
                )
                web_docs.append(doc)
            
            st.success(f"Tavily a retourné {len(web_docs)} résultats pertinents du Web.")
        return web_docs
    
    except Exception as e:
        st.error(f"Erreur lors de la recherche Tavily: {e}")
        return []

# --- Fonction de Reranking par Batch ---

def rerank_chunks_batch(question: str, retrieved_docs: List[Document], top_k: int = 5) -> Tuple[List[Document], List[float]]:
    """Utilise un LLM pour évaluer la pertinence et retourne les top_k documents et leurs scores."""
    if not retrieved_docs:
        return [], []
    
    chunks_text = ""
    for i, doc in enumerate(retrieved_docs):
        chunks_text += f"\n--- CHUNK {i} ---\n{doc.page_content}\n"
    
    rerank_prompt = f"""You are a relevance judge. Evaluate each chunk below for its relevance to the user's question.

Question: {question}

Chunks to evaluate:
{chunks_text}

For each chunk (0 to {len(retrieved_docs)-1}), provide:
- chunk_index: the chunk number
- is_relevant: true if it helps answer the question
- relevance_score: between 0.0 and 1.0
- reasoning: brief explanation

Evaluate ALL chunks."""

    try:
        rerank_chain = (
            ChatPromptTemplate.from_messages([("user", "{prompt}")])
            | llm_judge.with_structured_output(BatchRerankResult, method="function_calling")
        )
        
        result = rerank_chain.invoke({"prompt": rerank_prompt})
        
        scored_docs = []
        for evaluation in result.evaluations:
            if 0 <= evaluation.chunk_index < len(retrieved_docs):
                scored_docs.append({
                    "score": evaluation.relevance_score,
                    "document": retrieved_docs[evaluation.chunk_index],
                    "reasoning": evaluation.reasoning
                })
            else:
                 st.warning(f"Index de morceau invalide dans le résultat du reranker: {evaluation.chunk_index}")
        
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        final_docs = [item['document'] for item in scored_docs[:top_k]]
        final_scores = [item['score'] for item in scored_docs[:top_k]]
        
        return final_docs, final_scores
    
    except Exception as e:
        st.error(f"Erreur lors du reranking par lot : {e}. Utilisation des {top_k} premiers documents récupérés.")
        return retrieved_docs[:top_k], [0.5] * min(top_k, len(retrieved_docs))


# --- Enhanced RAG with Tavily Fallback ---

def enhance_context_with_tavily(question: str, docs: List[Document], scores: List[float], threshold: float = 0.5) -> List[Document]:
    """Vérifie si des chunks sont sous le seuil de pertinence et les remplace par des résultats Web si Tavily est disponible."""
    if not tavily_client:
        return docs 

    low_score_indices = [i for i, score in enumerate(scores) if score < threshold]
    low_score_count = len(low_score_indices)
    
    if low_score_count == 0:
        return docs
    
    st.info(f"{low_score_count} chunk(s) avec un score < {threshold} détecté(s). Tentative d'enrichissement via Tavily...")
    
    web_docs = search_web_with_tavily(question, max_results=low_score_count)
    
    if not web_docs:
        return docs
    
    enhanced_docs = list(docs) 
    web_doc_index = 0
    
    for i, score in enumerate(scores):
        if score < threshold and web_doc_index < len(web_docs):
            enhanced_docs[i] = web_docs[web_doc_index]
            web_doc_index += 1
            
    if web_doc_index > 0:
         st.success(f"{web_doc_index} chunk(s) remplacé(s) par des informations Web pour enrichir la réponse.")
         
    return enhanced_docs


# --- Initialisation du Vector Store ---

@st.cache_resource
def initialize_vectorstore(pdf_paths: List[str]):
    """Charge les PDFs, découpe les documents et crée le vector store."""
    embeddings = OpenAIEmbeddings() 
    all_documents = []
    
    if not pdf_paths or (len(pdf_paths) == 1 and not os.path.exists(pdf_paths[0])):
        st.warning("Aucun fichier PDF trouvé ou base vide. Le RAG utilisera une base vide.")
        empty_doc = Document(
            page_content="Base de connaissances interne vide. Veuillez ajouter des PDFs pour des réponses expertes.", 
            metadata={"source": "EMPTY_BASE", "page": -1}
        )
        vectorstore = Chroma.from_documents(documents=[empty_doc], embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 1}) 
    
    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = path
                
            all_documents.extend(documents)
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du PDF '{os.path.basename(path)}': {e}")
            error_doc = Document(
                page_content="Erreur de chargement du PDF. Réponse basée sur la connaissance générale du modèle.", 
                metadata={"source": "ERROR_LOAD", "page": -1}
            )
            all_documents.append(error_doc)
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      
        chunk_overlap=50     
    )
    doc_chunks = text_splitter.split_documents(all_documents)
    
    vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 6}) 

# Initialisation du retriever
doc_retriever = initialize_vectorstore(PDF_FILES_PATHS)


# --- Configuration du Prompt ---

try:
    system_prompt_path = "prompt_system_2.txt"
    context_prompt_path = "prompt_context.txt"
    
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_template = f.read()
    with open(context_prompt_path, 'r', encoding='utf-8') as f:
        context_template = f.read()
        
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        SystemMessagePromptTemplate.from_template(context_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
except Exception as e:
    st.warning(f"Impossible de charger les fichiers de prompt. Utilisation d'un prompt par défaut. Erreur: {e}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un mécanicien vélo expert. Réponds à la question en te basant sur le contexte fourni. S'il n'y a pas de contexte, réponds avec ta connaissance générale."),
        ("system", "CONTEXTE : {context}"),
        ("user", "{question}")
    ])


# --- Fonction Principale de RAG (avec Tavily Fallback) ---

def ask_mechanic(question: str, relevance_threshold: float = RELEVANCE_THRESHOLD) -> Tuple[str, List[Document]]:
    """
    Exécute la chaîne RAG avec récupération initiale, reranking par lot et enrichissement Tavily.
    """
    
    initial_docs = doc_retriever.invoke(question)
    
    with st.spinner("Évaluation de la pertinence des manuels internes..."):
        ranked_docs, scores = rerank_chunks_batch(question, initial_docs, top_k=5)

    final_docs = enhance_context_with_tavily(question, ranked_docs, scores, threshold=relevance_threshold)

    context_for_llm = format_docs(final_docs)
    
    final_rag_chain = (
        prompt 
        | llm_main
        | output_parser
    )
    
    try:
        response = final_rag_chain.invoke({"context": context_for_llm, "question": question})
        return response, final_docs
    
    except OutputParserException as e:
        return f"Erreur de parsing de sortie de l'LLM: {e}", []
    except Exception as e:
        return f"Erreur inattendue lors de l'appel à l'LLM: {e}", []


# --- Logique de l'Application Streamlit (Mode Chat) ---

st.set_page_config(page_title="Assistant Mécanicien Vélo 🚲", layout="wide")

# 1. Initialisation des variables d'état (Session)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre Assistant Mécanicien Vélo Expert. Posez-moi une question technique sur votre vélo."}
    ]
if "historique_sauvegarde" not in st.session_state:
    st.session_state["historique_sauvegarde"] = []
if "current_discussion_id" not in st.session_state:
    st.session_state["current_discussion_id"] = -1

# =======================================================
# Menu Latéral (st.sidebar)
# =======================================================

with st.sidebar:
    st.markdown("### ✏️ Nouvelle discussion")
    if st.button("Démarrer une nouvelle conversation", use_container_width=True):
        
        discussion_courante = st.session_state["messages"]
        
        if len(discussion_courante) > 1:
            titre_message = next((msg['content'] for msg in discussion_courante if msg['role'] == 'assistant' and not msg['content'].startswith("Bonjour !")), None)
            
            if titre_message:
                titre = titre_message.split('\n')[0][:30] 
                if len(titre_message) > 30:
                    titre += "..."
            else:
                titre = "Nouvelle discussion" 
                
            if st.session_state["current_discussion_id"] == -1:
                st.session_state["historique_sauvegarde"].append({
                    "titre": titre,
                    "messages": discussion_courante
                })
            else:
                index = st.session_state["current_discussion_id"]
                st.session_state["historique_sauvegarde"][index]["messages"] = discussion_courante
                st.session_state["historique_sauvegarde"][index]["titre"] = titre
        
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Bonjour ! Je suis votre Assistant Mécanicien Vélo Expert. Posez-moi une question technique sur votre vélo."}
        ]
        st.session_state["current_discussion_id"] = -1
        st.rerun() 
        
    st.divider()

    st.markdown("### Discussions Récentes")
    
    if st.session_state["historique_sauvegarde"]:
        for i, discussion in enumerate(reversed(st.session_state["historique_sauvegarde"])):
            titre = discussion["titre"]
            index_reel = len(st.session_state["historique_sauvegarde"]) - 1 - i 
            
            is_current = st.session_state["current_discussion_id"] == index_reel
            
            if st.button(titre, key=f"disc_sauv_{i}", use_container_width=True, type=("primary" if is_current else "secondary")):
                st.session_state["messages"] = discussion["messages"]
                st.session_state["current_discussion_id"] = index_reel
                st.rerun() 
    else:
        st.info("Aucune discussion enregistrée pour le moment. Commencez une conversation !")
        
# =======================================================
# Fin du Menu Latéral
# =======================================================

st.title("🚲 Assistant Mécanicien Vélo Expert")
st.markdown("Posez-moi votre question de mécanique, de diagnostic ou de réparation de vélo (VTT, route, urbain, etc.).")

# Affichage de l'historique des messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            is_welcome_message = message["content"].startswith("Bonjour !")
            is_empty_source = message["sources"][0].startswith("Aucune source pertinente trouvée")
            
            if not is_welcome_message:
                with st.expander("📚 Sources Utilisées (Transparence RAG)"):
                    if is_empty_source:
                        st.info(message["sources"][0])
                    else:
                        for item in message["sources"]:
                            st.markdown(item)


# Gestion de la nouvelle entrée utilisateur via st.chat_input
if user_question := st.chat_input("Votre question technique :"):
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse du problème, recherche dans les manuels et sur le Web, et rédaction de la procédure..."):
            
            try:
                response, final_docs = ask_mechanic(user_question)
                st.markdown(response)
                
                # --- Préparation des sources pour l'historique (LOGIQUE MISE À JOUR ICI) ---
                sources_markdown = []
                is_valid_source = final_docs and not (len(final_docs) == 1 and final_docs[0].metadata.get('source') in ["EMPTY_BASE", "ERROR_LOAD"])

                # NOUVEAU: Liste pour collecter les liens YouTube à part
                youtube_links = []

                if is_valid_source:
                    for i, doc in enumerate(final_docs):
                        # Récupération des métadonnées y compris URL pour le Web
                        page_info = doc.metadata.get('page')
                        page_display = page_info + 1 if isinstance(page_info, int) else 'N/A'
                        source_base = os.path.basename(doc.metadata.get('source', 'Inconnu'))
                        url_info = doc.metadata.get('url', '')
                        
                        
                        # --- LOGIQUE SPÉCIFIQUE YOUTUBE ---
                        if url_info and "youtube.com" in url_info.lower():
                            youtube_links.append(url_info)
                            url_display = f" [▶️ Lien YouTube]" # Affichage simple dans le chunk
                        else:
                            url_display = f" [🔗 {url_info}]" if url_info else ""
                        # --- FIN LOGIQUE SPÉCIFIQUE YOUTUBE ---

                        
                        sources_markdown.append(
                            f"**Chunk {i+1}** (Source: **{source_base}** - Page {page_display}){url_display}:\n```markdown\n{doc.page_content}\n```"
                        )
                    
                    with st.expander("📚 Sources Utilisées (Transparence RAG - Rerankées et Enrichies)"):
                        
                        # NOUVEAU: Affichage des liens YouTube en haut
                        if youtube_links:
                            st.markdown("### ▶️ Vidéos YouTube Pertinentes :")
                            for link in youtube_links:
                                # Utilisation de st.video ou st.markdown avec iframe (markdown est plus simple)
                                st.markdown(f"- **[Voir la vidéo sur YouTube]({link})**")
                                # Pour une intégration directe (moins conseillé pour la performance): st.video(link)
                            st.divider()

                        for item in sources_markdown:
                            st.markdown(item)
                else:
                    sources_markdown.append("Aucune source pertinente trouvée dans la base de connaissances (ou base vide).")
                    with st.expander("📚 Sources Utilisées (Transparence RAG)"):
                        st.info(sources_markdown[0])


            except Exception as e:
                error_message = f"Une erreur s'est produite lors du traitement de la requête: {e}"
                st.error(error_message)
                st.exception(e) 
                response = error_message
                sources_markdown = ["Erreur lors du traitement."]
                youtube_links = [] # Assurer qu'elle existe même en cas d'erreur
                
            # 5. Ajouter la réponse complète (texte + sources) à l'historique de la session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": sources_markdown,
                "youtube_links": youtube_links # Sauvegarder les liens YouTube pour l'historique
            })