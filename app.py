import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document 

# --- Configuration et Initialisation ---

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY non trouvée. Veuillez la définir dans votre fichier .env.")
    st.stop() 

PDF_FILE_PATH = "doc_shimano.pdf"
output_parser = StrOutputParser()

def format_docs(docs):
    """Aplatit la liste de documents récupérés en une seule chaîne de texte."""
    return "\n\n".join([f"--- Page {doc.metadata.get('page', 'N/A') + 1} ---\n{doc.page_content}" for doc in docs])

@st.cache_resource
def initialize_vectorstore(pdf_path: str):
    """Charge le PDF, découpe les documents et crée le vector store."""
    embeddings = OpenAIEmbeddings() 
    
    if not os.path.exists(pdf_path):
        st.warning(f"Fichier PDF non trouvé à l'emplacement: {pdf_path}. Le RAG utilisera une base vide.")
        empty_doc = Document(
            page_content="Base de connaissances interne vide. Veuillez ajouter 'doc_shimano.pdf' pour des réponses expertes.", 
            metadata={"source": "EMPTY_BASE"}
        )
        vectorstore = Chroma.from_documents(documents=[empty_doc], embedding=embeddings)
    else:
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"Erreur lors du chargement du PDF: {e}")
            error_doc = Document(
                page_content="Erreur de chargement du PDF. Réponse basée sur la connaissance générale du modèle.", 
                metadata={"source": "ERROR_LOAD"}
            )
            documents = [error_doc]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        doc_chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embeddings)

    return vectorstore.as_retriever(top_k=5) 

doc_retriever = initialize_vectorstore(PDF_FILE_PATH)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=800 
)

try:
    system_prompt_path = "prompt_system.txt"
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
        ("system", "Tu es un mécanicien vélo expert. Réponds à la question en te basant sur le contexte fourni."),
        ("system", "CONTEXTE : {context}"),
        ("user", "{question}")
    ])

def get_rag_chain():
    """Crée et retourne la chaîne RAG pour l'exécution."""
    retrieval_input_chain = (
        lambda x: x["question"]
    ) | doc_retriever | format_docs
    
    rag_chain = (
        RunnablePassthrough.assign(context=retrieval_input_chain) 
        | prompt
        | llm
        | output_parser
    )
    return rag_chain

rag_chain = get_rag_chain()

# --- Logique de l'Application Streamlit (Mode Chat) ---

st.set_page_config(page_title="Assistant Mécanicien Vélo 🚲", layout="wide")

# 1. Initialisation des variables d'état (Session)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre Assistant Mécanicien Vélo Expert. Posez-moi une question technique sur votre vélo."}
    ]
if "historique_sauvegarde" not in st.session_state:
    st.session_state["historique_sauvegarde"] = []
# 💡 NOUVEAU: ID de la discussion actuelle. -1 = nouvelle; Index = discussion chargée.
if "current_discussion_id" not in st.session_state:
    st.session_state["current_discussion_id"] = -1

# =======================================================
# Menu Latéral (st.sidebar)
# =======================================================

with st.sidebar:
    st.markdown("### ✏️ Nouvelle discussion")
    if st.button("Démarrer une nouvelle conversation", use_container_width=True):
        
        discussion_courante = st.session_state["messages"]
        
        # 1. SAUVEGARDE / MISE À JOUR de la discussion avant de réinitialiser
        if len(discussion_courante) > 1:
            
            titre = discussion_courante[1]['content'].split('\n')[0][:30] 
            if len(discussion_courante[1]['content']) > 30:
                titre += "..."
                
            # Si c'est une nouvelle discussion, on l'ajoute
            if st.session_state["current_discussion_id"] == -1:
                st.session_state["historique_sauvegarde"].append({
                    "titre": titre,
                    "messages": discussion_courante
                })
            # Sinon, on met à jour l'entrée existante pour éviter le duplicata
            else:
                index = st.session_state["current_discussion_id"]
                st.session_state["historique_sauvegarde"][index]["messages"] = discussion_courante
        
        # 2. RÉINITIALISER la session pour un nouveau chat
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Bonjour ! Je suis votre Assistant Mécanicien Vélo Expert. Posez-moi une question technique sur votre vélo."}
        ]
        st.session_state["current_discussion_id"] = -1 # Marque la nouvelle discussion
        st.rerun() 
        
    st.divider()

    st.markdown("### Discussions Récentes")
    
    if st.session_state["historique_sauvegarde"]:
        # Afficher du plus récent au plus ancien
        for i, discussion in enumerate(reversed(st.session_state["historique_sauvegarde"])):
            titre = discussion["titre"]
            # Calcule l'index réel dans la liste non inversée pour le mettre dans current_discussion_id
            index_reel = len(st.session_state["historique_sauvegarde"]) - 1 - i 
            
            # Mettre en évidence la discussion actuellement chargée
            is_current = st.session_state["current_discussion_id"] == index_reel
            
            if st.button(titre, key=f"disc_sauv_{i}", use_container_width=True, type=("primary" if is_current else "secondary")):
                # Charger la discussion sauvegardée
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

# 3. Affichage de l'historique des messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            is_welcome_message = message["content"].startswith("Bonjour !")
            is_empty_source = message["sources"][0] == "Aucune source pertinente trouvée dans la base de connaissances (ou base vide)."
            
            if not is_welcome_message:
                with st.expander("📚 Sources Utilisées (Transparence RAG)"):
                    if is_empty_source:
                        st.info(message["sources"][0])
                    else:
                        for item in message["sources"]:
                            st.markdown(item)


# 4. Gestion de la nouvelle entrée utilisateur via st.chat_input
if user_question := st.chat_input("Votre question technique :"):
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse du problème, recherche dans les manuels et rédaction de la procédure..."):
            
            try:
                retrieved_docs = doc_retriever.invoke(user_question)
                response = rag_chain.invoke({"question": user_question})
                st.markdown(response)
                
                # --- Préparation des sources pour l'historique ---
                sources_markdown = []
                is_valid_source = not (len(retrieved_docs) == 1 and retrieved_docs[0].metadata.get('source') in ["EMPTY_BASE", "ERROR_LOAD"])

                if retrieved_docs and is_valid_source:
                    for i, doc in enumerate(retrieved_docs):
                        page = doc.metadata.get('page', 'N/A')
                        source = doc.metadata.get('source', 'Inconnu')
                        
                        sources_markdown.append(
                            f"**Chunk {i+1}** (Page {page+1} du document **{os.path.basename(source)}**):\n```markdown\n{doc.page_content}\n```"
                        )
                    
                    with st.expander("📚 Sources Utilisées (Transparence RAG)"):
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
                
            # 5. Ajouter la réponse complète (texte + sources) à l'historique de la session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": sources_markdown
            })