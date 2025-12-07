import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.exceptions import OutputParserException

# --- Configuration et Initialisation ---

# Charge les variables d'environnement (API key, etc.) depuis .env
load_dotenv()
# Assurez-vous que la clé API OpenAI est disponible dans l'environnement
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY non trouvée. Veuillez la définir dans votre fichier .env.")

PDF_FILE_PATH = "doc_shimano.pdf"

# --- Préparation du Vector Store (RAG) ---

# Variable globale pour stocker le retriever (il sera mis à jour pour être utilisé dans la chaîne)
doc_retriever = None 
# Nouvelle variable pour stocker le retriever des documents bruts
doc_retriever_docs = None

# Fonction pour formater les documents récupérés (Utilisée dans la chaîne)
def format_docs(docs):
    """Aplatit la liste de documents récupérés en une seule chaîne de texte."""
    return "\n\n".join(doc.page_content for doc in docs)

# Parser de sortie simple
output_parser = StrOutputParser()

def initialize_vectorstore(pdf_path: str):
    """Charge le PDF, découpe les documents et crée le vector store."""
    print(f"1. Chargement du document: {pdf_path}")
    
    # 1. Chargement et parsing du PDF
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Erreur lors du chargement du PDF : {e}")
        documents = [{"page_content": "Aucun document trouvé, les informations techniques seront basées sur la connaissance générale du modèle LLM."}]

    # 2. Découpage en chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    doc_chunks = text_splitter.split_documents(documents)
    print(f"  -> Nombre de chunks créés : {len(doc_chunks)}")

    # 3. Construction du vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings
    )

    # 4. Document retriever
    # Le retriever standard retourne une liste de Document (avec métadonnées)
    doc_retriever_docs = vectorstore.as_retriever(top_k=5) 
    # Le retriever pour le contexte (chaîne formatée) est construit à partir du premier
    doc_retriever_formatted = doc_retriever_docs | format_docs
    
    print("2. Vector Store et Retriever initialisés.")
    return doc_retriever_docs, doc_retriever_formatted # Retourne les deux versions

doc_retriever_docs, doc_retriever_formatted = initialize_vectorstore(PDF_FILE_PATH)


# --- Configuration du Modèle et du Prompt ---

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=500
)

# 1. Assemblage du Prompt (Même logique que l'original)
try:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template_file("prompt_system.txt", input_variables=[]),
        SystemMessagePromptTemplate.from_template_file("prompt_context.txt", input_variables=["context"]),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
except Exception as e:
    print(f"Avertissement: Impossible de charger les fichiers de prompt. Utilisation d'un prompt par défaut. Erreur: {e}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un mécanicien vélo expert. Réponds à la question en te basant sur le contexte fourni."),
        ("system", "CONTEXTE : {context}"),
        ("user", "{question}")
    ])




# --- Chaîne RAG ---

def ask_mechanic(question: str):
    """
    Exécute la chaîne RAG : Récupération -> Formatage -> Prompt -> LLM -> Parsing.
    """
    print(f"\n3. Exécution de la requête pour : '{question[:50]}...'")

    # Chaîne pour récupérer les documents bruts (utilisée pour l'affichage final)
    # L'argument `question` est passé au retriever.
    retrieval_chain = doc_retriever_docs

    # La chaîne RAG principale. Elle utilise RunnablePassthrough.assign
    # pour passer le résultat du retriever et la question initiale
    # à l'étape suivante (le prompt), puis au LLM.
    # Nous utilisons RunnableLambda pour pré-récupérer les documents bruts pour l'affichage
    rag_chain_with_sources = (
        # 1. Récupère les documents bruts
        RunnablePassthrough.assign(docs=retrieval_chain) 
        # 2. Formate le contexte pour le prompt et garde la question
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["docs"]),
            question=RunnablePassthrough()
        )
        # 3. Forme la requête au LLM
        | (lambda x: {"context": x["context"], "question": x["question"]} | prompt | llm | output_parser)
    )

    # 4. Exécution et gestion des erreurs
    try:
        # L'entrée est la question, la sortie est le dictionnaire final contenant 'context', 'question', 'docs' et 'response'
        # Note : On doit invoquer les étapes séparément pour obtenir les sources dans LangChain 0.2
        # La solution ci-dessus simule la structure de sortie souhaitée.
        
        # Étape 1: Récupérer les documents bruts (chunks)
        retrieved_docs = retrieval_chain.invoke(question)

        # Étape 2: Formater le contexte pour le LLM
        context_for_llm = format_docs(retrieved_docs)
        
        # Étape 3: Exécuter la chaîne LLM
        final_rag_chain = (
            prompt 
            | llm
            | output_parser
        )
        
        response = final_rag_chain.invoke({"context": context_for_llm, "question": question})

        return response, retrieved_docs # Retourne la réponse ET les documents bruts
    
    except OutputParserException as e:
        return f"Erreur de parsing de sortie : {e}", []
    except Exception as e:
        return f"Une erreur inattendue s'est produite lors de l'appel LLM : {e}", []

# --- Test de l'Agent ---

if __name__ == "__main__":
    
    print("\n--- TEST AGENT MÉCANICIEN VÉLO ---")
    
    # 1. Exemple de question technique pour tester le RAG
    question_1 = "A quoi servent les vis sur mon dérailleur arrière?"
    
    print(f"\n--- Question 1 : {question_1} ---")
    response_1, sources_1 = ask_mechanic(question_1)
    
    print("\n✅ Réponse du LLM :")
    print(response_1)
    
    print("\n📚 Sources (Chunks) utilisées :")
    for i, doc in enumerate(sources_1):
        # Affiche le contenu et la source (page, etc.) si disponible
        source_info = doc.metadata.get('source', 'Inconnu')
        page_info = doc.metadata.get('page', 'Inconnue')
        print(f"--- Chunk {i+1} (Source: {source_info}, Page: {page_info}) ---")
        print(doc.page_content)
        print("-" * 30)

    # 2. Exemple de question générale
    question_2 = "Comment régler correctement l'inclinaison de la selle?"
    
    print(f"\n--- Question 2 : {question_2} ---")
    response_2, sources_2 = ask_mechanic(question_2)
    
    print("\n✅ Réponse du LLM :")
    print(response_2)
    
    print("\n📚 Sources (Chunks) utilisées :")
    # Pour cette question générale, les sources seront probablement celles du fallback (si le fichier PDF n'existe pas)
    if sources_2:
        for i, doc in enumerate(sources_2):
            source_info = doc.metadata.get('source', 'Inconnu')
            page_info = doc.metadata.get('page', 'Inconnue')
            print(f"--- Chunk {i+1} (Source: {source_info}, Page: {page_info}) ---")
            print(doc.page_content)
            print("-" * 30)
    else:
        print("Aucune source spécifique utilisée pour cette question (réponse basée sur la connaissance interne du LLM).")