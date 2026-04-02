import ollama
import pymupdf4llm
import re
import streamlit as st
import logging
import os

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough


# Configure logging
logging.basicConfig(level=logging.INFO)

doc_path = "./data/RGPD_UE.pdf"
md_path = "./data/RGPD_UE.md"
MODEL = "qwen2.5:7b"
EMBEDDING_MODEL= "bge-m3"
PERSIST_DIRECTORY = "./chroma_db"
VECTOR_STORE_NAME = "rag-rgpd-streamlit"
llm = ChatOllama(model=MODEL, temperature=0)


def ingest_pdf_to_md(doc_path, md_path):
    """
    Convertit un PDF en fichier Markdown, nettoie les en-têtes parasites
    (numéros de page, mentions Journal officiel) et charge le contenu
    via UnstructuredMarkdownLoader.

    Args:
        doc_path (str): Chemin vers le fichier PDF source.
        md_path (str): Chemin vers le fichier Markdown de sortie.

    Returns:
        list: Liste de documents LangChain chargés depuis le Markdown.
    """
    print("Conversion PDF → Markdown...")
    md_text = pymupdf4llm.to_markdown(doc_path)
    # Supprime les en-têtes de page type "L 119/36 FR" et "4.5.2016 FR"
    md_text = re.sub(r'^L\s*\d+/\d+\s*(?:FR)?\s*$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^\d{1,2}\.\d{1,2}\.\d{4}\s*(?:FR)?\s*$', '', md_text, flags=re.MULTILINE)
    # Supprime uniquement les lignes isolées "Journal officiel..." (pas dans les phrases)
    md_text = re.sub(r'^Journal officiel de l\'Union européenne[^\n]*$', '', md_text, flags=re.MULTILINE)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Done — {len(md_text)} caractères")
    loader = UnstructuredMarkdownLoader(md_path)
    data = loader.load()
    print(f"\nDone loading — {len(data)} documents")
    return data


def split_documents(documents):
    """
    Découpe les documents en chunks de taille fixe avec chevauchement,
    en respectant les séparateurs Markdown (titres, articles, listes).

    Args:
        documents (list): Liste de documents LangChain à découper.

    Returns:
        list: Liste de chunks (fragments) prêts pour l'embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400,
        separators=["\n## _Article", "\n## ", "\n- ", "\n\n", "\n"]
    )
    chunks = text_splitter.split_documents(documents)
    # Tagger chaque chunk selon son type et propager le numéro d'article
    current_article = None
    for chunk in chunks:
        match = re.match(r'Article\s+(\d+|premier)\b', chunk.page_content.strip())
        if match:
            current_article = match.group(1)
            chunk.metadata["type"] = "article"
            chunk.metadata["article"] = current_article
        elif current_article:
            chunk.metadata["type"] = "article"
            chunk.metadata["article"] = current_article
            chunk.page_content = f"[Article {current_article}]\n{chunk.page_content}"
        elif re.search(r'^\(\d+\)', chunk.page_content.strip()):
            chunk.metadata["type"] = "considerant"
        else:
            chunk.metadata["type"] = "other"
    
    print(f"Done splitting — {len(chunks)} chunks")
    return chunks


def load_vector_db():
    """
    Génère les embeddings des chunks via Ollama et les persiste
    dans une base vectorielle Chroma.

    Args:
        chunks (list): Liste de chunks à vectoriser.

    Returns:
        Chroma: Instance de la base vectorielle prête pour la recherche sémantique.
    """
    ollama.pull(EMBEDDING_MODEL)
    embedding=OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Loading existing directory")
        loader = UnstructuredMarkdownLoader(md_path)
        data = loader.load()
        chunks = split_documents(data)
        return vector_db, chunks
    else:
        # Load and process the PDF document
        data = ingest_pdf_to_md(doc_path, md_path)
        if data is None:
            return None
        
        # Slpit the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector database created and persisted")
    return vector_db, chunks


def create_retriever(vector_db, chunks):
    """
    Crée un retriever hybride combinant BM25 (recherche lexicale)
    et Chroma (recherche sémantique) via un EnsembleRetriever.

    Args:
        vector_db (Chroma): Base vectorielle pour la recherche sémantique.
        chunks (list): Chunks utilisés pour initialiser le BM25Retriever.

    Returns:
        EnsembleRetriever: Retriever hybride (60% BM25 / 40% sémantique).
    """
    # Filtrer les chunks BM25 sur les articles uniquement
    article_chunks = [c for c in chunks if c.metadata.get("type") == "article"]
    
    bm25 = BM25Retriever.from_documents(article_chunks)  # ← plus de considérants
    bm25.k = 8

    semantic = vector_db.as_retriever(
        search_kwargs={"k": 8}
    )
    
    retriever = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.4, 0.6]
    )
    return retriever


def create_chain(retriever, llm):
    """
    Construit la chaîne RAG complète : retriever → prompt → LLM → parser.
    Le prompt impose un format de réponse juridique strict basé sur les articles du RGPD.

    Args:
        retriever (EnsembleRetriever): Retriever hybride pour la recherche de contexte.
        llm (ChatOllama): Modèle de langage pour la génération de réponses.

    Returns:
        Runnable: Chaîne LangChain prête à être invoquée avec une question.
    """
    template = """Tu es un expert juridique RGPD. Réponds UNIQUEMENT à partir du contexte ci-dessous.
    Règles impératives :
    - Commence TOUJOURS ta réponse par "Article X, paragraphe Y :" avant de citer le contenu
    - Cite les montants et les âges EXACTEMENT tels qu'ils apparaissent dans le texte
    - Reproduis mot pour mot les formulations légales, sans les reformuler ni les résumer
    - N'utilise jamais tes propres catégories si elles ne sont pas dans le texte
    - Si l'information n'est pas dans le contexte, réponds : "Information non trouvée dans le document fourni"
    CONTEXTE :
    {context}
    QUESTION : {question}
    RÉPONSE (commence par "Article X, paragraphe Y :") :"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    """
    Fonction principale : orchestre le pipeline RAG complet.
    Charge le PDF, découpe en chunks, crée la base vectorielle,
    initialise le retriever et la chain, puis exécute une question exemple.
    """
    st.title("Document Assistant RGPD")
    
    # User input
    user_input = st.text_input("Entrez votre question:")

    if user_input:
        with st.spinner("Chargement de la réponse..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL, temperature=0)

                #load the vector database
                result = load_vector_db()
                if result is None:
                    st.error("Failed to load or create the vector database.")
                    return
                vector_db, chunk = result
                
                # Create the retriever
                retriever = create_retriever(vector_db, chunk)

                # Create the chain 
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("***Assistant***")
                st.write(response)
            except Exception as e:
                st.error(f"An error occured: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()