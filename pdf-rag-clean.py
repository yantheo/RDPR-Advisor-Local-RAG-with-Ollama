import ollama
import pymupdf4llm
import re

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

doc_path = "./data/RGPD_UE.pdf"
md_path = "./data/RGPD_UE.md"
MODEL = "qwen2.5:7b"
EMBEDDING_MODEL= "bge-m3"
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
    md_text = re.sub(r'Journal officiel de l\'Union européenne[^\n]*', '', md_text)
    md_text = re.sub(r'L\s*\d+/\d+\s*\n', '', md_text)
    md_text = re.sub(r'\d+\.\d+\.\d+\s*\nFR\s*\n', '', md_text)
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
        chunk_overlap=200,
        separators=["\n## ", "\n_Article", "\n- ", "\n\n", "\n"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Done splitting — {len(chunks)} chunks")
    return chunks


def create_vector_db(chunks):
    """
    Génère les embeddings des chunks via Ollama et les persiste
    dans une base vectorielle Chroma.

    Args:
        chunks (list): Liste de chunks à vectoriser.

    Returns:
        Chroma: Instance de la base vectorielle prête pour la recherche sémantique.
    """
    ollama.pull(EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name="rag-rgpd-markdown",
    )
    print("Done adding to vector database")
    return vector_db


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
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    semantic = vector_db.as_retriever(search_kwargs={"k": 5})
    retriever = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.6, 0.4]
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
    data = ingest_pdf_to_md(doc_path, md_path)
    if data is None:
        return

    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)
    llm = ChatOllama(model=MODEL, temperature=0)
    retriever = create_retriever(vector_db, chunks)
    chain = create_chain(retriever, llm)

    question = "Qu'est-ce qu'une violation de données à caractère personnel selon l'Article 4 ?"

    res = chain.invoke(input=question)
    print(question)
    print(res)


if __name__ == "__main__":
    main()