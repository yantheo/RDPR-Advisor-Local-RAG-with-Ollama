#1. Ingest PDF files
#2. Extract Text from PDF and split into small chunks
#3. Send the chunks to the embedding model
#4. Save the embeddings to a vector database
#5. Perform similarity search on the vector database to find similarity
#6. Retrieve the similar documents and present them to the user

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import ollama
import pymupdf4llm
import re

doc_path = "./data/RGPD_UE.pdf"
md_path = "./data/RGPD_UE.md"
model = "qwen2.5:7b"

# ------------------------------------------------------------------
# 1. CONVERSION PDF → MARKDOWN
# ------------------------------------------------------------------
print("Conversion PDF → Markdown...")
md_text = pymupdf4llm.to_markdown(doc_path)

md_text = re.sub(r'Journal officiel de l\'Union européenne[^\n]*', '', md_text)
md_text = re.sub(r'L\s*\d+/\d+\s*\n', '', md_text)
md_text = re.sub(r'\d+\.\d+\.\d+\s*\nFR\s*\n', '', md_text)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_text)
print(f"Done — {len(md_text)} caractères")

# ------------------------------------------------------------------
# 2. CHARGEMENT DU MARKDOWN
# ------------------------------------------------------------------
loader = UnstructuredMarkdownLoader(md_path)
data = loader.load()
print(f"\nDone loading — {len(data)} documents")

# ------------------------------------------------------------------
# 3. CHUNKING
# ------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,    # Plus petit = plus précis pour les définitions
    chunk_overlap=200,
    separators=["\n## ", "\n_Article", "\n- ", "\n\n", "\n"] 
    # Le "- " permet de couper proprement entre chaque définition de l'Article 4
)
chunks = text_splitter.split_documents(data)
print(f"Done splitting — {len(chunks)} chunks")

# ------------------------------------------------------------------
# 4. VECTOR DB
# ------------------------------------------------------------------
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

ollama.pull("qwen2.5:7b")
ollama.pull("bge-m3")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="bge-m3"),
    collection_name="rag-rgpd-markdown",
)
print("Done adding to vector database")

# ------------------------------------------------------------------
# 5. RETRIEVAL
# ------------------------------------------------------------------
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model=model, temperature=0)

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5

semantic = vector_db.as_retriever(search_kwargs={"k": 5})

retriever = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.6, 0.4]
)

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

# ------------------------------------------------------------------
# 6. TESTS
# ------------------------------------------------------------------
questions = [
    "Qu'est-ce qu'une violation de données à caractère personnel selon l'Article 4 ?",
    "Est-ce que je peux refuser que mes données soient utilisées pour de la publicité ?",
    "Dans quel délai une entreprise doit-elle répondre à ma demande d'accès à mes données ?"
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"QUESTION : {question}")
    print(f"{'='*50}")
    res = chain.invoke(input=(question))
    print(res)