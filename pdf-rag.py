#1. Ingest PDF files
#2. Extract Text from PDF and split into small chunks
#3. Send the chunks to the embedding model
#4. Save the embeddings to a vector database
#5. Perform similarity search on the vector database to find similarity
#6. Retrieve the similar documents and present them to the user

from langchain_community.document_loaders import UnstructuredPDFLoader
import ollama

doc_path = "./data/RGPD_UE.pdf"
model = "llama3.2"

if doc_path:
    loader = UnstructuredPDFLoader(
        file_path=doc_path,
        languages=["fra"]
    )
    data = loader.load()
    print("Done loading....")
else:
    print("Upload a PDF file")

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
    separators=["\nArticle ", "Article ", "\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(data)
print("Done splitting....")

ollama.pull("bge-m3")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="bge-m3"),
    collection_name="rag-rgpd",
)
print("Done adding to vector database")

from langchain_classic.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model=model, temperature=0)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Génère 3 reformulations différentes de la question suivante,
une par ligne, sans numérotation, pour améliorer la recherche documentaire.

Question originale : {question}
""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
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

res = chain.invoke(input=("Quelles sont les amendes administratives prévues par l'Article 83 du RGPD ?"))
print(res)