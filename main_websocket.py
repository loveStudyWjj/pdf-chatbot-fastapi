import json
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import asyncio
import uvicorn
from loguru import logger

app = FastAPI()

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('vectorstore_db'):
    os.mkdir('vectorstore_db')

template = """
Answer the question based on the context, in a concise manner, in markdown and using bullet points where applicable.

Context: {context}
History: {history}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],  # Inputs
    template=template,  # Template
)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
)

vector_store = Chroma(
    persist_directory='vectorstore_db',
    embedding_function=OllamaEmbeddings(
        base_url='http://127.0.0.1:11434',
        model="mistral"
    )
)


def delete_document(file_path):
    coll = vector_store.get()
    ids_to_del = [id for idx, id in enumerate(coll['ids']) if coll['metadatas'][idx]['source_file_path'] == file_path]
    if ids_to_del:
        vector_store._collection.delete(ids_to_del)


llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="mistral",
    verbose=True,
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]
    ),
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len
)


def handle_get_documents():
    coll = vector_store.get()
    source_file_paths = [metadata['source_file_path'] for metadata in coll['metadatas']]
    return list(set(source_file_paths))


def upload_pdf_to_vectorstore_db(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {"source_file_path": file_path.split("/")[-1]}

    vector_store.add_documents(docs)
    print(f"Successfully uploaded {len(docs)} documents from {file_path}")


@app.get("/get_documents/")
def get_documents():
    documents = handle_get_documents()
    return {"data": documents}


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    logger.info(f"get pdf:{file_location}")
    with open(file_location, "wb") as f:
        f.write(await file.read())
    upload_pdf_to_vectorstore_db(file_location)
    return {"message": "PDF uploaded and processed successfully"}


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )
    try:
        while True:
            user_input = await websocket.receive_text()
            response = qa_chain.invoke(user_input)
            answer = response["result"]
            context = response["source_documents"]
            for chunk in answer.split(" "):
                await websocket.send_text(json.dumps({"event_type": "answer", "data": chunk + " "}, ensure_ascii=False))
                await asyncio.sleep(0.05)

            file_names = [f"**{doc.dict().get('metadata', {}).get('source_file_path', '')}**" for doc
                          in context]
            if file_names:
                document = "\n".join(list(set(file_names)))
                await websocket.send_text(
                    json.dumps({"event_type": "answer", "data": f"\n\nSource PDF：\n\n{document}"}, ensure_ascii=False))
            # await websocket.send_text(
            #     json.dumps({"event_type": "document", "data": document}, ensure_ascii=False))
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
