import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv(verbose=True)

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
FAISS_DB_DIR = "vectorstore"


llm_model = ChatOpenAI(model="gpt-3.5-turbo-1106")


def load_text_documents(path: str) -> list[Document]:
    loader = DirectoryLoader(path=path, loader_cls=TextLoader, glob="*.txt")
    return loader.load()


def split_documents_to_chunk(documents: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def save_vector_store(chunked_documents: list[Document], save_dir: str) -> None:
    faiss_db = FAISS.from_documents(
        documents=chunked_documents, embedding=OpenAIEmbeddings()
    )
    faiss_db.save_local(save_dir)


def get_retrieval_chain(faiss_db: FAISS, llm_model: ChatOpenAI) -> RetrievalQA:
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        return_source_documents=True,
        input_key="prompt",
    )


def extract_context(
    ocr_data: dict[str, any], retrieval_chain: RetrievalQA
) -> dict[str, str]:
    
    template="""
    {key}に関連する内容をcontextから全て抜き出してください
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["key"], 
    )
    
    substracted = {}
    for key in ocr_data.keys():
        prompt_text=prompt.format(key=key)
        response=retrieval_chain({"prompt": prompt_text})
        substracted[key]=response.get("result")
    return substracted

def detect_sentence_error(
    ocr_data: dict[str, any], accurate_data: dict[str, any]
    ) -> dict[str, str]:
    
    llm = OpenAI(temperature=0.0)

    template="""
     Ocrの内容がContextの内容に含まれない場合「社内ドキュメントに含まれません」と回答してください。

    出力例：
    社内ドキュメントには「○○○○○○○○」と書かれています。

     Ocr：{ocr}
     Context：{context}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["ocr", "context"], 
    )


    error_detection = {}
    for key in ocr_data.keys():
        prompt_text=prompt.format(ocr=key+':'+ocr_data[key],context=accurate_data[key])
        response=llm(prompt_text)
        error_detection[key]=response

    return error_detection


def main(path: str, ocr_text: str):
    documents = load_text_documents(path=path)
    chunks = split_documents_to_chunk(documents)
    save_vector_store(chunked_documents=chunks, save_dir=FAISS_DB_DIR)
    retrieval_chain = get_retrieval_chain(
        faiss_db=FAISS.load_local(FAISS_DB_DIR, embeddings=OpenAIEmbeddings()),
        llm_model=llm_model,
    )
    accurate_data=extract_context(ocr_text, retrieval_chain)
    error=detect_sentence_error(ocr_text, accurate_data)
    return error


if __name__ == "__main__":
    path = "data/compliance_data"
    ocr_text = {
        "退去条件": "乙が甲を退職した場合",
        "社宅使用料":"全校の支払は、毎月25日（金融機関の休業日はその前日）までに翌月分の使用料を、乙の銀行口座より自動引き落としにて行う。",
    }
    result=main(path, ocr_text)
    print(result)



