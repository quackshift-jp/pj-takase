import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv(verbose=True)

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
FAISS_DB_DIR = "vectorstore"

llm_model = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    response_format={"type": "json_object"}
)


def load_text_documents(path: str) -> list[Document]:
    loader = DirectoryLoader(path=path, loader_cls=TextLoader, glob="*.txt")
    return loader.load()


def split_documents_to_chunk(documents: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
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


def detect_text_error(
    ocr_data: dict[str, any], retrieval_chain: RetrievalQA
) -> dict[str, str]:
    """業者記入テキストと事前格納したドキュメントを比較して、エラー検知を行う
    args:
    ocr_data dict[str,any]: OCRのテキスト
    ex)
        {
            "物件名": "物件A",
            "管理会社": "会社A",
            "賃料": 111,
            "退去条件": "退去条件A",
        }

    return dict[str,any]: エラー検知結果を報告
    ex)
        {
            "物件名": "結果",
            "管理会社": "結果",
            "賃料": "Warning: {正しい内容}ではないですか？",
            "退去条件":  "Warning: {正しい内容}ではないですか？",
        }
    """

    prompt = f"""
    contextだけに基づいて回答してください。
    contextに基づいてもわからない場合は、「検索結果が見つかりませんでした。」と答えてください。

    - [質問内容]
    JSON形式の入力とcontextを比較して、文脈として各項目の入力内容が正しいかを比較してください。
    「物件名・管理会社・賃料・退去条件」それぞれに対して、「正しく入力されています」または、「文脈または文字が異なっています」または、「検索結果が見つかりませんでした。」で回答するようにしてください。
    物件名、管理会社、賃料に関してはcontextと同じ文字が入っているかを確認してください。

    - [入力]
    {ocr_data}

    JSON形式の出力形式を指定します。
    - [出力形式]
    {{
    "output":[
        {{
        "物件名": str型,
        "管理会社": str型,
        "賃料": str型,
        "退去条件": str型
        }},
    ]
    }}
    """
    response = retrieval_chain({"prompt": prompt})
    return response


def main(path: str, ocr_text: str):
    documents = load_text_documents(path=path)
    chunks = split_documents_to_chunk(documents)
    save_vector_store(chunked_documents=chunks, save_dir=FAISS_DB_DIR)
    retrieval_chain = get_retrieval_chain(
        faiss_db=FAISS.load_local(FAISS_DB_DIR, embeddings=OpenAIEmbeddings()),
        llm_model=llm_model,
    )
    return detect_text_error(ocr_text, retrieval_chain)
