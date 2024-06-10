from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from .types import ChatHistory
from langchain.embeddings import OpenAIEmbeddings
from .models import Message
from dotenv import load_dotenv

load_dotenv()
def get_conversation_chain(vectorstore: FAISS, chat_history: list[ChatHistory]):
    llm = ChatOpenAI(temperature=0.9)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Pre-load the chat history into the memory
    for msg in chat_history:
        if msg["sender"]:
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])
    

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def get_vectorstore():
    embeddings = OpenAIEmbeddings(chunk_size=1024)

    vectorstore = FAISS.load_local(
        "api/model",
        index_name="CODE DE LA FAMILLE.pdf",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def get_conversation_context(conversation_id, limit=10):
    messages = Message.objects.filter(conversation_id=conversation_id).order_by(
        "-timestamp"
    )[:limit]
    # Reverse the messages to maintain the order of the conversation
    return reversed(messages)
