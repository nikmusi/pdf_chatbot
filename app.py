import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

from htmTemplates import css, bot_template, user_template


def load_pdf_text(pdf_docs):
    """Loads all given PDFs extracts the Text off all Pages of all PDFs and Returns a long String holding
    the content.

    Args:
        pdf_docs: list of PDF files, that the text should be extracted

    Returns:
        String: Readable Text included in the given PDF files
    """

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def split_text_into_chunks(full_text):
    """Splits the given Text into chunks of 800 Characters, with an overlap of 200 characters between each
    chunk.

    Args:
        full_text (String): the String that will be devided

    Returns:
        list: List of Strings containing the text chunks
    """

    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = r_splitter.split_text(full_text)
    return chunks


def embed_text_chunks(text_chunks):
    """Embedding the text chunks using the OpenAI-API for low computation requirements and storing them into
    a faiss vectorstore.

    Args:
        text_chunks (list): List of Strings that will be embedded an stored
                            as vectors

    Returns:
        FAISS: faiss vectorstore containing the embedded data
    """
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore


def create_conversation_chain(vectorstore):
    """Sets up the LLM (OpenAI-API), creates langchain prompt and a Conversational-Retrieval-Chain with buffer
    memory.

    Args:
        vectorstore (FAISS): The data to generate responses from

    Returns:
        ConversationalRetrievalChain: the chain to communicate with the chatbot
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def query_user_question(question):
    """Queries the vectordata (semantic search) on the given input and generates the chatbot response based.

    Args:
        question (String): question regarding the stored data
    """

    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # load question img to streamlit (as bytes, because local files are unreachable)
            file_ = open("./data/question.png", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            user_temp = user_template.replace("{{MSG}}", message.content)
            user_temp = user_temp.replace(
                "{{IMG}}", f"<img src='data:image/gif;base64,{data_url}' alt='cat gif'>"
            )

            st.write(user_temp, unsafe_allow_html=True)
        else:
            # load bot img to streamlit
            file_ = open("./data/bot.png", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            bot_temp = bot_template.replace("{{MSG}}", message.content)
            bot_temp = bot_temp.replace(
                "{{IMG}}", f"<img src='data:image/gif;base64,{data_url}' alt='cat gif'>"
            )

            st.write(bot_temp, unsafe_allow_html=True)


def main():
    """Runs the application, building the GUI and handling activities"""
    load_dotenv()

    # Page Title
    st.set_page_config(page_title="Chat with me!", page_icon=":books:")

    # Add css to streamlit
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Configure the simple prototype GUI
    col1, col2 = st.columns([0.25, 0.75])
    img_niko = Image.open("data/full_bot.png")
    with col1:
        st.write(" ")
        st.image(img_niko)

    with col2:
        st.header("Chat with you PDFs!")
        st.subheader(
            "Upload your PDF files inside the side menu and ask some questions regarding it's content!"
        )
    user_question = st.text_input("Ask a question about your PDFs!")
    if user_question:
        try:
            query_user_question(user_question)
        except TypeError:
            st.write("Please Process some PDFs before asking your questions!")
            

    # Configure PDF loading menu
    with st.sidebar:
        st.subheader("Your PDF Files! :books:")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                # Prepare the PDF data and store it in the vector database
                raw_text = load_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = embed_text_chunks(text_chunks)

                # Generate Conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
