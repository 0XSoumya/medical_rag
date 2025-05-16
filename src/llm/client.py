from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

def get_mistral_model():
    return ChatMistralAI(model='mistral-small')

def get_mistral_embeddings():
    return MistralAIEmbeddings()
