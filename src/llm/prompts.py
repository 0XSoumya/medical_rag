from langchain_core.prompts import PromptTemplate

Doctor_prompt = PromptTemplate(
    template=(
        "You are a skilled doctor, answer the question based only on provided context, "
        "if you cannot then say you don't know\n"
        "context = {context}\n"
        "question = {question}"
    ),
    input_variables=["context", "question"],
)
