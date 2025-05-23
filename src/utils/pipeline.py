from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.dense import format_docs
from src.llm.prompts import Doctor_prompt
from src.llm.client import get_mistral_model

def build_chain(retriever):
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough(),
    })
    parser = StrOutputParser()
    model = get_mistral_model()
    main_chain = parallel_chain | Doctor_prompt | model | parser
    return main_chain
