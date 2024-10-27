from langchain.prompts import PromptTemplate


def mistral_prompt() -> PromptTemplate: 
    """
    Returns a prompt for mistral instruct version for RAG. 
    """
    prompt = """[INST]
        Answer the following question based only on the provided context:

        <context> 
        {context}
        </context>

        Question: {question}
        [/INST] """

    return PromptTemplate(template=prompt, input_variables=["context", "question"])