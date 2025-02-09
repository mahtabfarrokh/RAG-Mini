from langchain.prompts import PromptTemplate


def mistral_prompt_qa() -> PromptTemplate: 
    """
    Returns a prompt for mistral instruct version for RAG Q&A. 
    """
    prompt = """[INST]
        Answer the following question only based on the provided context:

        <context> 
        {context}
        </context>

        Question: {question}

        Anwer:
        [/INST] """

    return PromptTemplate(template=prompt, input_variables=["context", "question"])


def llama3_prompt_qa() -> PromptTemplate: 
    """
    Returns a prompt for llama3 instruct version for RAG Q&A. 
    """
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Answer the following question only based on the provided context:
        <|eot_id|><|start_header_id|>user<|end_header_id|>
   
        {context}

        Question: {question}

        Anwer: 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return PromptTemplate(template=prompt, input_variables=["context", "question"])


