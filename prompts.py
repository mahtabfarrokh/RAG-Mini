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

        ## Question: {question}

        ## Anwer:
        [/INST] """

    return PromptTemplate(template=prompt, input_variables=["context", "question"])


def llama3_prompt_qa() -> PromptTemplate: 
    """
    Returns a prompt for llama3 instruct version for RAG Q&A. 
    """
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Answer the following question only based on the provided context:
        <|eot_id|><|start_header_id|>user<|end_header_id|>
   
        ## Context: {context}

        ## Question: {question}

        ## Anwer: 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return PromptTemplate(template=prompt, input_variables=["context", "question"])

def llama_chat_prompt_qa() -> PromptTemplate: 
    """
    Returns a prompt for llama chat version for RAG Q&A. 
    """
    prompt = """Answer the following question only based on the provided context:

        ## Context: {context}

        ## Question: {question}

        ## Anwer: """

    return PromptTemplate(template=prompt, input_variables=["context", "question"])


def load_prompt(model_name: str) -> PromptTemplate: 
    """
    Returns the correct prompt for the model name.
    """
    # todo add llama 2 prompt and mixtral prompt
    if model_name.lower().find("mistral") >=0 or model_name.lower().find("mixtral") >=0:
        return mistral_prompt_qa()
    elif model_name.lower().find("llama-3") >=0 :
        return llama3_prompt_qa()
    elif model_name.lower().find("llama") >=0 and model_name.lower().find("chat") >=0:
        return llama_chat_prompt_qa()
    else:
        raise ValueError("Model name not supported")