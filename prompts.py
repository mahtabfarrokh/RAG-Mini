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


def improved_judge_prompt_llama3() -> PromptTemplate: 
    """
    Returns a prompt for the improved judge version for RAG Q&A. 
    Propmt taken from: https://huggingface.co/learn/cookbook/en/llm_judge
    """
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You will be given a user_question and system_answer couple.
        Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
        Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

        Here is the scale you should use to build your answer:
        1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
        2: The system_answer is mostly not helpful: misses some key aspects of the question
        3: The system_answer is mostly helpful: provides support, but still could be improved
        4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

        Provide your feedback as follows:

        Total rating: (your rating, as a number between 1 and 4)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here are the question and answer.

        <|eot_id|><|start_header_id|>user<|end_header_id|>

        Question: {question}
        Answer: {answer}

        If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.

        Total rating:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return PromptTemplate(template=prompt, input_variables=["answer", "question"])


def improved_judge_prompt(model_name:str) -> PromptTemplate: 
    """
    Returns the correct prompt for the model name.
    """
    if model_name.lower().find("llama-3") >=0 :
        return improved_judge_prompt_llama3()
    else:
        raise ValueError("Model name not supported")