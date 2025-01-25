from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline



def load_pipeline(modl_name):
    model = AutoModelForCausalLM.from_pretrained("llms/" + modl_name,
                                             device_map="auto",
                                             trust_remote_code=False)
    
    tokenizer = AutoTokenizer.from_pretrained("llms/" + modl_name,
                                               use_fast=True)

    pipeline_llm = pipeline("text-generation", 
                            model = model, 
                            tokenizer=tokenizer,
                            max_new_tokens = 300, 
                            device_map ="auto")

    hf= HuggingFacePipeline(pipeline= pipeline_llm, model_kwargs={"temperature": 0})

    return hf, model, tokenizer