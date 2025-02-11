from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os 

from prompts import load_prompt
from data_loader import load_passages, load_qa


class rag_pipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.retriever = load_passages()
        self.qa_data = load_qa()
        self.hf = self.load_pipeline()
        self.batch_size = 8
        self.output_folder = "output/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


    def load_pipeline(self) -> HuggingFacePipeline:
        """
        Load the pipeline for the model
        """
        model = AutoModelForCausalLM.from_pretrained("llms/" + self.model_name,
                                                device_map="auto",
                                                trust_remote_code=False)
        
        tokenizer = AutoTokenizer.from_pretrained("llms/" + self.model_name,
                                                use_fast=True)

        pipeline_llm = pipeline("text-generation", 
                                model = model, 
                                tokenizer=tokenizer,
                                max_new_tokens = 300, 
                                device_map ="auto")

        hf= HuggingFacePipeline(pipeline= pipeline_llm, model_kwargs={"temperature": 0})

        return hf
    
    def run_pipeline(self) -> str:
        print("Running the pipeline...")
        prompt = load_prompt(self.model_name)
        self.prompt_chain = prompt | self.hf
        output = [] 
        print(len(self.qa_data))
        for i in range(0, len(self.qa_data), self.batch_size):
            print(i)
            question = self.qa_data["question"][i:i+self.batch_size].to_list()
            input_query = []
            for j in range(len(question)):
                res = self.retriever.similarity_search(question[j], k=2)
                context = "\n".join([doc.page_content for doc in res])
                input_query.append({"context": context, "question": question[j]})

            res = self.prompt_chain.batch(input_query)
            for j in range(len(res)):
                output.append(self.output_processing(res[j]))
       
        self.qa_data["answer"] = output
        save_path = self.output_folder + "output_" + self.model_name + ".csv"
        self.qa_data.to_csv(save_path, index=False)
        print("Success: saved the llm answers.")
        return save_path
        

    def output_processing(self, llm_output: str) -> str:
        """
        Process the output of the llm
        """
        # Todo: Add processing steps
        idx = llm_output.find("## Anwer:")
        if  idx >=0: 
            return llm_output[idx + len("## Anwer:"):].split('Answer the following question only based on the provided context:')[0]
        else: 
            return "Can't answer this question."


if __name__ == "__main__":
    model_name = "Mistral-7B-Instruct-v0.2"
    rag = rag_pipeline(model_name)
    rag.run_pipeline()