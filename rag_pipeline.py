from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from prompts import load_prompt
from data_loader import load_passages, load_qa


class rag_pipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hf = self.load_pipeline()
        self.retriever = load_passages()
        self.qa_data = load_qa()
        self.batch_size = 1
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
        prompt = load_prompt(self.model_name)
        self.prompt_chain = prompt | self.hf
        output = []
        for i in range(0, len(qa_data), self.batch_size):
            question = qa_data["question"][i:i+batch_size]
            input_query = []
            for i in range(self.batch_size):
                res = self.retriever.get_relevant_documents(question[i])
                context = "\n".join([doc.page_content for doc in res])
                input_query.append({"context": context, "question": question[i]})

            res = self.prompt_chain.batch(input_query)
            for i in range(self.batch_size):
                output.append(self.output_processing(res[i]))

        self.qa_data["answer"] = output
        self.qa_data.to_csv(self.output_folder + "output.csv", index=False)

        return self.output_folder + "output.csv"
        

    def output_processing(self, llm_output -> str) -> str:
        """
        Process the output of the llm
        """
        # Todo: Add processing steps
        print("-----------------")
        print(llm_output)
        return llm_output                 


if __name__ == "__main__":
    model_name = "Llama-2-7B-Chat-AWQ"
    rag = rag_pipeline(model_name)
    rag.run_pipeline()