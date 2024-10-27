# RAG Mini Project

**Warning: This repo is under construction! The code base is not complete yet!**

This repository contains a RAG (Retrieval-Augmented Generation) toy example designed to test whether large language models (LLMs) are biased towards their own family.

It has become a common practice to use a larger LLM to evaluate the responses of another LLM. In this project, the question we're exploring is: Even when asking questions such as "Is the response correct?" or "Is it faithful?", will the judge LLM show bias towards its own family?

My hypothesis is that it will! But let's find out if that holds true.

To load the embedding model:

```bash
git clone https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
```

I will be using qunatized versions of Llama and Mistral Family, you can download them by running:

```bash
mkdir llms
git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ
git clone https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
git clone https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ
git clone https://huggingface.co/casperhansen/llama-3-8b-instruct-awq
```
