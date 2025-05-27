
#Mounting the GDrive
from google.colab import drive
drive.mount('/content/drive')

!pip -q install llama-index
!pip -q install llama-index-embeddings-huggingface
!pip -q install peft
!pip -q install auto-gptq
!pip -q install optimum
!pip -q install bitsandbytes

#Hugging Face Embedder
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# import embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# Load Data Source (gDrive)
documents = SimpleDirectoryReader("/content/drive/MyDrive/RAG").load_data()

# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)

# set number of chunks to retreive
top_k = 3

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

# filter out retrieved chunks that are not similar enough
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# query documents
query = "what is the core feature of RAG models?"
response = query_engine.query(query)

# reformat response
context = "Context:\n"
for i in range(top_k):
    context = context + response.source_nodes[i].text + "\n\n"

print(context)

comment = query

# Load LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# prompt (no context)

prompt_template = lambda comment: f"""

Please respond to the following comment in a user friendly, conversational manner

Question: {comment}

"""

# prompt no context
prompt = prompt_template(comment)
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"].to("cuda"),
    attention_mask=inputs["attention_mask"].to("cuda"),
    max_new_tokens=280,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text.replace(prompt.strip(), "").strip()
print("Normal LLM Output:\n" , response)

# prompt (with context)

prompt_template_w_context = lambda context, comment: f"""
{context}
Please respond to the following comment in a user friendly, conversational manner. Use the context above if it is helpful.

Question: {comment}

"""

# prompt with context
prompt = prompt_template_w_context(context, comment)
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"].to("cuda"),
    attention_mask=inputs["attention_mask"].to("cuda"),
    max_new_tokens=280,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text.replace(prompt.strip(), "").strip()
print("RAG-Powered Output:\n" , response)

# Evaluation Prompt
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def evaluate_with_llm(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda"),
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return result

def query_and_validate(expected_response: str, actual_response: str) -> bool:
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=actual_response
    )

    print("\n=== Evaluation Prompt ===")
    print(prompt)

    evaluation_result = evaluate_with_llm(prompt)

    print("\n=== Evaluation Result ===")
    #print(evaluation_result.strip().splitlines()[-1].strip().lower())
    print(evaluation_result)


    if "true" in evaluation_result:
        return True
    elif "false" in evaluation_result:
        return False
    else:
        raise ValueError("Invalid evaluation result. Expected 'true' or 'false'.")

# Evaluate the actual RAG response
query_and_validate(
    expected_response="retrieval mechanism",
    actual_response=response
)
