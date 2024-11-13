from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import chainlit as cl

custom_prompt_template = """You are trained on UOBS dataset so give me the information about only Univesity of Blatistan skardu ok.if you do not know just only say I do not know sorry.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for generating responses
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Loading the model
def load_llm():
    # Load the locally downloaded GGUF model here
    llm = CTransformers(
        model="C:/Users/PMLS/.cache/huggingface/hub/models--hassu619--Llama-2-7b-chat-finetune_UOBS-GGUF/snapshots/9c4480a711c04c679afdf8ad6917eb8f297f0109/llama-2-7b-chat-finetune_uobs.Q4_K_M.ggml",  # GGUF model format
        model_type="llama",
        max_new_tokens=128,
        temperature=0.5
    )
    return llm

# Function to generate responses using the model directly
def generate_response(query, llm, prompt):
    formatted_prompt = prompt.format(context="", question=query)
    response = llm(formatted_prompt)
    return response

# QA Model Function
def qa_bot():
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return llm, qa_prompt

# Output function
def final_result(query):
    llm, qa_prompt = qa_bot()
    response = generate_response(query, llm, qa_prompt)
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    llm, qa_prompt = qa_bot()
    cl.user_session.set("llm", llm)
    cl.user_session.set("qa_prompt", qa_prompt)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the University of Baltistan. What is your query?"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("llm")
    qa_prompt = cl.user_session.get("qa_prompt")

    response = generate_response(message.content, llm, qa_prompt)
    
    await cl.Message(content=response).send()
