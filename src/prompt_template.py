# This file was originally part of PromtEngineer/localGPT and has been modified.
#
# The original code is licensed under the MIT License, a copy of which
# is available in the LICENSES/ directory.
#
# All modifications are licensed under the BSD-3-Clause License.

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

doxygen_prompt = """You are an expert code documentation specialist. You will use the provided context from technical literature to write high-quality documentation comments for code.
When generating comments:
- First understand how the specific function/method fits into the larger software system described in the context
- Briefly explain different parts of the code
- Connect the function's purpose to the broader architectural concepts when relevant
- Use consistent terminology from the literature/documentation
- Do not create full form of abbreviations unless it is derived from the context
- Be scientific and technical in your descriptions
- Format the comments appropriately for the language being documented
- Focus only on writing proper documentation comments - no explanations or meta-commentary is needed
If you cannot determine the function's purpose based on the provided context and code analysis, provide a basic comment based purely on the function signature and code."""

chatbot_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. 
Do not use any other information for answering user. Provide a concise and relevant answer to the question.
If you can not answer a user question based on the provided context then give no response.
**Always use triple backquotes to capture exact strings of C/C++ symbols or functions, variables, classes names from the source code.**
For example, if your response contains a function name foo, write it as ```foo```."""

#chatbot_prompt = """You are a helpful C++ code generation assistant, you will use the provided context to answer user questions.
#Read the given context before answering questions and think step by step. 
#Do not use any other information for answering user. Provide a concise and relevant answer to the question."""
# this is specific to Llama-2.
#system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
#Read the given context before answering questions and think step by step. If you can not answer a user question based on 
#the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""


class langchain_prompt_template:

    def __init__(self, promptTemplate_type=None, history=False, chatbot=False):
        self.promptTemplate_type = promptTemplate_type
        self.history = history
        self.chatbot = chatbot
        self.END_STR = ""

    def get_prompt_memory(self):

        if self.chatbot:
            system_prompt=chatbot_prompt
        else:
            system_prompt=doxygen_prompt

        if self.promptTemplate_type == "llama3":

            B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
            B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
            ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
            SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
            self.END_STRING = ASSISTANT_INST 
            if self.history:
                instruction = """
                Context: {history} \n {context}
                User: {question}"""

                prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
                prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
            else:
                instruction = """
                Context: {context}
                User: {question}"""

                prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
                prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        elif self.promptTemplate_type == "mistral":
            B_INST, E_INST = "<s>[INST] ", " [/INST]"
            self.END_STRING = E_INST 
            if self.history:
                prompt_template = (
                    B_INST
                    + system_prompt
                    + """
        
                Context: {history} \n {context}
                User: {question}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
            else:
                prompt_template = (
                    B_INST
                    + system_prompt
                    + """
                
                Context: {context}
                User: {question}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        else:
            print("\n* * * * * \n Err: Unknown model_type of PromptTemplate \n * * * * * \n")

        memory = ConversationBufferMemory(input_key="question", memory_key="history")

        print(f"Prompt template is of {self.promptTemplate_type} as: {prompt}")

        return (
            prompt,
            memory,
        )

