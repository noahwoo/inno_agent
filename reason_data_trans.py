import sys
import os
import json
import requests

from typing import List, Dict, Callable
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from langchain.llms import HuggingFaceHub
from transformers import BertTokenizer, ErnieModel, ErnieForQuestionAnswering
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub.inference_api import InferenceApi

print(os.environ['OPENAI_API_KEY'])
print(os.environ['HUGGINGFACEHUB_API_TOKEN'])

