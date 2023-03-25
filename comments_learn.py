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

print(os.environ['HOME'])
print(os.environ['OPENAI_API_KEY'])
print(os.environ['HUGGINGFACEHUB_API_TOKEN'])
def gpt_params_spend() :
    gpt_conf = {'GPT3-Small' : {'n_layers' : 12, 'd_model' : 768, 'n_head':12, 'd_head' : 64},
                'GPT3-Medium' : {'n_layers' : 24, 'd_model' : 1024, 'n_head':16, 'd_head' : 64},
                'GPT3-Large' : {'n_layers' : 24, 'd_model' : 1536, 'n_head':16, 'd_head' : 96},
                'GPT3-XL  ' : {'n_layers' : 24, 'd_model' : 2048, 'n_head':24, 'd_head' : 128},
                'GPT3-2.7B' : {'n_layers' : 32, 'd_model' : 2560, 'n_head':32, 'd_head' : 80},
                'GPT3-6.7B' : {'n_layers' : 32, 'd_model' : 4096, 'n_head':32, 'd_head' : 128},
                'GPT3-13B' : {'n_layers' : 40, 'd_model' : 5140, 'n_head':40, 'd_head' : 128},
                'GPT3-175B(GPT3)' : {'n_layers' : 96, 'd_model' : 12288, 'n_head':96, 'd_head' : 128}}
    n_vocab = 50257
    n_ctx = 2048
    for key, conf in gpt_conf.items() :
        pos_emb = n_ctx * conf['d_model']
        word_emb = n_vocab * conf['d_model'] 
        attn = 4 * conf['n_layers'] * conf['n_head'] * conf['d_head'] * conf['d_model']
        n_ff = 4*conf['d_model']
        ffn = conf['n_layers'] * (2*n_ff * conf['d_model'] + n_ff + conf['d_model'])
        layer_norm = (conf['n_layers'] * 2 + 1)* conf['d_model']
        total = pos_emb + word_emb + attn + ffn + layer_norm
        print(("{11}\ttotal:{10:,d}\tpos_emb:{0:,d}({1:.2f}%)\tword_emb:{2:,d}({3:.2f}%)"
            + "\tattn:{4:,d}({5:.2f}%)\tffn:{6:,d}({7:.2f}%)\tlayer_norm:{8:,d}({9:.2f}%)").format(
            pos_emb, float(pos_emb)/total * 100, 
            word_emb, float(word_emb)/total * 100, 
            attn, float(attn)/total * 100, 
            ffn, float(ffn)/total * 100, 
            layer_norm, float(layer_norm)/total, total, key))

def gpt_computation_spend() : 
    # TODO: calculate tflops of one round forward pass
    pass

# LLM APIs
def hf_hosted_inference_http(payload : str, model = 'gpt2') :
    API_URL = "https://api-inference.huggingface.co/models/{0}".format(model)
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def hf_hosted_inference_api(payload : str, model = 'gpt2') :
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
    inference = InferenceApi(repo_id=model, token=API_TOKEN)
    return inference(inputs=payload)

def yiyan_token() :
    yiyan_api_key=os.environ['YIYAN_API_KEY']
    yiyan_sec_key=os.environ['YIYAN_SEC_KEY']
    # api-token url
    TOKEN_URL = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={yiyan_api_key}&client_secret={yiyan_sec_key}"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # get token
    response = requests.request("POST", TOKEN_URL, headers=headers, data = '')
    response = json.loads(response.content.decode('utf-8'))
    token_fn = 'access_token'
    if token_fn in response :
        return response[token_fn]
    return None

def yiyan_inference_api(payload : str, model = 'yiyan') :
    access_token = yiyan_token()
    assert access_token is not None

    # augument payload for single round chat
    messages = list()
    messages.append({"role" : "user", "content" : payload})
    aug_payload = json.dumps({"messages" : messages})
    # print(aug_payload)

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    API_URL = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/agile/chat/completions?access_token={access_token}"
    response = requests.request("POST", API_URL, headers=headers, data = aug_payload)
    response = json.loads(response.content.decode('utf-8'))
    return response['result']


def extract_author_reply(input_stream) :
    # read video comments line by line

    authid_prev = None
    auth_reply_buffer = []

    for line in input_stream:
        video_comments = json.loads(line)
        title = video_comments['title']
        authid = video_comments['mthid']
        for comment in video_comments['comment'] :
            content = comment['comment_content']
            if 'reply_list' in comment :
                for reply in comment['reply_list'] :
                    if reply['is_author'] == '1' :
                        # print("{0}\t#{1}\t#{2}\t#{3}".format(authid, title, content, reply['content']))
                        auth_reply_buffer.append({"post":content, "reply":reply["content"]})
        # prompt for one author
        if authid_prev != None and authid != authid_prev : 
            prompt_one_author(authid_prev, auth_reply_buffer)
            auth_reply_buffer = []

        authid_prev = authid
    # the last author
    if len(auth_reply_buffer) > 0 :
        prompt_one_author(authid_prev, auth_reply_buffer)
        auth_reply_buffer = []

def prompt_one_author(authid : str, examples : List[Dict[str, str]], few_shot_ratio = 0.8, min_reply = 4, max_reply = 16) :

    example_num = len(examples)
    few_shot_num = int(example_num * few_shot_ratio)

    if few_shot_num < min_reply :
        return
    # cut to max_reply
    if few_shot_num > max_reply :
        few_shot_num = max_reply
    
    # print(examples)
    # llm to use
    llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)

    # ernie
    # tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-xbase-zh")
    # llm = ErnieForQuestionAnswering.from_pretrained("nghuyong/ernie-3.0-xbase-zh")

    # hugging face genernal
    # llm = HuggingFaceHub(repo_id="gpt2-large")

    # hugging face T5 hosted inference
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-T5-xxl")
    # llm = T5ForConditionalGeneration.from_pretrained("google/flan-T5-xxl")

    # Few-shot learning template
    format_template = """
    评论：{post}
    回复：{reply}\n
    """
    
    example_prompt = PromptTemplate(
        input_variables = ['post', 'reply'],
        template = format_template
    )
    
    example_selector = LengthBasedExampleSelector(
        examples = examples[:few_shot_num],
        example_prompt = example_prompt, 
        max_length = 3000, #FIXME: hard code max length for text-davinci-003, 4096
        get_text_length = lambda x: 2*len(x) # 2 tokens per Chinese character
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = "作为一个内容创作者，请给用户对你发布内容的评论做出回复。\n", 
        suffix = "评论：{post}\n回复：",
        input_variables = ['post'], 
        example_separator = "\n"
    )

    for idx, example in enumerate(examples) :
        if idx < few_shot_num :
            print("Author: {0}\tPost: {1}\nAuthor-Reply: {2}\n".format(
                authid,
                example['post'], 
                example['reply']))
        else :
            # print(few_shot_prompt.format(post=example['post']))
            # try :
            #    davinci_reply = llm(few_shot_prompt.format(post=example['post']))
            # except Exception as e:
            #    davinci_reply = "API Error: " + e
            
            try :
                yiyan_reply = yiyan_inference_api(few_shot_prompt.format(post=example['post']))
            except Exception as e:
                yiyan_reply = "API Error: " + e

            print(f"Author: {authid}\tPost: {example['post']}\nAuthor reply: {example['reply']}\nYIYAN replay: {yiyan_reply}")
            print('-'*30)
    
if __name__ == '__main__' :
    # extract reply author from json record
    extract_author_reply(sys.stdin)

    # gpt params counting
    # gpt_params_spend()