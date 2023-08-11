import sys
import os
import json
import optparse

from typing import List, Dict
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from nltk.translate.bleu_score import sentence_bleu

from inference_api import YiyanInferenceApi, HfInferenceApi
import jieba
from rouge_chinese import Rouge

print(os.environ['HOME'])
print(os.environ['OPENAI_API_KEY'])
print(os.environ['HUGGINGFACEHUB_API_TOKEN'])

LLM_VENDORS = {"openai" : {"llm" : OpenAI(model_name="text-davinci-003"), "context_size" : 2048, "prompt_in_response" : False}, 
               "yiyan" : {"llm" : YiyanInferenceApi("yiyan-007"), "context_size" : 1024, "prompt_in_response" : True},
               "hf" : {"llm" : HfInferenceApi("bigscience/bloomz"), "context_size" : 1024, "prompt_in_response" : True}}
rouge = Rouge()

def reformat_author_replay(input_stream, format:str, train_split:float = 0.7, output_file:str = "output") :
    # read video comments line by line
    authid_prev = None
    auth_reply_buffer = []

    records_buffer = []
    for line in input_stream:
        video_comments = json.loads(line)
        title = video_comments['title']
        authid = video_comments['mthid']
        for comment in video_comments['comment'] :
            content = comment['comment_content']
            if 'reply_list' in comment :
                for reply in comment['reply_list'] :
                    if reply['is_author'] == '1' :
                        auth_reply_buffer.append({"post":content, "reply":reply["content"], "title":title})
        # prompt for one author
        if authid_prev != None and authid != authid_prev : 
            records_buffer += reformat_one_author(authid_prev, auth_reply_buffer, format)
            auth_reply_buffer = []

        authid_prev = authid
    # the last author
    if len(auth_reply_buffer) > 0 :
        records_buffer += reformat_one_author(authid_prev, auth_reply_buffer, format)
        auth_reply_buffer = []

    total = len(records_buffer)
    train_size = int(total * train_split)

    with open(f"{output_file}.train.json", "w") as f :
        f.write(json.dumps(records_buffer[:train_size], ensure_ascii=False))

    with open(f"{output_file}.test.json", "w") as f :
        f.write(json.dumps(records_buffer[train_size:], ensure_ascii=False))

def reformat_one_author(authid, reply_buffer, format) :

    records = []

    if format == 'alpaca' :
        for pr in reply_buffer :
            records.append({"instruction":f"请回复用户针对<{pr['title']}>的评论", 
                            "input": f"评论:{pr['post']}", 
                            "output": f"回复:{pr['reply']}"})
    return records

def extract_author_reply(input_stream, llm_vendor : str) :
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
            prompt_one_author(authid_prev, auth_reply_buffer, llm_vendor)
            auth_reply_buffer = []

        authid_prev = authid
    # the last author
    if len(auth_reply_buffer) > 0 :
        prompt_one_author(authid_prev, auth_reply_buffer, llm_vendor)
        auth_reply_buffer = []

def prompt_one_author(authid : str, examples : List[Dict[str, str]], llm_vendor, few_shot_ratio = 0.8, min_reply = 4, max_reply = 16) :

    example_num = len(examples)
    few_shot_num = int(example_num * few_shot_ratio)

    if few_shot_num < min_reply :
        return
    # cut to max_reply
    if few_shot_num > max_reply :
        few_shot_num = max_reply
    
    # print(examples)
    # llm to use
    assert llm_vendor in LLM_VENDORS, f"No LLM vendor found for {llm_vendor}"
    llm = LLM_VENDORS[llm_vendor]['llm']
    context_size = LLM_VENDORS[llm_vendor]['context_size']
    context_size_buff = context_size // 10

    prompt_in_response = LLM_VENDORS[llm_vendor]['prompt_in_response']

    # Few-shot learning template
    format_template = """
    评论：{post}\n
    回复：{reply}\n
    """
    
    example_prompt = PromptTemplate(
        input_variables = ['post', 'reply'],
        template = format_template
    )
    
    example_selector = LengthBasedExampleSelector(
        examples = examples[:few_shot_num],
        example_prompt = example_prompt, 
        max_length = context_size - context_size_buff,
        get_text_length = lambda x: 2*len(x) # 2 tokens per Chinese character
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = "作为一个内容创作者，请给用户对你发表内容的评论做出回复。\n", 
        suffix = "评论：{post}\n回复：",
        input_variables = ['post'], 
        example_separator = "\n"
    )

    rouge_score, npred = 0.0, 0
    for idx, example in enumerate(examples) :
        if idx < few_shot_num :
            print("Author: {0}\tPost: {1}\nAuthor-Reply: {2}\n".format(
                authid,
                example['post'], 
                example['reply']))
        else : 
            try :
                prompt_instance = few_shot_prompt.format(post=example['post'])
                llm_reply = llm(prompt_instance)
                if prompt_in_response :
                    llm_reply = llm_reply[0]['generated_text'][len(prompt_instance):]
            except Exception as e:
                llm_reply = "API Error: " + e

            print(f"Author: {authid}\tPost: {example['post']}\nAuthor reply: {example['reply']}\nLLM({llm_vendor}) replay: {llm_reply}")

            rouge_score += rouge.get_scores(' '.join(jieba.cut(example['reply'],  cut_all=False)), 
                                            ' '.join(jieba.cut(llm_reply, cut_all=False)))[0]['rouge-l']['f']
            npred += 1
            print('-'*100)  
    print(f"ROUGE-l: {rouge_score/npred:.4f}")

if __name__ == '__main__' :

    parser = optparse.OptionParser()
    parser.add_option("-m", "--llm", dest="llm", default="hf", help="vendor of large language model")
    opts, args = parser.parse_args()

    # extract reply author from json record
    # extract_author_reply(sys.stdin, opts.llm)
    
    reformat_author_replay(sys.stdin, format="alpaca")

    # ROUGE-l f-score
    # BLOOM:  0.0994
    # OpenAI: 0.0773