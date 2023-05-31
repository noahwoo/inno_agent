from datasets import load_dataset
import json
import requests
from collections import defaultdict

from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from inference_api import YiyanInferenceApi, HfInferenceApi
import regex
import sys

from mpi4py import MPI

dataset_mapping_url = f"https://raw.githubusercontent.com/SJTU-LIT/ceval/main/subject_mapping.json"

llm = YiyanInferenceApi("yiyan-007", debug=False)
# llm = OpenAI(model_name="text-davinci-003")

def get_data_mapping(mapping_url):
    headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    response = requests.request("GET", mapping_url, headers=headers)
    return json.loads(response.content.decode('utf-8'))

def escape_format(text : str) -> str :
    return text.replace('{', '{{').replace('}', '}}')

def gen_by_fewshot_prompt(task_name, fewshot_set, test_set, output) :

    prompt_template = """问题：{question} \n选项\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n思考：{explanation}\n答案：{answer}\n\n"""

    example_prompt = PromptTemplate(
        input_variables = ['question', 'A', 'B', 'C', 'D', 'explanation', 'answer'],
        template = prompt_template
    )
    
    fewshot_examples = []
    for id in range(len(fewshot_set)) :
        fewshot_examples.append({'question': escape_format(fewshot_set[id]['question']), 
                                 'A': escape_format(fewshot_set[id]['A']),
                                 'B': escape_format(fewshot_set[id]['B']),
                                 'C': escape_format(fewshot_set[id]['C']),
                                 'D': escape_format(fewshot_set[id]['D']),
                                 'explanation': escape_format(fewshot_set[id]['explanation']), 
                                 'answer': fewshot_set[id]['answer']})

    example_selector = LengthBasedExampleSelector(
        examples = fewshot_examples,
        example_prompt = example_prompt, 
        max_length = 1024,
        get_text_length = lambda x: len(x) # tokens per Chinese character
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = f"请作为一个{task_name}科目的考生，参考示例回答单项选择题，示例如下：\n", 
        suffix = """下面是要回答的问题，可以先写出思考过程，然后给出ABCD的一个选项作为答案，确保用<答案：>作为答案的前缀\n
                    问题：{question}\n选项\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n思考：""",

        input_variables = ['question', 'A', 'B', 'C', 'D'], 
        example_separator = "\n\n"
    )

    correct, total = 0, 0
    for id in range(len(test_set)) :
        try :
            grounded_prompt = few_shot_prompt.format(question = escape_format(test_set[id]['question']),
                                                            A = escape_format(test_set[id]['A']),
                                                            B = escape_format(test_set[id]['B']),
                                                            C = escape_format(test_set[id]['C']),
                                                            D = escape_format(test_set[id]['D']))
            completion = llm(grounded_prompt)
            llm_answer = extract_answer(completion)
            answer = test_set[id]['answer']
            print(f"question: {task_name}-{id}, llm_answer={llm_answer}, answer={answer}, completion={repr(completion)}", 
                  file = output)
            total += 1
            if llm_answer == answer :
                correct += 1
        except Exception as e :
            print(f"Exception {e} in answer generation for question <{id}> in task <{task_name}>")

    return (correct, total)

def extract_answer(completion) :
    answer_prefixes = ['答案：', '答案为', '答案是', '故选',
                        '选项中最接近的是', '选项为', '答案应该是', 
                        '所以答案选']
    for answer_prefix in answer_prefixes :
        if answer_prefix in completion :
            pos = completion.find(answer_prefix) + len(answer_prefix)
            if pos < len(completion) :
                answer = completion[pos]
                if answer == '{' and pos + 1 < len(completion) :
                    answer = completion[pos + 1]
                if answer in ['A', 'B', 'C', 'D'] :
                    return answer
    return 'Err'

if __name__ == "__main__" :

    data_mapping = get_data_mapping(dataset_mapping_url)
    g_correct, g_total = 0.0, 0.0

    tested_subject = 0
    topK = len(data_mapping.keys())

    # distribute with MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output = f'yiyan-ceval-result-rank{rank}.log'
    with open(output, 'w') as f :
        for id, name in enumerate(data_mapping.keys()) :
            if id % comm.size == rank : # process a sub-task
                dataset=load_dataset(r"ceval/ceval-exam",name=name)
                # print(dataset['dev'][0])
                (correct, total) = gen_by_fewshot_prompt(data_mapping[name][1], dataset['dev'], dataset['val'], f)
                g_correct += correct
                g_total += total
                print(f"Task <{name}>: correct={correct}, total={total}", file=f)
                print(f"Global accuracy so far: {g_correct/g_total:.4f}(g_correct={g_correct}, g_total={g_total})", file=f)

                tested_subject += 1
                if tested_subject >= topK :
                    break