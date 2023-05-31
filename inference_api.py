from huggingface_hub.inference_api import InferenceApi
import requests
import json
import os
import time

# Huggingface hosted inference APIs
class HfInferenceApi(InferenceApi) :
    def __init__(self, repo_id) :
        super().__init__(repo_id=repo_id)
        self.token = os.environ['HUGGINGFACEHUB_API_TOKEN']
        self.repo_id = repo_id

    def _hf_hosted_inference_http(self, payload : str) :
        API_URL = "https://api-inference.huggingface.co/models/{0}".format(self.repo_id)
        API_TOKEN = self.token
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))
    
    def _hf_hosted_inference_api(self, payload : str) :
        API_TOKEN = self.token
        inference = InferenceApi(repo_id=self.repo_id, token=API_TOKEN)
        return inference(inputs=payload,
                         params={'max_length':50, 
                                 'min_length':5, 
                                 'do_sample':True, 
                                 'top_k':50, 
                                 'top_p':0.95, 
                                 'temperate':0.9, 
                                 'num_return_sequences':1})
    
    def __call__(self, inputs) :
        return self._hf_hosted_inference_api(inputs)

# Yiyan hosted inference APIs
class YiyanInferenceApi():
    def __init__(self, repo_id, debug = False, max_retries = 5):
        self.repo_id = repo_id
        self.debug = debug
        self.max_retries = max_retries
        self.token = self._yiyan_token()

    def _yiyan_token(self) :
        yiyan_api_key=os.environ['YIYAN_API_KEY']
        yiyan_sec_key=os.environ['YIYAN_SEC_KEY']
        if self.debug :
            print(f"Yiyan api key: {yiyan_api_key}, Yiyan sec key: {yiyan_sec_key}")
        
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
    
    def yiyan_inference_api(self, payload : str) :
        access_token = self.token
        assert access_token is not None, f"No access token found. Please check your environment variables"

        # augument payload for single round chat
        messages = list()
        messages.append({"role" : "user", "content" : payload})
        aug_payload = json.dumps({"messages" : messages})
        # print(aug_payload)

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        # API_URL = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/agile/chat/completions?access_token={access_token}"
        API_URL = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={access_token}"

        retry = 0
        while retry < self.max_retries :
            response = requests.request("POST", API_URL, headers=headers, data = aug_payload)
            response = json.loads(response.content.decode('utf-8'))
            if self.debug :
               print(response)
            if 'result' in response :
                return response['result']
            time.sleep(1)
            retry += 1
        assert False, f"Yiyan inference API failed"

    def __call__(self, inputs):
        return self.yiyan_inference_api(inputs)