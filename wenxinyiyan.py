from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, root_validator
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

def _create_retry_decorator(llm: WxYiyan) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def generate_with_retry(llm: WxYiyan, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _generate_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.call(**_kwargs)
        if 'error_code' not in resp['output']:
            return resp
        else:
            raise HTTPError(
                f"HTTP error occurred: status_code: {resp['output']['error_code']} \n "
                f"code: {resp['output']['error_code']} \n message: {resp['output']['error_msg']}"
            )

    return _generate_with_retry(**kwargs)


def stream_generate_with_retry(llm: WxYiyan, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _stream_generate_with_retry(**_kwargs: Any) -> Any:
        stream_resps = []
        resps = llm.client.call(**_kwargs)
        for resp in resps:
            if 'error_code' not in resp['output']:
                stream_resps.append(resp)
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp['output']['error_code']} \n "
                    f"code: {resp['output']['error_code']} \n message: {resp['output']['error_msg']}"
                )
        return stream_resps

    return _stream_generate_with_retry(**kwargs)


class WxYiyan(LLM):
    """Wenxinyiyan large language models.

    To use, you should have the
    environment variable ``YIYAN_API_KEY`` set with your API key and ``YIYAN_SEC_KEY`` set with your security key

    Example:
        .. code-block:: python

            from langchain.llms import WxYiyan
            yiyan = WxYiyan()
    """

    client: Any  #: :meta private:
    model_name: str = "ERNIE-Bot"
    """Model name to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system : Optional[str] = None
    """system role of model"""

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    temperature: float = 0.2
    """Temerature for soft-max"""

    penalty_score : float = 1.0
    """penalty score for repeated tokens"""

    yiyan_api_key: Optional[str] = None
    """Yiyan api key provided by Qianfan Baidu Cloud."""

    yiyan_sec_key: Optional[str] = None
    """Yiyan sec key provied by Qianfan Baidu Cloud."""
    
    n: int = 1
    """How many completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "wenxinyiyan"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        # check environment variables
        get_from_dict_or_env(values, "yiyan_api_key", "YIYAN_API_KEY")
        get_from_dict_or_env(values, "yiyan_sec_key", "YIYAN_SEC_KEY")

        try :
            values['client'] = YiyanClient()
        except :
            raise ImportError(
                "Can not initialize YiyanClient."
                "Please check your environment settings."
            )
        return values
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling WxYiyan API."""
        normal_params = {
            "top_p": self.top_p,
            "temperature" : self.temperature,
            "penalty_score" : self.penalty_score,
            "system" : self.system
        }

        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to WxYiyan's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = yiyan("Tell me a joke.")
        """
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }

        completion = generate_with_retry(
            self,
            prompt=prompt,
            **params,
        )
        return completion["output"]["text"]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            params["stream"] = True
            for stream_resp in stream_generate_with_retry(
                self, prompt=prompts[0], **params
            ):
                generations.append(
                    [
                        Generation(
                            text=stream_resp["output"]["text"],
                            generation_info=dict(
                                finish_reason=stream_resp["output"]["finish_reason"],
                            ),
                        )
                    ]
                )
        else:
            for prompt in prompts:
                completion = generate_with_retry(
                    self,
                    prompt=prompt,
                    **params,
                )
                generations.append(
                    [
                        Generation(
                            text=completion["output"]["text"],
                            generation_info=dict(
                                finish_reason=completion["output"]["finish_reason"],
                            ),
                        )
                    ]
                )
        return LLMResult(generations=generations)

#############################
# Yiyan hosted inference APIs
#############################
import os
import requests
import json
from collections import defaultdict

HOSTED_LLMS = { 
    "ERNIE-Bot" : "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions", 
    "ERNIE-Bot-turbo" : "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant", 
    "BLOOMZ-7B" : "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1"
}

class YiyanClient():
    def __init__(self, debug = False):
        self.debug = debug
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
        assert False, "Failed to access token" 
    
    def yiyan_inference_api(self, payload : str, model_name : str, **kwargs) :
        
        # augument payload for single round chat
        messages = list()
        messages.append({"role" : "user", "content" : payload})

        # add ext kwargs
        # aug_payload = json.dumps({"messages" : messages})
        data = defaultdict(dict)
        if 'system' in kwargs and kwargs['system'] is not None:
            data['system'] = kwargs['system']

        data["messages"] = messages
        for key, val in kwargs.items() :
            if key not in ['prompt', 'model', 'system'] :
                data[key] = val
        aug_payload = json.dumps(data)

        if self.debug :
            print(json.dumps(data, ensure_ascii=False))

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        API_URL = f"{HOSTED_LLMS[model_name]}?access_token={self.token}"
        
        response = requests.request("POST", API_URL, headers=headers, data = aug_payload)
        response = json.loads(response.content.decode('utf-8'))
        if self.debug :
            print(response)
        
        resp_ret = defaultdict(dict)

        if 'result' in response :
            resp_ret['output']['text'] = response['result']
            resp_ret['output']['finish_reason'] = 'Success'
        if 'error_code' in response :
            resp_ret['output']['error_code'] = response['error_code']
            resp_ret['output']['error_msg'] = response['error_msg']
            resp_ret['output']['finish_reason'] = 'Error'
            if self.debug :
                print(f"Error code: {response['error_code']}")
                print(f"Error message: {response['error_msg']}")
        return resp_ret
        
    def call(self, **kwargs) :
        return self.yiyan_inference_api(kwargs['prompt'], kwargs['model'], **kwargs)
    
    def __call__(self, inputs, **kwargs):
        return self.yiyan_inference_api(inputs, **kwargs)