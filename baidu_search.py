import logging
import re
from typing import List, Optional, Dict
from collections import defaultdict

from pydantic import BaseModel, Field

import requests
from pydantic import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.llms import LlamaCpp
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
import hashlib
import os

logger = logging.getLogger(__name__)

### Search API Wrapper

class BaiduSearchAPIWrapper(BaseModel):
    """Wrapper for Bing Search API."""
    k : int = 10

    baidu_search_api_key: Optional[str] = None
    """Search api key provided by Qianfan Baidu Cloud."""

    baidu_search_sec_key: Optional[str] = None
    """Search sec key provied by Qianfan Baidu Cloud."""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        # TODO: access environment for API key
        # check environment variables
        get_from_dict_or_env(values, "baidu_search_api_key", "BAIDU_SEARCH_API_KEY")
        get_from_dict_or_env(values, "baidu_search_sec_key", "BAIDU_SEARCH_SEC_KEY")

        return values

    def run(self, query: str) -> str:
        """Run query through BaiduSearch and parse result."""
        snippets = []
        results = self._baidu_search_results(query, count=self.k)
        if len(results) == 0:
            return "No good Bing Search Result was found"
        for result in results:
            snippets.append(result["snippet"] if "snippet" in result else "")

        return " ".join(snippets)


    def results(self, query: str, num_results: int = 10, page_num: int = 0) -> List[Dict]:
        """Run query through BaiduSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._baidu_search_results(query, page_num=page_num)
        if len(results) == 0:
            return [{"Result": "No good Baidu Search Result was found"}]
        
        for result in results:
            metadata_result = {
                "snippet": result["snippet"],
                "title": result["title"],
                "link": result["link"],
                "aladdinTemplateName": result["aladdinTemplateName"]
            }
            metadata_results.append(metadata_result)

        return metadata_results

    def _get_local_ip(self) :
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            sock.connect(('192.255.255.255', 1))
            IP = sock.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            sock.close()
        return IP
    
    def _baidu_search_results(self, search_term: str, page_num: int = 0) -> List[dict]:
        url = "http://m.baidu.com/s?"
        data = {
            "word":search_term,
            "from": os.environ['BAIDU_SEARCH_API_KEY'],
            "clientip":self._get_local_ip(),
            "cip":self._get_local_ip(),
            "tn":"apijson",
            "pn" : page_num
        }

        key  = os.environ['BAIDU_SEARCH_SEC_KEY']
        auth = "{ak}{word}{clientip}{key}".format(ak=data["from"], word=data["word"], clientip=data["clientip"], key=key)
        auth = hashlib.md5(auth.encode("utf-8"))
        auth = auth.hexdigest()
        auth = auth[7]+auth[3]+auth[17]+auth[13]+auth[1]+auth[21]
        data["auth"] = auth
       
        response = requests.get(url, params=data)
        assert response.status_code == 200 , "Request to Baidu search failed : status code: {}".format(response.status_code)

        print(f"content: {response.content}")
        # req.raw
        data = response.json()

        # parse data and construct the search result
        results = []
        for result in data['root']['result']:
            # filter aladdin by now
            if 'aladdinTemplateName' in result and result['aladdinTemplateName'] not in ['recommend_list'] :
                r = defaultdict(str)
                r['aladdinTemplateName'] = result['aladdinTemplateName']
                r['link'] = result['result']['url'] if 'url' in result['result'] else None
                r['title'] = result['result']['title'] if 'title' in result['result'] else None
                r['snippet'] = result['result']['abstraction'] if 'abstraction' in result['result'] else None
                results.append(r)
        return results

### 
class SearchQueries(BaseModel):
    """Search queries to run to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Baidu"
    )


DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Baidu search \
results. \n <</SYS>> \n\n [INST] Generate THREE Baidu search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Baidu search \
results. Generate THREE Baidu search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)


class LineList(BaseModel):
    """List of questions."""
    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)


class WebResearchRetriever(BaseRetriever):
    """Retriever for web research based on the Google Search API."""

    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    llm_chain: LLMChain
    search: BaiduSearchAPIWrapper = Field(..., description="Baidu Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: RecursiveCharacterTextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )

    @classmethod
    def from_llm(
        cls,
        vectorstore: VectorStore,
        llm: BaseLLM,
        search: BaiduSearchAPIWrapper,
        prompt: Optional[BasePromptTemplate] = None,
        num_search_results: int = 1,
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        ),
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: BaiduSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks

        Returns:
            WebResearchRetriever
        """

        if not prompt:
            QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
                default_prompt=DEFAULT_SEARCH_PROMPT,
                conditionals=[
                    (lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)
                ],
            )
            prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

        # Use chat model prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=QuestionListOutputParser(),
        )

        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
        )

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Baidu) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_serch_results pages per Baidu search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Search Baidu for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """

        # Get search questions
        logger.info("Generating questions for Baidu Search ...")
        result = self.llm_chain({"question": query})
        logger.info(f"Questions for Baidu Search (raw): {result}")
        questions = getattr(result["text"], "lines", [])
        logger.info(f"Questions for Baidu Search: {questions}")

        # Get urls
        logger.info("Searching for relevat urls ...")
        urls_to_look = []
        for query in questions:
            # Baidu search
            search_results = self.search_tool(query, self.num_search_results)
            logger.info("Searching for relevat urls ...")
            logger.info(f"Search results: {search_results}")
            for res in search_results:
                urls_to_look.append(res["link"])

        # Relevant urls
        urls = set(urls_to_look)

        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))

        logger.info(f"New URLs to load: {new_urls}")
        # Load, split, and add new urls to vectorstore
        if new_urls:
            loader = AsyncHtmlLoader(new_urls)
            html2text = Html2TextTransformer()
            logger.info("Indexing new urls...")
            docs = loader.load()
            docs = list(html2text.transform_documents(docs))
            docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(docs)
            self.url_database.extend(new_urls)

        # Search for relevant splits
        # TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
        docs = []
        for query in questions:
            docs.extend(self.vectorstore.similarity_search(query))

        # Get unique docs
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        }
        unique_documents = list(unique_documents_dict.values())
        return unique_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError
