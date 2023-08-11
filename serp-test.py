from serpapi.google_search import GoogleSearch
import os

os.environ['SERPAPI_API_KEY'] = '5ca1f5544dd75a93725b3017b9870c502c364be13f2610a3c2b83cdd5126487c'
print(f"SerpAPI Key: {os.environ['SERPAPI_API_KEY']}")
params = {
  "engine": "google_scholar",
  "q": "deep learning",
  "api_key": os.environ['SERPAPI_API_KEY']
}

search = GoogleSearch(params)
results = search.get_dict()
organic_results = results["organic_results"]
for result in organic_results :
    print(f"#{result['position']}: {result['title']}")
    print(f"Link: {result['link']}")
