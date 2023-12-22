import openai  # openai==0.28.1
import asyncio
from typing import List


class Embeddings:
    def __init__(self):
        self.client = openai.Embedding()

    def embed_documents(self, texts: List[str], api_base=None, max_concurrent_requests=100) -> List[List[float]]:
        """
        Embed a list of documents.
        :param texts: A list of documents to embed.
        :param api_base: The base URL of the OpenAI API.
        :param max_concurrent_requests: The maximum number of concurrent requests to OpenAI.
        :return: A list of document embeddings.
        """

        all_embedding = []
        for i in range(0, len(texts), max_concurrent_requests * 32):
            results = self._embed_documents(texts[i:i + max_concurrent_requests * 32], api_base=api_base)
            all_embedding.extend(results)
        return all_embedding

    def _embed_documents(self, texts: List[str], api_base=None) -> List[List[float]]:
        openai.api_key = 'xxxx'
        openai.api_base = api_base
        all_embedding = []
        result = asyncio.run(self.generate_concurrently(texts, self.client))
        for response in result:
            all_embedding.extend([e['embedding'] for e in response['data']])

        return all_embedding

    def embed_query(self, text: str, api_base=None) -> List[float]:
        openai.api_key = 'xxxx'
        openai.api_base = api_base
        return self.client.create(input=text, model="text2vec-large-chinese")['data'][0]['embedding']

    @staticmethod
    async def generate_concurrently(texts, client):
        tasks = []
        for i in range(0, len(texts), 32):
            tasks.append(client.acreate(input=texts[i:i + 32], model='m3e'))
        return await asyncio.gather(*tasks)


if __name__ == '__main__':
    import time

    start = time.time()
    emb = Embeddings()
    result = emb.embed_documents(texts=['text'],api_base="http://ip:port")

    print(time.time() - start)

    print(len(result))
