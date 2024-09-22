# https://github.com/langchain-ai/langchain/issues/8623

import pandas as pd

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from typing import List
from pydantic import Field

class ClimateQARetriever(BaseRetriever):
    vectorstore:VectorStore
    sources:list = ["IPCC","IPBES","IPOS"]
    reports:list = []
    threshold:float = 0.6
    k_summary:int = 3
    k_total:int = 10
    namespace:str = "vectors",
    min_size:int = 200,


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # Check if all elements in the list are either IPCC or IPBES
        assert isinstance(self.sources,list)
        assert all([x in ["IPCC","IPBES","IPOS"] for x in self.sources])
        assert self.k_total > self.k_summary, "k_total should be greater than k_summary"

        # Prepare base search kwargs
        filters = {}

        if len(self.reports) > 0:
            filters["short_name"] = {"$in":self.reports}
        else:
            filters["source"] = { "$in":self.sources}

        # Search for k_summary documents in the summaries dataset
        filters_summaries = {
            **filters,
            "report_type": { "$in":["SPM"]},
        }

        docs_summaries = self.vectorstore.similarity_search_with_score(query=query,filter = filters_summaries,k = self.k_summary)
        docs_summaries = [x for x in docs_summaries if x[1] > self.threshold]

        # Search for k_total - k_summary documents in the full reports dataset
        filters_full = {
            **filters,
            "report_type": { "$nin":["SPM"]},
        }
        k_full = self.k_total - len(docs_summaries)
        docs_full = self.vectorstore.similarity_search_with_score(query=query,filter = filters_full,k = k_full)

        # Concatenate documents
        docs = docs_summaries + docs_full

        # Filter if scores are below threshold
        docs = [x for x in docs if len(x[0].page_content) > self.min_size]
        # docs = [x for x in docs if x[1] > self.threshold]

        # Add score to metadata
        results = []
        for i,(doc,score) in enumerate(docs):
            doc.metadata["similarity_score"] = score
            doc.metadata["content"] = doc.page_content
            doc.metadata["page_number"] = int(doc.metadata["page_number"]) + 1
            # doc.page_content = f"""Doc {i+1} - {doc.metadata['short_name']}: {doc.page_content}"""
            results.append(doc)

        # Sort by score
        # results = sorted(results,key = lambda x : x.metadata["similarity_score"],reverse = True)

        return results