import os
import numpy as np
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Agent3Prompt import CATEGORY_PROMPTS
from LoadPapers import load_combined_papers

# ==========================================
# 0. LLMs API Configuration (The following example uses GPT-5 Mini via OpenRouter)
# ==========================================
# TODO: Add your API key and URL (example: using OpenRouter to call GPT-5-mini)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-5-mini",
    temperature=0
)


# ==========================================
# 1. Defining Sysytem State
# ==========================================
class GraphState(TypedDict):
    query: str
    taxonomy_category: str
    retrieved_docs: List[dict]
    summary: str
    final_response: str


# ==========================================
# 2. Defining the Structured Output of the Taxonomy
# ==========================================
class TaxonomyClassification(BaseModel):
    category: str = Field(
        description="""
        Classify the user's question into the following AI for Inclusive Education taxonomy:
        - Disability-General
        - Disability-NDD-General
        - Disability-NDD-ASD
        - Disability-NDD-ADHD
        - Disability-NDD-ID
        - Disability-NDD-SLD
        - Disability-Sensory-General
        - Disability-Sensory-Visual
        - Disability-Sensory-Hearing
        - Disability-Motor
        - Sociocultural-General
        - Sociocultural-Gender
        - Sociocultural-Language
        - Sociocultural-Race and Ethnicity
        - Socioeconomic
        - Geographic
        - Giftedness
        - Lifestage
        """
    )


# ==========================================
# 3. RAG Vector Retriever
# ==========================================
CATEGORY_FILE_MAP = {
    "Disability-General": "Included Papers/Disability-Based Inclusion/General/combined_texts.txt",
    "Disability-NDD-General": "Included Papers/Disability-Based Inclusion/Neuro-developmental Disorder/General/combined_texts.txt",
    "Disability-NDD-ASD": "Included Papers/Disability-Based Inclusion/Neuro-developmental Disorder/ASD/combined_texts.txt",
    "Disability-NDD-ADHD": "Included Papers/Disability-Based Inclusion/Neuro-developmental Disorder/ADHD/combined_texts.txt",
    "Disability-NDD-ID": "Included Papers/Disability-Based Inclusion/Neuro-developmental Disorder/ID/combined_texts.txt",
    "Disability-NDD-SLD": "Included Papers/Disability-Based Inclusion/Neuro-developmental Disorder/SLD/combined_texts.txt",
    "Disability-Sensory-General": "Included Papers/Disability-Based Inclusion/Sensory Impairment/General/combined_texts.txt",
    "Disability-Sensory-Visual": "Included Papers/Disability-Based Inclusion/Sensory Impairment/Visual Impairment/combined_texts.txt",
    "Disability-Sensory-Hearing": "Included Papers/Disability-Based Inclusion/Sensory Impairment/Hearing Impairment/combined_texts.txt",
    "Disability-Motor": "Included Papers/Disability-Based Inclusion/Motor Impairment/combined_texts.txt",
    "Sociocultural-General": "Included Papers/Sociocultural-Based Inclusion/General/combined_texts.txt",
    "Sociocultural-Gender": "Included Papers/Sociocultural-Based Inclusion/Gender/combined_texts.txt",
    "Sociocultural-Language": "Included Papers/Sociocultural-Based Inclusion/Language/combined_texts.txt",
    "Sociocultural-Race and Ethnicity": "Included Papers/Sociocultural-Based Inclusion/Race and Ethnicity/combined_texts.txt",
    "Socioeconomic": "Included Papers/Socioeconomic-Based Inclusion/combined_texts.txt",
    "Geographic": "Included Papers/Geographic-Based Inclusion/combined_texts.txt",
    "Giftedness": "Included Papers/Giftedness-Based Inclusion/combined_texts.txt",
    "Lifestage": "Included Papers/Lifestage-Based Inclusion/combined_texts.txt"
}


class VectorStore:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._cache = {}
        self._tfidf = TfidfVectorizer(stop_words="english")

    def search(self, category: str, query: str) -> List[dict]:
        docs = self._load_or_fallback(category)
        if not docs:
            return [{"text": "No related literature。", "metadata": {"source": "N/A"}}]

        doc_texts = [d["text"] for d in docs]
        corpus = doc_texts + [query]
        tfidf = self._tfidf.fit_transform(corpus)
        doc_vecs = tfidf[:-1]
        query_vec = tfidf[-1]

        scores = cosine_similarity(query_vec, doc_vecs).flatten()
        top_n = min(self.top_k, len(docs))
        top_idx = np.argsort(scores)[-top_n:][::-1]

        results = [docs[i] for i in top_idx]
        for i, r in enumerate(results):
            r["_score"] = float(scores[top_idx[i]])
        return results

    def _load_or_fallback(self, category: str) -> List[dict]:
        if category in self._cache:
            return self._cache[category]

        path = CATEGORY_FILE_MAP.get(category, "")
        if path and os.path.isfile(path):
            _, docs = load_combined_papers(path)
            self._cache[category] = docs
            print(f"  [RAG] Loading {len(docs)} papers ← {path}")
            return docs

        print(f"  [RAG] No data file for category「{category}」, returning empty.")
        return []


retriever = VectorStore(top_k=5)


# ==========================================
# 4. Defining Agents (Nodes)
# ==========================================

# Agent 1: The Classifier
def classify_intent(state: GraphState):
    print("--- Executing Agent 1: Requirement Classification ---")
    query = state["query"]

    structured_llm = llm.with_structured_output(TaxonomyClassification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Acting as an expert in inclusive education and artificial intelligence, "
                   "precisely categorize the user's request according to the given systematic taxonomy."
                   """
                   ### Examples and Clarifications:
                  1. **Specific Classification:**
                  - **User Query:** "How can we design a teacher-student communication assistive system for children with autism?"
                    **Classification:** 'Disability-NDD-ASD'
                  - **User Query:** "How can AI technology be used to help elderly individuals learn new knowledge?"
                    **Classification:** 'Lifestage'

                  2. **Handling Multi-Subcategory Queries (Fallback to 'General'):**
                  If a user's query spans multiple subcategories under the same primary category, classify it into the 'General' tier of that specific category.
                  - **User Query:** "How can we help visually and hearing-impaired children access course material information?" 
                    *(Note: This involves both visual and hearing impairments under the Sensory category)*
                    **Classification:** 'Disability-Sensory-General'
                  - **User Query:** "How can AI be leveraged to support refugee children in their learning?"
                    *(Note: This intersects multiple sociocultural dimensions like language, race and ethnicity)*
                    **Classification:** 'Sociocultural-General'
                   """),
        ("human", "{query}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"query": query})

    print(f"Classification Results: {result.category}\n")
    return {"taxonomy_category": result.category}


# Agent 2: The Retriever & Summarizer
def retrieve_and_summarize(state: GraphState):
    print("--- Executing Agent 2: Literature Search and Summarization ---")
    query = state["query"]
    category = state["taxonomy_category"]

    docs = retriever.search(category=category, query=query)

    context_str = "\n\n".join(
        [f"[Paper: {doc['metadata']['source']}] : {doc['text']}" for doc in docs]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior academic researcher in the field of AI for Inclusive Education. "
                   "Please summarize the technological solutions and design recommendations for addressing the pain points of Students with Special Education Needs and Disabilities and relevant educational stakeholders in the following literature. "
                   "The literature reference IDs must be retained."),
        ("human", "The developer's requirements: {query}\n\nRelevant literature content:\n{context}")
    ])

    chain = prompt | llm
    summary_result = chain.invoke({"query": query, "context": context_str})

    print(f"Summary generated.\n")
    return {"retrieved_docs": docs, "summary": summary_result.content}


# ==========================================
# Agent 3: 18-Category Specialized Prompt Template
# ==========================================

# Agent 3: The Architect
def generate_final_response(state: GraphState):
    print("--- Executing Agent 3: Generating System Design Guidelines ---")
    query = state["query"]
    summary = state["summary"]
    category = state["taxonomy_category"]

    system_prompt = CATEGORY_PROMPTS.get(
        category,
        """
        You are a top-tier Architectural Consultant in the field of **AI in Inclusive Education**. You are currently guiding a developer through the design of their AI-driven inclusive education system.
        Using the provided literature summary and the developer's specific requirements, provide final system design guidance.
    
        ### General principles for reference
        **Secure & Ubiquitous Access:** Prioritize high responsiveness, mobile portability, and strict data privacy.
        **Localized Design:** Adapt interfaces and pedagogy to regional cultural, institutional, and linguistic standards.
        **System Integration:** Unify fragmented multimodal assistive tools (e.g., screen readers, interpreters) into a single, cohesive platform.
        **Data Resilience:** Mitigate the scarcity of SEND training data by utilizing data augmentation and LLM-driven synthetic data generation."

        ### Strict Requirements
        1. **Tone**: Maintain a direct, professional, and concise academic advisory tone.
        2. **Technical Depth**: The guidance must include specific recommendations for the **AI technology stack**.
        3. **Termination** Do NOT include any promotional, conversational, or helpful follow-up offers at the end of your response
        4. **Citations**: You must use the `[Literature ID]` format for strict intra-sentence referencing whenever mentioning specific technologies, methodologies, or conclusions.
        5. **Reference List**: Include a formal "Reference List" at the end of the response, mapping every ID to its source.
    
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "The developer's requirements: {query}\n\nLiterature Summary:\n{summary}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query, "summary": summary})

    return {"final_response": response.content}


# ==========================================
# 5. Building LangGraph Workflow
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("classifier", classify_intent)
workflow.add_node("summarizer", retrieve_and_summarize)
workflow.add_node("generator", generate_final_response)

workflow.add_edge("classifier", "summarizer")
workflow.add_edge("summarizer", "generator")
workflow.add_edge("generator", END)

workflow.set_entry_point("classifier")
app = workflow.compile()

# ==========================================
# 6. Exexution
# ==========================================
if __name__ == "__main__":
    user_query = input("Enter your query: ")

    final_state = app.invoke({"query": user_query})

    print("=" * 50 + "\n[Final Result]:\n")
    print(final_state["final_response"])