from typing import Annotated, Literal, Optional, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START
from langgraph.graph.message import StateGraph, add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langsmith import traceable
from dotenv import load_dotenv
from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

load_dotenv()


class RAGState(TypedDict):
    """
    State schema for our agentic RAG workflow.

    LangGraph uses TypedDict for state (not Pydantic in 1.x).
    The Annotated[list, add] tells LangGraph to merge lists.
    """

    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int
    vectordb: Chroma
    
    selected_headings: list[str]


class QueryIntentSchema(BaseModel):
    intent: Literal["global", "local"] = Field(
        description="Choose 'global' if the user asks for a summary, high-level learnings, conclusions, or an overview of the entire paper. Choose 'local' if they want specific data points, metrics, definitions, or localized facts."
    )


class HeadingSelectionSchema(BaseModel):
    headings: list[str] = Field(
        default_factory=list,
        description="Return up to 5 section headings from the paper that are most relevant to the user's query.",
    )


class ProductionAgent:
    def __init__(self, llm):
        print("Init")
        self.llm = llm

    def retrieve_documents(self, state: RAGState):
        vectordb = state.get("vectordb")
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        query = state.get("rewritten_query") or state["query"]
        documents = retriever.invoke(query)
        print(documents)

        return {"documents": documents}

    def grade_documents(self, state: RAGState) -> dict:
        """
        Grade retrieved documents for relevance to the query.
        This is the KEY difference from traditional RAG - we evaluate before generating.
        """
        query = state["query"]
        documents = state["documents"]

        print(f"\n[GRADE] Evaluating {len(documents)} documents for relevance...")

        grading_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a relevance grader. Given a user query and a document,
    determine if the document contains information relevant to answering the query.

    Output ONLY a number between 0 and 1:
    - 1.0 = Highly relevant, directly answers the query
    - 0.7 = Somewhat relevant, contains related information
    - 0.3 = Marginally relevant, tangentially related
    - 0.0 = Not relevant at all

    Output ONLY the number, nothing else.""",
                ),
                (
                    "human",
                    """Query: {query}

    Document: {document}

    Relevance score (0-1):""",
                ),
            ]
        )

        # Grade each document and calculate average
        scores = []
        relevant_docs = []

        for doc in documents:
            chain = grading_prompt | self.llm
            result = chain.invoke({"query": query, "document": doc})

            try:
                score = float(result.content.strip())
            except ValueError:
                score = 0.5  # Default if parsing fails

            scores.append(score)

            print(f"  - {doc.metadata.get('heading', 'unknown')}: {score:.2f}")

            if score >= 0.5:  # Keep documents with score >= 0.5
                relevant_docs.append(doc)

        print("Relevant docs")
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"[GRADE] Average relevance: {avg_score:.2f}")
        print(f"[GRADE] Keeping {len(relevant_docs)}/{len(documents)} documents")

        return {"documents": relevant_docs, "relevance_score": avg_score}

    def rewrite_query(self, state: RAGState) -> dict:
        """
        Rewrite the query to improve retrieval.
        If the first retrieval does not yield meaningful documents, the model upgrades
        the search to a broader, global overview-style query.
        """
        query = state["query"]
        retry_count = state.get("retry_count", 0)
      
        print(f"\n[REWRITE] Attempt {retry_count + 1}: Improving query...")

      
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query rewriter for a RAG system.
    The original query didn't retrieve relevant documents.

    Rewrite the query to be more specific and likely to match relevant documents.
    Consider:
    - Adding synonyms or related terms
    - Being more specific about what information is needed
    - Rephrasing to match how documentation is typically written

    Output ONLY the rewritten query, nothing else.""",
                ),
                (
                    "human",
                    """Original query: {query}

    Rewritten query:""",
                ),
            ]
        )

        chain = rewrite_prompt | self.llm
        result = chain.invoke({"query": query})
        rewritten = result.content.strip()

        print(f"[REWRITE] Original: '{query}'")
        print(f"[REWRITE] Rewritten: '{rewritten}'")

        return {"rewritten_query": rewritten, "retry_count": retry_count + 1}

    def generate_answer(self, state: RAGState) -> dict:
        """
        Generate the final answer using retrieved documents.
        """
        query = state["query"]
        documents = state["documents"]

        print(f"\n[GENERATE] Creating answer from {len(documents)} documents...")

        # Format documents
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('heading', 'unknown')}\n{doc.page_content}"
                for doc in documents
            ]
        )

        generate_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant answering questions based on provided context.

    Use ONLY the information in the context to answer. If the context doesn't contain
    enough information, say so clearly.

    Always cite your sources by mentioning which heading the information came from.""",
                ),
                (
                    "human",
                    """Context:
    {context}

    Question: {query}

    Answer:""",
                ),
            ]
        )

        chain = generate_prompt | self.llm
        result = chain.invoke({"context": context, "query": query})

        print(f"[GENERATE] Answer generated")
        print(result.content)
        return {"generation": result.content}

    def generate_fallback(self, state: RAGState) -> dict:
        """
        Generate a fallback response when retrieval fails after all retries.
        """
        query = state["query"]

        print(
            f"\n[FALLBACK] Retrieval failed after {state.get('retry_count', 0)} attempts"
        )

        fallback_message = f"""I couldn't find relevant information to answer your question: "{query}"

    This could mean:
    1. The information isn't in my knowledge base
    2. Try rephrasing your question with different terms
    3. The topic might not be covered in the available documents

    Would you like to try a different question?"""

        return {"generation": fallback_message}

    # ============================================================
    # ROUTING FUNCTIONS
    # ============================================================

    def should_retry_or_generate(
        self,
        state: RAGState,
    ) -> Literal["rewrite", "generate", "fallback"]:
        """
        Decide whether to retry retrieval or proceed to generation.

        This is the BRAIN of agentic RAG - making decisions based on retrieval quality.
        """
        relevance_score = state.get("relevance_score", 0)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        documents = state.get("documents", [])

        print(
            f"\n[ROUTER] Evaluating: score={relevance_score:.2f}, retries={retry_count}/{max_retries}, docs={len(documents)}"
        )

        # If we have relevant documents, generate
        if relevance_score >= 0.5 and len(documents) > 0:
            print("[ROUTER] -> GENERATE (good relevance)")
            return "generate"

        # If we can retry, rewrite query
        if retry_count < max_retries:
            print("[ROUTER] -> REWRITE (low relevance, retrying)")
            return "rewrite"

        # Out of retries
        if len(documents) > 0:
            print("[ROUTER] -> GENERATE (out of retries, using available docs)")
            return "generate"
        else:
            print("[ROUTER] -> FALLBACK (no relevant documents)")
            return "fallback"

    # ============================================================
    # BUILD THE GRAPH
    # ============================================================

    @traceable(name="question_answer")
    def build_agentic_rag_graph(self):
        """
        Build the LangGraph workflow for agentic RAG.

        Flow:
        1. retrieve -> grade -> [decision]
        2. If low relevance and retries left: rewrite -> retrieve (loop)
        3. If good relevance or out of retries: generate
        4. If no documents at all: fallback
        """

        # Create the graph with our state schema
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade", self.grade_documents)
        workflow.add_node("rewrite", self.rewrite_query)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("fallback", self.generate_fallback)

        # Set entry point
        #workflow.set_entry_point("classify_intent")
        workflow.set_entry_point("retrieve")
        # Add edges
        workflow.add_edge(START, "retrieve")
        #workflow.add_edge("classify_intent", "retrieve")
        #workflow.add_edge("select_headings", "retrieve")
        workflow.add_edge("retrieve", "grade")

        # Conditional edge from grade
        workflow.add_conditional_edges(
            "grade",
            self.should_retry_or_generate,
            {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"},
        )

        # After rewrite, go back to retrieve
        workflow.add_edge("rewrite", "retrieve")

        # Terminal nodes
        workflow.add_edge("generate", END)
        workflow.add_edge("fallback", END)

        # Compile the graph
        app = workflow.compile()

        return app


# agent = ProductionAgent()
# result = agent.invoke("Do you know the North Korean Dialect? What's the difference between North Korean dialect and South Korean dialect?")
# print(result)
