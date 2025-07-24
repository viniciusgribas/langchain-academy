from operator import add
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# The structure of the logs


class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

# Failure Analysis Sub-graph


class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]


class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]


def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}


def generate_summary_fa(state):
    """ Generate summary of failures """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}


fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary_fa)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

# Summarization subgraph


class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]


class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]


def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}


def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}


qs_builder = StateGraph(QuestionSummarizationState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

# Entry Graph


class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str  # This will only be generated in the FA sub-graph
    report: str  # This will only be generated in the QS sub-graph
    # This will be generated in BOTH sub-graphs
    processed_logs:  Annotated[List[int], add]


def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

# Wrapper functions to handle state transformation for subgraphs


def failure_analysis_node(state):
    # Extract relevant state for failure analysis
    fa_input: FailureAnalysisState = {
        "cleaned_logs": state["cleaned_logs"],
        "failures": [],
        "fa_summary": "",
        "processed_logs": []
    }
    # Run the failure analysis subgraph
    fa_graph = fa_builder.compile()
    result = fa_graph.invoke(fa_input)
    # Return only the fields that should be updated in the main state
    return {
        "fa_summary": result["fa_summary"],
        "processed_logs": result["processed_logs"]
    }


def question_summarization_node(state):
    # Extract relevant state for question summarization
    qs_input: QuestionSummarizationState = {
        "cleaned_logs": state["cleaned_logs"],
        "qs_summary": "",
        "report": "",
        "processed_logs": []
    }
    # Run the question summarization subgraph
    qs_graph = qs_builder.compile()
    result = qs_graph.invoke(qs_input)
    # Return only the fields that should be updated in the main state
    return {
        "report": result["report"],
        "processed_logs": result["processed_logs"]
    }


entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", question_summarization_node)
entry_builder.add_node("failure_analysis", failure_analysis_node)

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()
