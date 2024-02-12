from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from test_data import test_data

load_dotenv()

loader = DirectoryLoader('data')

documents = loader.load()

print("there are:", len(documents))
print(documents[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0
)

documents = text_splitter.split_documents(documents)

print("there are now:", len(documents), "after splitting")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

#  save the embeddings in FAISS
vector_store = FAISS.from_documents(documents, embeddings)

# create a retriever from the vector store
retriever = vector_store.as_retriever()

call_transcript_query = "List all the calls where agent Lucas Green was involved"

retrieved_documents = retriever.invoke(call_transcript_query)

for doc in retrieved_documents:
    print(doc)

# create a prompt template

template = """Answer the question based only on the following context. If you cannot answer the question with the 
context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Setting Up our Basic QA Chain
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

result = retrieval_augmented_qa_chain.invoke({"question": call_transcript_query})

print("the content of the document", result["response"].content)
print("the context is:", result["context"])

# Ragas

""" create a list to hold test questions """
test_data_questions = [test_data.test_call_transcript_query_1,
                       test_data.test_call_transcript_query_2]

""" create a list to hold the ground truth """
test_data_answers = [test_data.test_call_transcript_1,
                     test_data.test_call_transcript_2]

answers = []
contexts = []

for question in test_data_questions:
    response = retrieval_augmented_qa_chain.invoke({"question": question})
    answers.append(response["response"].content)
    contexts.append([context.page_content for context in response["context"]])

from datasets import Dataset

response_dataset = Dataset.from_dict({
    "question": test_data_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": test_data_answers
})

print(response_dataset[0])

# Evaluating with Ragas
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]

results = evaluate(response_dataset, metrics)
print("Ragas eval is", results)

results_df = results.to_pandas()

"""loop through the results df and print the results"""
for index, row in results_df.iterrows():
    print(f"Question: {row['question']}")
    print(f"Answer: {row['answer']}")
    print(f"Ground Truth: {row['ground_truth']}")
    print(f"Faithfulness: {row['faithfulness']}")
    print(f"Answer Relevancy: {row['answer_relevancy']}")
    print(f"Answer Correctness: {row['answer_correctness']}")
    print(f"Context Recall: {row['context_recall']}")
    print(f"Context Precision: {row['context_precision']}")
    print("\n")

# Testing a More Performant Retriever
# from langchain.retrievers import MultiQueryRetriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain import hub
# from langchain.chains import create_retrieval_chain
#
# advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever,
#                                                   llm=primary_qa_llm)
#
# retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# document_chain = create_stuff_documents_chain(primary_qa_llm,
#                                               retrieval_qa_prompt)
#
# retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)
#
# response = retrieval_chain.invoke({"input": "list all the calls happened on the 15th of July 2023"})
# print(response["answer"])
