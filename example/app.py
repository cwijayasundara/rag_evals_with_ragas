from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

loader = WebBaseLoader(
    "https://blog.langchain.dev/langchain-v0-1-0/"
)

documents = loader.load()

print(documents[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50
)

documents = text_splitter.split_documents(documents)

print(len(documents))

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever()

retrieved_documents = retriever.invoke("Why did they change to version 0.1.0?")

for doc in retrieved_documents:
    print(doc)

# Creating a Prompt Template

retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

print("retrieval_qa_prompt", retrieval_qa_prompt.messages[0].prompt.template)

template = """Answer the question based only on the following context. If you cannot answer the question with the 
context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Setting Up our Basic QA Chain

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

question = "What are the major changes in v0.1.0?"

result = retrieval_augmented_qa_chain.invoke({"question": question})

print(result["response"].content)

question = "What is LangGraph?"

result = retrieval_augmented_qa_chain.invoke({"question": question})

print(result["response"].content)
print(result["context"])

# Ragas
# Synthetic Test Set Generation : ground truths

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(documents)
print(len(documents))

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator_llm = "gpt-3.5-turbo-0125"
critic_llm = "gpt-3.5-turbo-0125"

# generator with openai models
generator = TestsetGenerator.with_openai(generator_llm, critic_llm)

testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25,
                                                                                         multi_context: 0.25})
print(testset.test_data[0])

# load to a pandas dataframe
test_df = testset.to_pandas()

print(test_df.head())

test_questions = test_df["question"].values.tolist()
test_groundtruths = test_df["ground_truth"].values.tolist()

answers = []
contexts = []

for question in test_questions:
    response = retrieval_augmented_qa_chain.invoke({"question": question})
    answers.append(response["response"].content)
    contexts.append([context.page_content for context in response["context"]])

from datasets import Dataset

response_dataset = Dataset.from_dict({
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": test_groundtruths
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

from langchain.retrievers import MultiQueryRetriever

advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever,
                                                  llm=primary_qa_llm)

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(primary_qa_llm, retrieval_qa_prompt)

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)
response = retrieval_chain.invoke({"input": "What are the major changes in v0.1.0?"})
print(response["answer"])
