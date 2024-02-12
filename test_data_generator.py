from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader('data')

documents = loader.load()

for document in documents:
    document.metadata['file_name'] = document.metadata['source']

# Test data generation using Ragas
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator_llm = "gpt-3.5-turbo-0125"
critic_llm = "gpt-3.5-turbo-0125"

# generator with openai models
generator = TestsetGenerator.with_openai(generator_llm, critic_llm)

# generate testset
testset = generator.generate_with_langchain_docs(documents,
                                                 test_size=2,
                                                 distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

testset.to_pandas()

""" print all the data in the testset """
print(testset.test_data[0])
