from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# --- Step 1: Setup ---

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


docs_dir = Path("groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print("Groundwork documents directory found.")


# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(docs_dir).load_data()

print(f"\nLoaded {len(documents)} documents:")

for document in documents:
    print("-", document.metadata["file_name"])


# --- Step 3: Build the Index and Query Engine ---

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# The default LlamaIndex LLM in my environment was deprecated,
# so I set the model directly.
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    top_source = response.source_nodes[0]
    file_name = top_source.node.metadata.get("file_name", "Unknown")
    score = top_source.score
    text = top_source.node.get_content()

    print("\nTop source:")
    print("Document:", file_name)
    print(f"Similarity score: {score:.4f}")
    print("Text:", text[:200].replace("\n", " "))


# Step 4 Reflection
#
# The answers were confident and matched the information in the documents.
# I was surprised that the top source was not always the document I expected.
# For example, the hours question showed our_story.txt as the top source even though
# the actual hours are in faq.txt.
# The dairy-free question also returned seasonal_specials.txt first, while menu.txt
# contains the full list of dairy-free milk options.
# This showed me that the model can still give the right answer using the other
# retrieved chunks even when the first result is not the best one.


# --- Step 5: Find a Failure ---

failure_question = "Which Groundwork location is the quietest for studying?"

print("\n" + "=" * 60)
print("Failure Test")
print("Question:", failure_question)

failure_response = query_engine.query(failure_question)

print("\nAnswer:")
print(failure_response)

print("\nRetrieved sources:")

for i, source_node in enumerate(failure_response.source_nodes, start=1):
    file_name = source_node.node.metadata.get("file_name", "Unknown")
    score = source_node.score
    text = source_node.node.get_content()

    print(f"\nSource {i}")
    print("Document:", file_name)
    print(f"Similarity score: {score:.4f}")
    print("Text:", text[:200].replace("\n", " "))


# Step 5 Reflection
#
# I asked which Groundwork location is the quietest for studying because
# that information is not included in the documents.
# The retrieval was weak because the source scores were low and the first
# two documents were not really related to the question.
# The model handled it well and did not guess. It said that the information
# was not provided, so the tone was more careful than the earlier answers.
# This shows that it is safer when the model admits that the answer is missing
# instead of sounding confident without enough evidence.
# It also shows why I should still check the retrieved sources instead of
# automatically trusting an AI-generated answer.
# I would improve the system by adding a similarity threshold and telling the
# model to avoid answering when the retrieved context is too weak.


# --- Step 6: Reflection ---
#
# 1. The main LlamaIndex pipeline took about five lines of code to load the
# documents, set the models, build the index, and create the query engine.
# The keyword RAG function needed much more code just to clean words, compare
# them, calculate scores, sort the results, and choose a document.
# A framework is useful because it handles steps like chunking, embedding,
# and indexing without me having to build each part myself.
#
# 2. Another useful example could be an employee assistant for a company.
# It could answer questions from HR policies, benefits documents, onboarding guides,
# and other internal documents without employees having to search through every file.
#
# 3. RAG can still give a bad answer even when retrieval works correctly.
# The model can misunderstand the retrieved context or make a claim that the source
# does not actually support. Retrieval gives the model better information, but it
# does not completely remove hallucinations.