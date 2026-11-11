from dotenv import load_dotenv
import os


# --- Setup ---

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---

# Concepts Q1
#
# Scenario A: RAG
# The information is in their own PDFs and those files are updated often.
# RAG makes sense because the model can search the current documents when it needs an answer.
#
# Scenario B: Fine-tuning
# They already have a lot of examples written in the exact style they want.
# Fine-tuning can help the model learn that specific writing style from those examples.
#
# Scenario C: Prompt engineering
# This is only one short report and she only needs it for one task.
# I would just give the report to the model in the prompt instead of building a RAG system.


# Concepts Q2
#
# A confidently wrong answer can be more harmful because it sounds trustworthy,
# so a person may believe it without checking.
# For example, if an AI gives the wrong medical information with confidence,
# someone could make a bad decision based on that answer.
# If the model says "I'm not sure," it gives the user a reason to double-check.
# The tone matters because confident language can make an answer feel more reliable
# even when the information is actually wrong.


# Concepts Q3
#
# 1. Extract text from source documents
# First, we get the text out of the files so the system can work with it.
#
# 2. Split text into chunks
# The documents are broken into smaller pieces so we can search only the useful parts.
#
# 3. Convert text chunks into embeddings
# Each chunk is turned into a vector that represents its meaning.
#
# 4. Receive the user's query
# The user asks a question.
#
# 5. Embed the user's query
# The question is also turned into a vector so it can be compared with the chunks.
#
# 6. Retrieve the most relevant chunks
# The system finds the chunks that are closest to the meaning of the question.
#
# 7. Inject retrieved chunks into the prompt
# The relevant chunks are added to the prompt as context for the model.
#
# 8. Generate a response from the LLM
# The model uses the question and retrieved context to create the final answer.


# --- Keyword RAG ---

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(result[0][0])

# loyalty.txt was selected even though hours.txt is the better answer.
# Several documents had the same score, and "your" matched more than one document.
# Because the scores were tied, the sorting caused loyalty.txt to be selected.


# Keyword Q2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(result[0][0])

# No document was selected because none of the query words matched the documents.
# Keyword RAG did not get this right because it only looks for exact word matches.
# menu.txt seems like the closest document because the question is about drink options.
# Semantic retrieval would be better at finding related menu content,
# but it should still say if the answer is not in the documents.


# Keyword Q3
#
# I think loyalty.txt will be selected because the question is about rewards.
# Even though the exact word "rewards" is not in the document,
# the loyalty document is the one that talks about earning points and free drinks.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(result[0][0])

# My prediction was wrong.
# loyalty.txt is still the document that makes the most sense,
# but none of the exact query words appeared in it.
# This shows another limitation of keyword search when the wording is different.


# --- Semantic RAG Concepts ---

# Semantic Q1
#
# 1. A vector embedding is a way of turning text into numbers that represent
# its meaning. Texts with similar meanings should have similar vectors.
#
# 2. The chunk with a score of 0.85 is more relevant than the one with 0.30.
# A higher cosine similarity means the chunk is closer in meaning to the query.
#
# 3. Semantic search compares meaning instead of only exact words.
# Because of that, it can still find a related chunk even if the wording is different.


# Semantic Q2
#
# | Feature                 | Keyword RAG                    | Semantic RAG              |
# |-------------------------|--------------------------------|---------------------------|
# | What is compared?       | Exact word overlap             | Meaning of the text       |
# | What is retrieved?      | Full document                  | Relevant chunks           |
# | Can it handle synonyms? | No                             | Yes                       |
# | Storage format          | Plain text dictionary          | Embeddings / vector index |
# | Relevance score         | Number of overlapping keywords | Cosine similarity         |


# --- LlamaIndex ---

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI


# The default LlamaIndex LLM in my environment was deprecated,
# so I set the model directly.
Settings.llm = OpenAI(model="gpt-4o-mini")


# LlamaIndex Q1

documents = SimpleDirectoryReader("brightleaf_pdfs").load_data()

print(f"\nLoaded {len(documents)} Brightleaf documents.")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nRetrieved sources:")

    for i, source_node in enumerate(response.source_nodes, start=1):
        score = source_node.score
        text = source_node.node.get_content()

        print(f"\nSource {i}")
        print(f"Similarity score: {score:.4f}")
        print("Text:", text[:150].replace("\n", " "))


# LlamaIndex Q1 Notes
#
# Benefits query:
# The first retrieved chunk is clearly relevant because it comes from the employee
# benefits information. The other two chunks are about partnerships and the company
# mission, so they do not seem very useful for this question.
# The answer sounds confident and specific and does not really hedge.
# I did not expect the partnership and mission chunks to be retrieved for this question.
#
# Security query:
# The first retrieved chunk is clearly relevant because it contains the security policy.
# The other two chunks are about benefits and the company mission, so they are not very
# relevant to the question.
# The answer also sounds confident and specific instead of saying things like
# "based on the context" or "I'm not sure."
# I was surprised that the benefits and mission chunks were included in the top three.


# LlamaIndex Q2

q = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    print("\n" + "=" * 60)
    print(f"similarity_top_k={top_k}")
    print("Question:", q)

    test_engine = index.as_query_engine(similarity_top_k=top_k)
    response = test_engine.query(q)

    print("\nAnswer:")
    print(response)

    print("\nSource scores:")
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f"Source {i}: {source_node.score:.4f}")


# LlamaIndex Q2 Notes
#
# The answer did not change much between top_k=1 and top_k=5.
# The first chunk was already enough to answer the question well.
# With top_k=5, the system added more chunks with lower similarity scores.
# More context is not always better because extra chunks can add information
# that is less relevant and may make the response noisier.


# LlamaIndex Q3

hard_question = "Does BrightLeaf provide free lunch to employees?"

print("\n" + "=" * 60)
print("Question:", hard_question)

response = query_engine.query(hard_question)

print("\nAnswer:")
print(response)

print("\nRetrieved sources:")

for i, source_node in enumerate(response.source_nodes, start=1):
    score = source_node.score
    text = source_node.node.get_content()

    print(f"\nSource {i}")
    print(f"Similarity score: {score:.4f}")
    print("Text:", text)


# LlamaIndex Q3 Notes
#
# I asked if BrightLeaf provides free lunch because I did not expect that
# information to be in the documents.
# The model handled it well and said that free lunch was not mentioned.
# The first retrieved chunk was related to employee benefits, which makes sense,
# but the other two chunks were not very useful for this question.
# I would improve the system by checking similarity scores and being more careful
# when the retrieved chunks are weak or not directly related to the question.


# LlamaIndex Q4

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


judge_llm = OpenAI(model="gpt-4o-mini", temperature=0)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

q = "What employee benefits does BrightLeaf offer?"

response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(
    response=response
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("\n" + "=" * 60)
print("LlamaIndex Q4 - Good Query")
print("Question:", q)
print("Faithfulness score:", faithfulness_result.score)
print("Relevancy score:", relevancy_result.score)


bad_q = "What is BrightLeaf's employee lunch menu?"

bad_response = query_engine.query(bad_q)

bad_faithfulness = faithfulness_evaluator.evaluate_response(
    response=bad_response
)

bad_relevancy = relevancy_evaluator.evaluate_response(
    query=bad_q,
    response=bad_response
)

print("\n" + "=" * 60)
print("LlamaIndex Q4 - Low Quality Query")
print("Question:", bad_q)
print("Answer:", bad_response)
print("Faithfulness score:", bad_faithfulness.score)
print("Relevancy score:", bad_relevancy.score)


# LlamaIndex Q4 Notes
#
# A faithfulness score of 1.0 means the evaluator thinks the answer is supported
# by the retrieved context. A score of 0.0 means it does not see enough support
# for the answer in the context.
#
# Relevancy checks whether the response is related to the user's question.
# This is different from faithfulness because an answer can be relevant to the
# question but still not be supported by the retrieved documents.
#
# In my latest run, the good query received 1.0 for both scores.
# The lunch menu query received 0.0 for faithfulness but 1.0 for relevancy.
# The response was still related to the question, but the documents did not
# contain lunch menu information.
#
# I also noticed that the evaluator scores can change between runs,
# which shows that LLM-based evaluation is not always perfectly consistent.
#
# LLM-as-a-judge means another language model is used to evaluate the response.
# It is useful for RAG because answer quality is not always something we can
# measure with a simple exact-match accuracy score.