import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


# --- Completions API ---

# API Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print("Response:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal tokens:")
print(response.usage.total_tokens)


# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    print(f"\nTemperature {temperature}:")
    print(response.choices[0].message.content)


# Temperature 0 is more predictable, while higher temperatures
# can give more varied answers.
# If I needed a consistent output, I would use temperature 0.


# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

print("\nThree Panda Facts:")

for i, choice in enumerate(response.choices, start=1):
    print(f"{i}. {choice.message.content}")


# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work.",
        }
    ],
    max_tokens=15,
)

print("\nShort Response:")
print(response.choices[0].message.content)


# The response was cut short because max_tokens limited how much
# the model could generate.
# This can help control response length and cost in a real application.


# --- System Messages and Personas ---

# System Q1

messages = [
    {
        "role": "system",
        "content": (
            "You are a patient, encouraging Python tutor. "
            "You always explain things simply and end with a word of encouragement."
        ),
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nPatient Tutor:")
print(response.choices[0].message.content)


messages = [
    {
        "role": "system",
        "content": (
            "You are a dramatic pirate who teaches Python. "
            "Keep your explanation short and talk like a pirate."
        ),
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nPirate Tutor:")
print(response.choices[0].message.content)


# The information was similar, but the tone and style changed
# because of the system message.


# System Q2

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "My name is Jordan and I'm learning Python.",
    },
    {
        "role": "assistant",
        "content": (
            "Nice to meet you, Jordan! Python is a great choice. "
            "What would you like to work on?"
        ),
    },
    {
        "role": "user",
        "content": "Can you remind me what my name is?",
    },
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nConversation Memory:")
print(response.choices[0].message.content)


# The model knows Jordan's name because I included the previous
# messages in the conversation history.


# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of this review as "
                    "positive, negative, or mixed.\n\n"
                    f"Review: {review}"
                ),
            }
        ],
    )

    print(f"\nReview {i}:")
    print(response.choices[0].message.content)


# Prompt Q2 - One-Shot

example = """
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    prompt = f"""
Classify the sentiment as positive, negative, or mixed.

{example}

Review: "{review}"
Sentiment:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    print(f"\nOne-Shot Review {i}:")
    print(response.choices[0].message.content)


# Adding one example made most of the answers shorter and closer
# to the format I wanted, but it was not perfectly consistent.


# Prompt Q3 - Few-Shot

examples = """
Example 1:
Review: "The setup was easy and everything worked perfectly."
Sentiment: positive

Example 2:
Review: "The app is slow and keeps freezing."
Sentiment: negative

Example 3:
Review: "The design looks great, but some features are confusing."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    prompt = f"""
Classify the sentiment as positive, negative, or mixed.

{examples}

Review: "{review}"
Sentiment:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    print(f"\nFew-Shot Review {i}:")
    print(response.choices[0].message.content)


# Zero-shot is useful when the task is simple and clear.
# One-shot helps show the format I want.
# Few-shot is useful when I want the model to follow a pattern more consistently.


# Prompt Q4 - Chain of Thought

problem = """
A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per year
than her post-raise salary.

What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                f"{problem}\n"
                "Show the calculation in short steps, then clearly label "
                "the final answer."
            ),
        }
    ],
)

print("\nSalary Calculation:")
print(response.choices[0].message.content)


# Breaking the problem into smaller steps can help the model
# handle one calculation at a time and makes mistakes easier to notice.


# Prompt Q5 - Structured Output

review = (
    "I've been using this tool for three months. "
    "It handles large datasets well, but the UI is clunky "
    "and the export options are limited."
)

prompt = f"""
Analyze this review and return only valid JSON.

Use exactly these keys:
- sentiment
- confidence
- reason

The confidence must be a float from 0 to 1.
The reason must be one sentence.

Review:
{review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

raw_response = response.choices[0].message.content

print("\nRaw JSON Response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nSentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nCould not parse the response as JSON.")
    print("Raw response:")
    print(raw_response)


# Prompt Q6 - Delimiters

user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. "
    "Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

print("\nInstruction Text:")
print(response.choices[0].message.content)


regular_text = (
    "The weather was cool this morning, and the park was quiet. "
    "A few people were walking their dogs near the lake."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{regular_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

print("\nRegular Text:")
print(response.choices[0].message.content)


# Delimiters separate the user's text from my instructions.
# This helps prevent the model from treating the user's text as instructions.


# --- Local Models with Ollama ---

# Ollama Q1

"""
Ollama output:

A large language model is a type of AI that can understand and generate text,
learning from vast amounts of data to improve its performance over time.
It enables tasks like answering questions, writing articles, or summarizing
information by analyzing patterns in the text.
"""

ollama_prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": ollama_prompt,
        }
    ],
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)


# Ollama's answer was simpler, and OpenAI gave a little more detail.
# One advantage of running a model locally is that the prompt can stay
# on my computer.
# One disadvantage is that a small local model can be less powerful
# and it also uses my computer's resources.