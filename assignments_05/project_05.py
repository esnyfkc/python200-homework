import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


# --- Task 1: Setup and System Prompt ---

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """
You are a job application coach helping career changers improve their
resumes, cover letters, and other job application materials.

Stay focused on job application materials and related questions.

Always remind the user to review and edit your suggestions before
submitting them anywhere.

You may not know the specific norms or expectations of every industry,
so remind the user to use their own judgment when deciding what to include.
"""


# I made the system prompt specific to job applications so the assistant
# stays focused instead of giving unrelated advice.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point to sound stronger and clearer.
Use strong action verbs, but do not add numbers, percentages,
timeframes, results, tools, or responsibilities that are not
already stated in the original bullet.

Respond ONLY with a valid JSON list, no other text.
Each item must have these two keys:
"original" and "improved"

Bullet points:
```
{bullet_text}
```
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = get_completion(messages)

    # Sometimes the model adds a JSON code block, so I remove it before parsing.
    cleaned_response = response.strip()

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]

    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    cleaned_response = cleaned_response.strip()

    results = json.loads(cleaned_response)

    for item in results:
        print("\nOriginal:", item["original"])
        print("Improved:", item["improved"])

    return results


# These bullets are weak because they are vague and use basic verbs.
# The model mostly makes them clearer by using stronger verbs and wording.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should be 3-5 sentences.
Keep it confident, specific, and avoid generic phrases.

Here are two examples of the style and tone to follow.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse and recently completed
a data analytics bootcamp.
Opening: After seven years as a registered nurse, I recently completed
a data analytics bootcamp as I prepare to move into data analysis.
My background combines years of healthcare experience with my recent
technical training. I am applying for this Data Analyst role with
experience from both areas.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations and two years
learning Python.
Opening: After ten years in retail banking operations, I have spent
the last two years learning Python. I am now applying for a Junior
Software Engineer role in fintech. My background combines direct
experience in banking operations with my recent programming work.

Now write an opening paragraph for this person.

Role: {job_title}
Background: {background}

Do not invent or infer experience, skills, achievements, results,
responsibilities, motivations, interests, or goals.

Do not assume skills based only on a previous job title.

Use only facts that are explicitly stated in the background.

Opening:
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return get_completion(messages)


# I chose examples of people moving from another career into a technical role
# because that matches the purpose of this tool.
# Few-shot examples help control the tone, structure, and amount of detail.


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print(
            "Please rephrase your message so I can help with "
            "your job application."
        )
        return False

    return True


# --- Tests for Tasks 2, 3, and 4 ---

def run_task_tests():
    print("\n--- Task 2 Test ---")

    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]

    rewrite_bullets(bullets)

    print("\n--- Task 3 Test ---")

    job_title = "Junior Data Engineer"
    background = (
        "Five years of experience as a middle school math teacher; "
        "recently completed a Python course and built data pipelines "
        "using Pandas."
    )

    cover_letter = generate_cover_letter(job_title, background)

    print("\nCover Letter Opening:")
    print(cover_letter)

    print("\n--- Task 4 Test ---")

    safe_text = "Can you help me improve my resume?"
    flagged_text = "I want to hurt someone."

    print("Safe test:", is_safe(safe_text))
    print("Flagged test:", is_safe(flagged_text))


# I used run_task_tests() while building the project to check Tasks 2-4.
# I keep it separate so the chatbot does not print all the test outputs
# every time I start it.


# --- Task 5: Chatbot Loop ---

def run_chatbot():
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Other questions about your job application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print(
                "\nJob Application Helper: "
                "Good luck with your applications!"
            )
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print(
                "\nJob Application Helper: "
                "Paste your bullet points below, one per line."
            )
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            if not raw_bullets:
                print(
                    "\nJob Application Helper: "
                    "No bullet points were entered."
                )
                continue

            bullet_text = "\n".join(raw_bullets)

            if not is_safe(bullet_text):
                continue

            rewrite_bullets(raw_bullets)

            print(
                "\nJob Application Helper: "
                "Please review and edit these suggestions before "
                "using them in your resume."
            )

        elif "cover letter" in user_input.lower():
            job_title = input(
                "Job Application Helper: What is the job title? "
            ).strip()

            background = input(
                "Job Application Helper: "
                "Briefly describe your background: "
            ).strip()

            if not job_title or not background:
                print(
                    "\nJob Application Helper: "
                    "I need both the job title and your background."
                )
                continue

            if not is_safe(job_title) or not is_safe(background):
                continue

            cover_letter = generate_cover_letter(
                job_title,
                background
            )

            print("\nJob Application Helper:")
            print(cover_letter)

            print(
                "\nPlease review and edit this draft before submitting it. "
                "I may not know all the expectations for your specific "
                "industry, so use your own judgment too."
            )

        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)

            messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                }
            )


if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Option A - Comment block
#
# The bot can give biased advice because the text it learned from may
# represent some communication styles or industries more than others.
# I also noticed during testing that the model can add details that I
# never gave it, so I would not submit its output without checking it first.
# If I used this tool professionally, I would keep the moderation check
# and show a clear reminder before someone uses the final text.
# I would also remind users to use their own judgment because expectations
# can be different depending on the industry.