import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


# --- Task 1: Define Tools ---

df = None

DATA_DIR = Path("happiness_project")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Returns:
        A dictionary with the shape and column names of the merged dataset.
    """
    global df

    frames = []

    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
        )

        year_df = year_df.rename(columns=COLUMN_NAMES)

        year_df["year"] = int(
            path.stem.split("_")[-1]
        )

        frames.append(year_df)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary of descriptive statistics, or an error if no data
        is loaded or the column does not exist.
    """
    if df is None:
        return {
            "error": "No data is loaded yet."
        }

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found."
        }

    return df[column].describe().to_dict()


@tool
def compute_correlation(
    col1: str,
    col2: str,
) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: The first column name.
        col2: The second column name.

    Returns:
        A dictionary with the column names, Pearson correlation,
        and p-value, or an error for invalid input.
    """
    if df is None:
        return {
            "error": "No data is loaded yet."
        }

    if (
        col1 not in df.columns
        or col2 not in df.columns
    ):
        return {
            "error": "One or both columns were not found."
        }

    try:
        pearson_r, p_value = pearsonr(
            df[col1],
            df[col2],
        )
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5,
) -> list[dict] | dict:
    """Return the top N countries for a column in a specific year.

    Args:
        column: The column used to rank the countries.
        year: The year to filter the dataset by.
        n: The number of countries to return.

    Returns:
        A list with the top countries and requested values,
        or an error for invalid input.
    """
    if df is None:
        return {
            "error": "No data is loaded yet."
        }

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found."
        }

    if year not in df["year"].unique():
        return {
            "error": f"Year {year} was not found."
        }

    if n <= 0:
        return {
            "error": "n must be greater than 0."
        }

    top_rows = (
        df[df["year"] == year]
        .sort_values(
            by=column,
            ascending=False,
        )
        .head(n)
    )

    return top_rows[
        ["country", column]
    ].to_dict(
        orient="records"
    )


# --- Task 2: Build the Agent ---

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns,
computing correlations, and ranking countries.

The load_happiness_data tool returns the shape and column names.
When you need the actual dataset for custom Python code, use the df
variable that is provided to you.

Write Python code directly only when the tools are not enough,
such as for custom plots or calculations the tools do not cover.

Be concise and student-friendly.
"""


agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats",
    ],
    max_steps=8,
)


# --- Task 3: Guided Queries ---

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


if __name__ == "__main__":

    for query in queries:
        print(f"\n--- Query: {query} ---")

        response = agent.run(
            query,
            reset=False,
            additional_args={"df": df},
        )

        print(response)


    # --- Task 4: My Questions ---

    my_query_1 = (
        "Use get_top_n_countries to show me the top 3 countries "
        "by gdp_per_capita in 2024."
    )

    print(f"\n--- My Query 1: {my_query_1} ---")

    response_1 = agent.run(
        my_query_1,
        reset=False,
        additional_args={"df": df},
    )

    print(response_1)

    # Comment: This query triggered tool use because the agent used
    # get_top_n_countries to answer the question.


    my_query_2 = (
        "Using df and Python code, find which region had the biggest "
        "increase in average happiness_score from 2015 to 2024."
    )

    print(f"\n--- My Query 2: {my_query_2} ---")

    response_2 = agent.run(
        my_query_2,
        reset=False,
        additional_args={"df": df},
    )

    print(response_2)

    # Comment: This query triggered code generation because none of my
    # tools can compare regional averages across different years.


# --- Reflection ---
#
# 1. In Query 3, the agent returned a Pearson correlation of 0.6313
# and said that it was statistically significant. It gave the correct
# result, but it did not explain the strength or direction of the
# relationship in words. The correlation is positive, and the p-value
# showed as 0.0 because my tool rounds it to four decimal places.
#
# 2. I was surprised that the agent could recover after making a mistake.
# In Query 1, it first treated the result from load_happiness_data like
# a DataFrame. After getting an error, it checked the result and found
# another way to answer the question correctly.
#
# 3. One useful extra tool would compare a column between different
# years or regions. For example, it could show how the average happiness
# score changed for each region between two years. This would make
# trend questions easier to answer.