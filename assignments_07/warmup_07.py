from datetime import datetime
import json
import os
from pathlib import Path

import matplotlib

# Use a non-interactive backend so CodeAgent plots can be saved without
# opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    ToolCallingAgent,
    tool,
)


# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1


def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}


print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2


if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")


client = OpenAI()


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


# Prediction:
# I do not think this query will trigger a tool call because the only
# available tool is get_current_time, which does not help with a
# temperature conversion.
# I expect one API call because the model should answer the question directly.


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""

    else:
        print("No tools needed....")

    return first_message.content or ""


result_q2 = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit"
)

print("\nQ2 Result:")
print(result_q2)

# My prediction was correct. The agent did not call get_current_time
# because that tool was not useful for this question.
# It answered directly, so only one API call was made.


# Q3


tools.append(celsius_to_fahrenheit_schema)


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time
and convert Celsius temperatures to Fahrenheit.
Use get_current_time when the user asks about the time.
Use celsius_to_fahrenheit when the user asks to convert Celsius to Fahrenheit."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            if function_name == "get_current_time":
                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":
                tool_result = celsius_to_fahrenheit(
                    **function_args
                )

            else:
                tool_result = (
                    f"Error: unknown tool {function_name}."
                )

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""

    else:
        print("No tools needed....")

    return first_message.content or ""


response_a = run_agent(
    "What is 37 degrees Celsius in Fahrenheit?"
)
print("Response A:", response_a)

# The agent called celsius_to_fahrenheit because the question
# asked for a Celsius to Fahrenheit conversion.


response_b = run_agent(
    "What is the boiling point of water in plain English?"
)
print("Response B:", response_b)

# The agent did not call a tool because this was a general
# knowledge question that it could answer directly.


# --- Lesson 03: Multi-Tool Agent ---

# Q4


RESOURCES_DIR = Path("resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(
        self,
        filename: str,
    ) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"

        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []

        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = (
                files[0]
                if files
                else "your_file.csv"
            )

            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }

        return None

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }

        return {
            "files": files
        }

    def load_csv(
        self,
        filename: str,
    ):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(
            filename
        )

        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": (
                    f"Could not find '{filename}' in resources/."
                ),
                "available_files": (
                    self._available_csv_files()
                ),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": (
                f"Loaded {filename} with shape {self.df.shape}."
            ),
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()

        if error:
            return error

        return self.df.columns.tolist()

    def summarize_columns(
        self,
        columns: list[str] | None = None,
    ):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all").
        """
        error = self._ensure_loaded()

        if error:
            return error

        if columns is None:
            data = self.df

        else:
            missing = [
                c
                for c in columns
                if c not in self.df.columns
            ]

            if missing:
                return {
                    "error": (
                        "These columns are not in the data: "
                        f"{missing}"
                    )
                }

            data = self.df[columns]

        summary = (
            data.describe(include="all")
            .transpose()
            .round(3)
        )

        return summary.to_dict()

    def describe_column(
        self,
        column: str,
    ):
        """
        Return basic statistics for one column.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": (
                    f"'{column}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        summary = (
            self.df[column]
            .describe()
            .to_dict()
        )

        cleaned = {}

        for key, value in summary.items():
            if isinstance(
                value,
                (int, float),
            ):
                cleaned[key] = round(
                    value,
                    3,
                )
            else:
                cleaned[key] = value

        return cleaned

    def compute_correlation(
        self,
        col1: str,
        col2: str,
    ):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if (
            col1 not in self.df.columns
            or col2 not in self.df.columns
        ):
            return {
                "error": (
                    "One or both columns were not found. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        pearson_r, p_value = pearsonr(
            self.df[col1],
            self.df[col2],
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(
                float(pearson_r),
                4,
            ),
            "p_value": round(
                float(p_value),
                4,
            ),
        }

    def plot_data(
        self,
        y: str,
        x: str | None = None,
        plot_type: str = "line",
    ):
        """
        Plot from the active CSV.

        If x is None, plot y against the row index.
        If x is provided, plot y against x.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in [
            "scatter",
            "line",
        ]:
            return (
                "Error: I can only do "
                "'scatter' or 'line'."
            )

        if y not in self.df.columns:
            return (
                f"Error: column '{y}' is not in "
                f"{self.df.columns.tolist()}"
            )

        if x == y:
            x = None

        if (
            plot_type == "scatter"
            and x is None
        ):
            return (
                "Error: scatter plots need "
                "both x and y columns."
            )

        title_csv = (
            self.csv_name
            or "current CSV"
        )

        if x is None:
            ax = self.df[y].plot(
                kind="line"
            )

            ax.set_title(
                f"{title_csv} | "
                f"Line plot: {y} vs row index"
            )

            plt.show()

            return (
                f"Plotted {y} vs row index "
                "as a line plot."
            )

        if x not in self.df.columns:
            return (
                f"Error: column '{x}' is not in "
                f"{self.df.columns.tolist()}"
            )

        ax = self.df.plot(
            x=x,
            y=y,
            kind=plot_type,
        )

        ax.set_title(
            f"{title_csv} | "
            f"{plot_type.title()} plot: "
            f"{y} vs {x}"
        )

        plt.show()

        return (
            f"Plotted {y} vs {x} "
            f"as a {plot_type}."
        )


csv_backend = CsvManager(
    RESOURCES_DIR
)


node_tools = {
    "list_csv_files": (
        csv_backend.list_csv_files
    ),
    "load_csv": (
        csv_backend.load_csv
    ),
    "get_columns": (
        csv_backend.get_columns
    ),
    "summarize_columns": (
        csv_backend.summarize_columns
    ),
    "describe_column": (
        csv_backend.describe_column
    ),
    "compute_correlation": (
        csv_backend.compute_correlation
    ),
    "plot_data": (
        csv_backend.plot_data
    ),
}


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": (
                "List available CSV files "
                "in the resources/ folder."
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": (
                "Load a CSV file from the resources/ folder "
                "and make it the active dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "CSV filename in resources/, "
                            "e.g. 'bike_commute.csv'."
                        ),
                    }
                },
                "required": [
                    "filename"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": (
                "Get the column names of "
                "the currently loaded CSV."
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": (
                "Show basic summary statistics "
                "for columns using pandas.describe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of column names. "
                            "If omitted, summarize all columns."
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": (
                "Show basic summary statistics "
                "for a single column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Column name to describe."
                        ),
                    }
                },
                "required": [
                    "column"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation "
                "between two columns in the loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": (
                            "The first column for "
                            "the correlation."
                        ),
                    },
                    "col2": {
                        "type": "string",
                        "description": (
                            "The second column for "
                            "the correlation."
                        ),
                    },
                },
                "required": [
                    "col1",
                    "col2",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": (
                "Plot data from the active CSV. "
                "If only y is provided, "
                "plot y vs row index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": (
                            "Column name for y-axis."
                        ),
                    },
                    "x": {
                        "type": "string",
                        "description": (
                            "Optional column name "
                            "for x-axis."
                        ),
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": [
                            "scatter",
                            "line",
                        ],
                        "description": (
                            "Type of plot to create."
                        ),
                    },
                },
                "required": [
                    "y"
                ],
            },
        },
    },
]


# Q5


def run_agent_cycle(
    messages,
    user_text,
    max_tool_rounds=5,
):
    """
    Run through one ReAct agent loop using a simple tool-using agent.
    """
    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    def observe_tool_result(
        tool_call_id,
        result,
    ):
        content = (
            json.dumps(
                result,
                default=str,
            )
            if not isinstance(
                result,
                str,
            )
            else result
        )

        return {
            "role": "tool",
            "tool_call_id": (
                tool_call_id
            ),
            "content": content,
        }

    for loop_idx in range(
        max_tool_rounds
    ):
        response = (
            client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=tools_schema,
            )
        )

        msg = (
            response
            .choices[0]
            .message
        )

        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry[
                "tool_calls"
            ] = [
                tc.model_dump()
                for tc in msg.tool_calls
            ]

        messages.append(
            assistant_entry
        )

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = (
                tool_call
                .function
                .name
            )

            tool_args = json.loads(
                tool_call.function.arguments
                or "{}"
            )

            print(
                f"ACT: {name}({tool_args})"
            )

            fn = node_tools.get(
                name
            )

            if fn is None:
                result = {
                    "error": (
                        f"Tool '{name}' not found."
                    )
                }

            else:
                try:
                    result = (
                        fn(**tool_args)
                        if tool_args
                        else fn()
                    )

                except Exception as e:
                    print(
                        f"Tool error in {name}: "
                        f"{type(e).__name__}: {e}"
                    )

                    result = {
                        "error": (
                            f"Tool '{name}' failed: "
                            f"{type(e).__name__}: {e}"
                        )
                    }

            messages.append(
                observe_tool_result(
                    tool_call.id,
                    result,
                )
            )

    return (
        "I hit the tool-round limit. "
        "Try a simpler request."
    )


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


result = run_agent_cycle(
    messages,
    (
        "Load bike_commute.csv and compute the correlation between "
        "avg_traffic_density and avg_speed_kmh."
    ),
)

print("\nQ5 Result:")
print(result)


# Q6

# system: gives the agent its instructions
# user: contains the user's request
# assistant: contains the model response or its tool request
# tool: contains the result returned by a tool

print("\nQ6 Messages:")
print(
    json.dumps(
        messages,
        indent=2,
        default=str,
    )
)


# --- Lesson 04: smolagents ---

# Q7


csv_manager = csv_backend


@tool
def compute_correlation(
    col1: str,
    col2: str,
) -> dict:
    """Compute the Pearson correlation between two columns.

    Args:
        col1: The first column name.
        col2: The second column name.

    Returns:
        A dictionary with the correlation coefficient and p-value,
        or an error if the data or columns are not available.
    """
    return csv_manager.compute_correlation(
        col1,
        col2,
    )


print(
    compute_correlation.description
)

# With smolagents I do not need to write the full JSON schema myself.
# It uses the function name, type hints, and docstring to build the tool.
# I still need to give it clear type hints and a useful docstring.


# Q8


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return (
        csv_manager.list_csv_files()
    )


@tool
def load_csv(
    filename: str,
) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. It can include .csv or not.

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(
        filename
    )


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return (
        csv_manager.get_columns()
    )


@tool
def summarize_columns(
    columns: list[str] | None = None,
) -> dict:
    """Return summary statistics for selected columns or all columns.

    Args:
        columns: Column names to summarize. If None, summarize all columns.

    Returns:
        A dict of summary statistics, or an error dict.
    """
    return (
        csv_manager
        .summarize_columns(columns)
    )


@tool
def describe_column(
    column: str,
) -> dict:
    """Describe one column using basic statistics.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic statistics, or an error dict.
    """
    return (
        csv_manager
        .describe_column(column)
    )


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line",
) -> str | dict:
    """Plot data from the active CSV.

    Args:
        y: Column name for the y-axis.
        x: Column name for the x-axis. If None, use row index.
        plot_type: The plot type, either "line" or "scatter".

    Returns:
        A success message, or an error.
    """
    return csv_manager.plot_data(
        y=y,
        x=x,
        plot_type=plot_type,
    )


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]


api_key = os.getenv(
    "OPENAI_API_KEY"
)

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)


SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)


tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)


CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- If the user requests plot styling that plot_data cannot control,
  write matplotlib code directly.
- When writing custom matplotlib code, do not use plt.show().
- Save the custom plot to outputs/q8_codeagent.png and close it after saving.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""


code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "numpy",
    ],
    max_steps=8,
)


prompt = (
    "Load bike_commute.csv. Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)


response_tool = tool_agent.run(
    prompt
)

response_code = code_agent.run(
    prompt,
    additional_args={
        "csv_manager": csv_manager
    },
)


print(
    "\nQ8 ToolCallingAgent Response:"
)
print(response_tool)

print(
    "\nQ8 CodeAgent Response:"
)
print(response_code)


# Q8 Reflection:
#
# The ToolCallingAgent used plot_data to create the scatter plot.
# The tool does not have a color option, so it could not make the
# points green.
#
# The CodeAgent first used plot_data, but then it wrote its own
# matplotlib code and used color="green". It also saved the plot
# to outputs/q8_codeagent.png.
#
# This showed me that ToolCallingAgent works well when the task fits
# the tools it already has. CodeAgent is more useful when the task
# needs extra customization that the tools do not support.


# Q9

# 1. I would use a ToolCallingAgent for a task with clear and limited
# actions, like loading a CSV, checking its columns, or looking up data
# through an API. It is a good fit when the tools already cover everything
# the agent needs to do.
#
# 2. A CodeAgent can generate and execute new code, so the code could have
# a bug or do something unexpected. A ToolCallingAgent is more limited
# because it can only call the tools the developer already provided.