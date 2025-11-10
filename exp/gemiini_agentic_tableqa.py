import json
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import LLMMathChain
from time import sleep 
import pandas as pd
from tqdm import tqdm
import io

api_keys = [
]

table_context = None
sleep_duration = 0

llm = GoogleGenerativeAI(
	model="gemini-1.5-pro-002",
	# model="gemini-2.0-flash-exp",
	google_api_key=api_keys[0],
	temperature=0.3,
	verbose = False
)

def create_safe_python_tool():
	def safe_exec(code: str) -> str:
		try:
			df = pd.read_csv(io.StringIO(table_context))
			local_vars = {"df": df, "pd": pd}
			code = code.replace("python", "")
			code = code.replace("Python", "")
			code = code.replace("```", "")
			exec(code, local_vars)
			sleep(sleep_duration)
			# Get the result from the last expression
			result = local_vars.get('result', None)
			return str(result)
		except Exception as e:
			return f"Error: {str(e)}"

	return Tool(
		name="Python",
		func=safe_exec,
		description="Execute Python code for data analysis and calculations. The data is available as 'df' and pandas is imported as 'pd'. Store your final answer in a variable named 'result'. Do not use print statements. Do not import pandas as it is already imported. Do not create the dataframe as it is already present. Example: result = df['column'].mean()"
	)

tools = [
	# math_tool,
	create_safe_python_tool()
]

class DelayCallbackHandler:
	def on_llm_start(self, *args, **kwargs):
		sleep(sleep_duration)
		pass

agent = initialize_agent(
	tools=tools,
	llm=llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=False,
	callback_handlers=[DelayCallbackHandler()]
)

prompt = """You are an expert at answering questions based on tabular data. You have access to a math LLM tool to help with calculations. Your task is to analyze a given table and a question to provide the correct answer. Follow these steps meticulously:
Understand the Table: Examine the table's column headers to understand the meaning of each column. Identify the data types within each column (e.g., numerical, text, date).
Interpret the Question: Carefully read the question to identify the information requested and the relationships between the data it implies. Break down the question into steps if necessary. Determine which columns are relevant for answering the question.
Reasoning: Act as a reasoning agent and determine a strategy to solve the question.

You have access to a python tool:
Python_REPL: For data analysis using pandas and for performing calculations. Pandas is already imported as pd, whereas, the data is already available as 'df'. Do not use print statements.

- Use Python for operations on the data like:
  - Arithmetic operations
  - Aggregations (mean, sum, count)
  - Filtering
  - Grouping
  - Complex calculations
  - Other operations that 
  
In terms of decimal numbers, always round to two decimal places.

Combine several steps to solve the problem correctly.
You can always derive the answer from the table.
Think step by step and make sure to provide the correct answer to the question.

If you are not able to answer, return 'none'.
Format your final answer concisely."""

placeholder = """
table:
{table}

question: {question}"""

def process_query(table, question):
	global table_context
	table_context = table  # Make table available to Python tool
	try:
		full_prompt = prompt + placeholder.format(table=table, question=question)
		response = agent.run(full_prompt)
		if response == 'none':
			print("\nAnswering none")
			return None
		return response
	except Exception as e:
		print("ERROR ERROR ERROR")
		print(e)
		return None

for annotator in ['srija', 'adnan']:
	print(annotator[0], "verification")
	df_big = pd.read_json(f"../../{annotator}_verification/complete_df.json")
	if 'answer' not in df_big.columns:
		df_big['answer'] = None
	null_rows = df_big[df_big['answer'].isna()]
	print(f"Remaining rows to process: {len(null_rows)}")
	
	if len(null_rows) == 0:
		print("All answers completed!")
		continue
		
	for idx, row in tqdm(null_rows.iterrows(), total= len(null_rows)):
		try:
			# llm.google_api_key = api_keys[0]
			llm.google_api_key = api_keys[idx % len(api_keys)]
			# llm.google_api_key = api_keys[idx % 3]
			question = row['question']
			table = open(f"../../{annotator}_verification/my_data/{row['chart_name']}/charts/data.csv").read()
			# print("question:", question)
			response = process_query(table, question)
			# print("response:", response)
			df_big.loc[idx, 'answer'] = response
			
			df_big.to_json(f"../../{annotator}_verification/complete_df.json", orient='records')

		except Exception as e:
			print(f"Error processing row {idx}: {str(e)}")
			continue