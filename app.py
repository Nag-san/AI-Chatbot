import os
import requests
import logging
import pandas as pd
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from flask_cors import CORS
from langgraph.graph import StateGraph, state

app = Flask(__name__)
CORS(app)

#Database Config
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqlconnector://root:root@localhost/Company_DB"
app.config["SQLACHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

HUGGINGFACE_API_KEY = "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"
MODEL_NAME = "tiiuae/falcon-7b-instruct"

def query_huggingface(user_query):
    """
    Converts a natural language query into an SQL query using Hugging Face API.
    """
    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    prompt = f"""
    You are an AI that converts natural language into SQL queries.

ONLY return the SQL query—no explanations, formatting, or extra text.

Database Schema:
Products Table:
Columns:
product_id (BIGINT, PRIMARY KEY, AUTO_INCREMENT)
name (VARCHAR)
price (DECIMAL)
category (VARCHAR)
supplier_id (INT, FOREIGN KEY to Suppliers)

Suppliers Table:
Columns:
supplier_id (INT, PRIMARY KEY)
name (VARCHAR)
contact_name (VARCHAR)
phone (VARCHAR)
email (VARCHAR)
location (TEXT)


Now, convert this query into SQL:
User Query: {user_query}
SQL Query:
    """

    payload = {"inputs": prompt}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    else:
        return None  # Handle errors later

def execute_sql(sql_query):
    """
    Executes the SQL query and converts results into a CSV string.
    """
    try:
        result = db.session.execute(text(sql_query))
        rows = [dict(row) for row in result.mappings()]

        if rows:
            df = pd.DataFrame(rows)
            csv_content = df.to_csv(index=False)
        else:
            csv_content = "No data returned."

        return csv_content, None  # No errors
    except Exception as e:
        return None, str(e)  # Return error message

def summarize_csv(csv_content):
    """
    Uses AI to summarize the CSV results into a short sentence.
    """
    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    prompt = f"""
    The following CSV contains results of an SQL query:
    ```{csv_content}```
    Summarize the key insights in one short sentence:
    """
    
    payload = {"inputs": prompt}
    response = requests.post(url, json=payload, headers=headers)
    app.logger.info(response)
    print(response, flush=True)
    
    if response.status_code == 200:
        result = response.json()[0]["generated_text"].strip()
        # Extract ONLY the SQL query (remove explanations)
        sql_lines = result.split("\n")
        clean_sql = ""

        for line in sql_lines:
            if line.strip().upper().startswith("SELECT") or line.strip().upper().startswith("INSERT") or e.strip().upper().startswith("UPDATE") or line.strip().upper().startswith("DELETE"):
                clean_sql = line.strip()
                clean_sql = clean_sql.lstrip("SQL:")
                break

        return clean_sql if clean_sql else "ERROR: AI response did not contain a valid SQL query."
    else:
        return None      
    

# =================== Flask Routes ===================
@app.route("/query", methods=["POST"])
def chatbot():
    """
    Processes user queries by:
    1. Converting natural language to SQL using AI.
    2. Executing SQL in the database.
    3. Converting results to CSV.
    4. Summarizing the CSV using AI.
    """
    data = request.json
    user_query = data.get("query")

    # Step 1: Generate SQL Query
    sql_query = query_huggingface(user_query)
    if not sql_query or "ERROR" in sql_query:
        return jsonify({"error": "Failed to generate SQL query."})

    # Step 2: Execute SQL Query
    sql_lines = sql_query.split("\n")
    clean_sql = ""
    for line in sql_lines:
        if line.strip().upper().startswith("SELECT") or line.strip().upper().startswith("INSERT") or line.strip().upper().startswith("UPDATE") or line.strip().upper().startswith("DELETE"):
            clean_sql = line.strip()
            break
 
    csv_content, error = execute_sql(clean_sql)
    if error:
        return jsonify({"error": error})

    # Step 3: Summarize the CSV
    summary = summarize_csv(csv_content)

    return jsonify({
        "user_query": user_query,
        "sql_query": sql_query,
        "csv_content": csv_content,
        "summary": summary
    })

@app.route("/")
def home():
    return jsonify({"message": "Chatbot API is running."})

if __name__ == "__main__":
    app.run(debug=True)
