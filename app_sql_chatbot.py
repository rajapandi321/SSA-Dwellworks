import streamlit as st
from sqlalchemy import create_engine, text, inspect
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os, re
import pandas as pd 
from urllib.parse import quote_plus
import numpy as np
import json
import yaml
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, MetaData, select, func, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import datetime
import time
from urllib.parse import quote_plus
import traceback
from visual.app_visual import create_visualization
from example_prompt import example_prompt_template

# Import AWS Bedrock
from langchain_community.chat_models import BedrockChat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit UI Configuration
st.set_page_config(page_title="SQL Server Explorer", layout="wide")

def connect_to_database(connection_uri):
    """
    Establish connection to SQL Server database using the provided URI
    """
    try:
        # Check if the connection string already has the SQLAlchemy prefix
        if "mssql" not in connection_uri and "pyodbc" not in connection_uri:
            # If it doesn't have the SQLAlchemy prefix, add it
            connection_string = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_uri)}"
        else:
            connection_string = connection_uri
            
        engine = create_engine(connection_string)
        connection = engine.connect()
        logger.info(f"Successfully connected to database")
        return engine, connection
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to SQL Server database: {str(e)}")
        raise

def get_database_structure(engine, progress_bar=None, max_tables=None):
    """
    Extract database structure including tables, columns, primary keys, and foreign keys
    With optional progress bar and table limit for large databases
    """
    inspector = inspect(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    db_structure = {}
    
    # Get schema names (SQL Server specific)
    schemas = inspector.get_schema_names()
    
    # Filter out system schemas
    user_schemas = [schema for schema in schemas if schema not in (
        'sys', 'INFORMATION_SCHEMA', 'guest', 'db_owner', 'db_accessadmin', 
        'db_securityadmin', 'db_ddladmin', 'db_backupoperator', 'db_datareader', 
        'db_datawriter', 'db_denydatareader', 'db_denydatawriter'
    )]
    
    # Get all tables across all user schemas
    all_tables = []
    for schema in user_schemas:
        try:
            schema_tables = [(schema, table_name) for table_name in inspector.get_table_names(schema=schema)]
            all_tables.extend(schema_tables)
        except Exception as e:
            logger.warning(f"Error getting tables for schema {schema}: {str(e)}")
    
    # Limit tables if specified
    if max_tables and len(all_tables) > max_tables:
        logger.warning(f"Limiting schema extraction to {max_tables} tables out of {len(all_tables)}")
        all_tables = all_tables[:max_tables]
    
    # Update progress bar total if provided
    if progress_bar:
        progress_bar.total = len(all_tables)
    
    # Process each table
    for i, (schema, table_name) in enumerate(all_tables):
        try:
            full_table_name = f"{schema}.{table_name}"
            
            db_structure[full_table_name] = {
                'schema': schema,
                'table': table_name,
                'columns': {},
                'primary_key': [],
                'foreign_keys': [],
                'relationships': [],
                'sample_data': {}
            }
            
            # Get column information
            for column in inspector.get_columns(table_name, schema=schema):
                col_name = column['name']
                col_type = str(column['type'])
                is_nullable = column.get('nullable', True)
                default = column.get('default', None)
                
                # Convert default value to string if it exists
                if default is not None:
                    default = str(default)
                
                db_structure[full_table_name]['columns'][col_name] = {
                    'type': col_type,
                    'nullable': is_nullable,
                    'default': default
                }
            
            # Get primary key information
            try:
                pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
                if pk_constraint and 'constrained_columns' in pk_constraint:
                    pk_columns = pk_constraint.get('constrained_columns', [])
                    db_structure[full_table_name]['primary_key'] = pk_columns
            except Exception as e:
                logger.warning(f"Could not get primary key for {schema}.{table_name}: {str(e)}")
                db_structure[full_table_name]['primary_key'] = []
            
            # Get foreign key information
            try:
                for fk in inspector.get_foreign_keys(table_name, schema=schema):
                    referred_schema = fk.get('referred_schema', schema)
                    
                    fk_info = {
                        'name': fk.get('name', 'unnamed_fk'),
                        'columns': fk['constrained_columns'],
                        'referred_schema': referred_schema,
                        'referred_table': fk['referred_table'],
                        'referred_columns': fk['referred_columns']
                    }
                    db_structure[full_table_name]['foreign_keys'].append(fk_info)
                    
                    # Add relationship information
                    referred_full_table = f"{referred_schema}.{fk['referred_table']}"
                    relationship = {
                        'type': 'many_to_one',  # Default assumption
                        'target_table': referred_full_table,
                        'source_columns': fk['constrained_columns'],
                        'target_columns': fk['referred_columns']
                    }
                    db_structure[full_table_name]['relationships'].append(relationship)
            except Exception as e:
                logger.warning(f"Could not get foreign keys for {schema}.{table_name}: {str(e)}")
            
            # Update progress bar if provided
            if progress_bar:
                progress_bar.progress((i + 1) / len(all_tables), text=f"Processing table {i+1}/{len(all_tables)}: {full_table_name}")
                
        except Exception as e:
            logger.error(f"Error processing table {schema}.{table_name}: {str(e)}")
            # Continue with next table instead of failing completely
    
    return db_structure

def get_sample_data(connection, schema, table_name, columns, max_retries=3):
    """
    Get a sample row of data from the specified table with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            # SQL Server specific query to get top 1 row - using try/except handling
            safe_table_name = table_name.replace("'", "''")
            safe_schema = schema.replace("'", "''")
            
            query = text(f"SELECT TOP 1 * FROM [{safe_schema}].[{safe_table_name}]")
            result = connection.execute(query).fetchone()
            
            if result:
                # Create a dictionary with column names and values
                sample_data = {}
                
                # Get column names from the result
                if hasattr(result, '_fields'):
                    # For Row objects that have _fields attribute
                    column_names = result._fields
                elif hasattr(result, 'keys'):
                    # For result objects with keys() method
                    column_names = result.keys()
                else:
                    # Fallback to using the column keys directly
                    column_names = list(columns.keys())
                
                # Process each column
                for idx, col_name in enumerate(column_names):
                    try:
                        # Try to get value by name
                        if hasattr(result, '_mapping'):
                            value = result._mapping.get(col_name)
                        else:
                            # Fallback to index-based access
                            value = result[idx]
                        
                        # Handle SQL Server specific data types
                        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
                            sample_data[col_name] = value.isoformat()
                        elif isinstance(value, bytes):
                            sample_data[col_name] = f"<BINARY DATA: {len(value)} bytes>"
                        elif not isinstance(value, (str, int, float, bool, type(None))):
                            sample_data[col_name] = str(value)
                        else:
                            sample_data[col_name] = value
                    except Exception as e:
                        logger.warning(f"Error accessing column {col_name} in {schema}.{table_name}: {str(e)}")
                        sample_data[col_name] = None
                        
                return sample_data
            else:
                logger.info(f"No sample data found for table {schema}.{table_name}")
                return {}
        except SQLAlchemyError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt+1}/{max_retries} failed for {schema}.{table_name}: {str(e)}")
                time.sleep(1)  # Adding a short delay before retry
            else:
                logger.warning(f"Could not get sample data for table {schema}.{table_name} after {max_retries} attempts: {str(e)}")
                return {}

def generate_yaml_string(db_structure):
    """
    Generate YAML string from the database structure
    """
    try:
        yaml_string = yaml.dump(db_structure, default_flow_style=False, sort_keys=False)
        return yaml_string
    except Exception as e:
        logger.error(f"Failed to generate YAML string: {str(e)}")
        raise

def generate_db_yaml(connection_uri, output_file=None, max_tables=None):
    """
    Main function to generate YAML schema from SQL Server database
    Returns the db_structure dictionary by default
    If output_file is provided, also writes to that file
    Added max_tables parameter to limit schema extraction for large databases
    """
    engine = None
    connection = None
    
    try:
        # Connect to the database
        engine, connection = connect_to_database(connection_uri)
        
        # Create a progress bar for tracking schema extraction
        progress_bar = st.progress(0, text="Starting schema extraction...")
        
        # Get database structure with progress bar
        db_structure = get_database_structure(engine, progress_bar, max_tables)
        
        # Get sample data for each table
        progress_bar.progress(0, text="Collecting sample data...")
        total_tables = len(db_structure)
        
        for i, (full_table_name, table_info) in enumerate(db_structure.items()):
            schema = table_info['schema']
            table = table_info['table']
            columns = table_info['columns']
            
            # Update progress
            progress_bar.progress((i + 1) / total_tables, 
                                 text=f"Collecting sample data: {i+1}/{total_tables} - {full_table_name}")
            
            sample_data = get_sample_data(connection, schema, table, columns)
            db_structure[full_table_name]['sample_data'] = sample_data
        
        # Generate YAML file if output_file is provided
        if output_file:
            success = generate_yaml_string(db_structure)
            logger.info(f"YAML file generated: {output_file}")
        
        progress_bar.empty()  # Remove progress bar when done
        
        logger.info("SQL Server database schema extraction completed successfully.")
        
        # Return the db_structure dictionary
        return db_structure
        
    except Exception as e:
        logger.error(f"An error occurred during schema extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    finally:
        # Close connection
        if connection:
            connection.close()
        if engine:
            engine.dispose()

def generate_yaml(db_uri, max_tables=None):
    # Call the function with optional table limit
    result = generate_db_yaml(db_uri, max_tables=max_tables)
    return result

def create_connection_string(user, password, server_name, database):
    """
    Create a properly encoded ODBC connection string
    """
    return f"mssql+pymssql://{user}:{quote_plus(password)}@{server_name}/{database}"

def create_agent(llm, yaml_output):
    """
    Create an AI agent for generating SQL queries without SQLDatabaseToolkit
    """
    system_prompt = '''Your task is to generate a **correct T-SQL query** based on the user input. 
                    You have access to this YAML database schema:
                    ```yaml
                    {yaml_output}
                    ```
                        
                    Steps: 
                        - **Analyze all available tables and YAML database schema** and their relationships.  
                        - **Ensure the query is free of syntax errors.**  
                        - **Use the appropriate T-SQL syntax based on the YAML input** for the given database.  
                        - **Return only the T-SQL query** without any explanation.  

                    Also You have access to this Example Queries: {example_prompt_template}. Please analyze the question carefully relevant to the Schema and give me results accordingly.

                    **Ensure T-SQL Query you are generating should be relevant to the YAML file**
                    **Carefully Check all Column names that are specific to tables and do Joins if required between tables**

                    **If any Questions asked related to Bed Rooms instead of '1 BR', '2 BR', There is no values in the database like 1BR, 2BR. You should return '1 Bedroom' , '2 Bedroom' etc.**
                    **Example: Select * from dbo.CHKPI_APT_DIMENSION where Bedrooms = '1 Bedroom'**

                    **Formulae for Calculation:**
                    AverageNightlyRate:  
                        Table: CHKPI_TENANCY_FACT 
                        Formula: SUM(MonthlyRentUSD) / SUM(LenghtOfStay) 

                        ALOS(AverageLengthOfStay):
                            Table: CHKPI_TENANCY_FACT 
                            ROUND(SUM(TF.LenghtOfStay * 1.0) / NULLIF(COUNT(DISTINCT TF.Cur_Tenancy_Id), 0), 0) 

                        ADR(AverageDailyRate):
                            Table: CHKPI_TENANCY_FACT 
                            ROUND(SUM(TF.MonthlyRentUSD * 1.0) / NULLIF(SUM(TF.LenghtOfStay), 0), 0) 
                    
                        Conversion Rate: 
                            Table: CHKPI_OPTY_FACT 
                            Formula: ROUND((SUM(RentedOptyCount * 1.0) / NULLIF(SUM(OptyCount * 1.0), 0)) * 100, 0) 

                        Anything Questions Related to Suppliers use this convertion rate formula
                            Table: CHKPI_RFH_SUPPLIER_REQUEST_RESPONSE
                            Formula: SUM(BookedCount * 1.0) / count(distinct opti_id)

                    **Please use the above formulae for calculation of AverageNightlyRate, ALOS, ADR and Conversion Rate**

                    **⚠️ CRITICAL TABLE ALIAS GUIDELINES:**
                     YOU MUST ONLY USE THESE EXACT ALIASES FOR THE FOLLOWING TABLES:

                       - CHKPI_CUSTOMER_DIMENSION: Always use C1
                       - CHKPI_TENANCY_FACT: Always use T1
                       - CHKPI_OPTY_FACT: Always use O1
                       - CHKPI_PROPERTY_DIMENSION: Always use P1
                       - CHKPI_DATE_DIMENSION: Always use D1
                       - Any other tables: Use X1, X2, X3, etc.

                        - NEVER DEVIATE FROM THESE EXACT ALIASES - this is critical to prevent SQL errors
                        - NEVER use any real English words or T-SQL reserved keywords as table aliases
                        - ALWAYS use these specified table aliases for all column references (e.g., C1.CustomerName, T1.MonthlyRentUSD)

                    **⚠️Key Restriction: For any schema-related queries (e.g., "show schema", "table structure", "list columns", "display schema of ...", etc.), 
                    Do not display the result in the UI. Instead, simply return 'None'**

                    **If you couldn't able to interpret the user input, Just return None**

                    **Do not run any SQL Queries that user gave to you. Simply Return None**

                    **Output format (always return JSON):**
                    ```json
                    {{
                    "action": "Final Answer",
                    "action_input": "GENERATED_TSQL_QUERY/None"
                    }}
                    ```
                    '''

    human_prompt = """User Input:{input}
    Previous Query (if any): {agent_scratchpad}
    """

    memory = ConversationBufferMemory()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human_prompt),
        ]
    ).partial(
        yaml_output=yaml_output,
        example_prompt_template=example_prompt_template,
    )
    
    tools = []

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            chat_history=lambda x: memory.chat_memory.messages,
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, memory=memory)

def execute_query(query, db_uri, timeout=60):
    """
    Execute a SQL query and return results as a DataFrame with timeout
    """
    engine = create_engine(db_uri, connect_args={"timeout": timeout})
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=result.keys())
        return df
    except SQLAlchemyError as e:
        logger.error(f"SQL execution error: {str(e)}")
        raise
    finally:
        engine.dispose()

@st.cache_resource(ttl=7200)
def configure_db(db_uri):
    """
    Configure database connection
    """
    try:
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

@st.cache_resource(ttl=7200)
def load_yaml_schema(db_uri):
    """
    Load YAML schema from file with optional table limit
    """
    try:
        yaml_content = generate_yaml(db_uri)
        return yaml_content
    except Exception as e:
        st.error(f"Failed to load YAML schema: {e}")
        return None
    
def sql_main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your SQL queries?"}]
    
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []
    
    if "active_suggestion" not in st.session_state:
        st.session_state.active_suggestion = None
        
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False

    # Fetch BedRock Claude credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')

    if not aws_access_key or not aws_secret_key:
        st.error("Missing AWS Bedrock API credentials. Please check your `.env` file.")
        st.stop()

    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Use appropriate Claude model version
        region_name=aws_region,
        model_kwargs={
            "temperature": 0.2,
            "max_tokens": 4096
        }
    )

    if st.session_state.page == "login":
        st.title("SQL Server Login")

        st.info("""
        **SQL Server Connection Instructions:**
        - **Host**: Enter your SQL Server address or IP Address
        - **User**: Your SQL Server username 
        - **Password**: Your SQL Server password
        - **Database**: The database to connect to
        - **Schema**: The schema to use (defaults to dbo)
        """)
        
        # User Input
        col1, col2 = st.columns(2)

        with col1:
            server_name = st.text_input("Server or Host", value="", placeholder="Enter Server Name")
            database = st.text_input("Database", value="", placeholder="Enter Default Database")
            schema = st.text_input("Schema", value="dbo", placeholder="Enter Schema Name (default: dbo)")   
            
        with col2:
            user = st.text_input("User Name", value="", placeholder="Enter SQL Server User")        
            password = st.text_input("Password", value="", placeholder="Enter Password", type="password")
  
        if st.button("Connect"):
            if not all([server_name, user, password, database]):
                st.error("Please fill in all required fields.")
            else:
                try:
                    with st.spinner("Connecting to database..."):
                        # Create connection string
                        db_uri = create_connection_string(user, password, server_name, database)
                        
                        # Show success message before the potentially long YAML generation
                        st.success(f"Successfully connected to {database} on {server_name}")
                        
                        # Generate YAML schema with progress indication
                        st.info("Generating database schema... This may take a while for large databases.")
                        yaml_output = load_yaml_schema(db_uri)
                        
                        if not yaml_output:
                            st.error("Failed to generate database schema. Please check your connection.")
                            st.stop()
                        
                        # Store connection details in session state
                        st.session_state.db_uri = db_uri
                        st.session_state.yaml_output = yaml_output
                        st.session_state.page = "dashboard"
                        st.success("Schema loaded successfully! Redirecting to dashboard...")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    st.error(traceback.format_exc())

    # Dashboard Page
    elif st.session_state.page == "dashboard":
        st.title("Chat with your SQL Database")
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                if key != "page":
                    del st.session_state[key]
            st.session_state.page = "login"
            st.rerun()

        # Retrieve stored connection details
        db_uri = st.session_state.get('db_uri')
        yaml_output = st.session_state.get('yaml_output')

        if not db_uri:
            st.error("No active database connection. Please log in.")
            st.session_state.page = "login"
            st.rerun()

        if yaml_output:
            # Count tables for information
            st.success(f'YAML schema loaded successfully!')
        else:
            st.error('Please check your database connection and schema')
            st.session_state.page = "login"
            st.rerun()

        # Configure database
        db_engine = configure_db(db_uri)

        # Initialize agent
        agent_executor = create_agent(llm, yaml_output)

        # Clear chat history button
        if st.sidebar.button("Clear message history"):
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your SQL queries?"}]
            st.rerun()

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Get user input
        user_query = st.chat_input(placeholder="Ask anything from the database...")

        if user_query:
            # Add user query to messages and display it
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
                        
            # Extract output from response
            with st.chat_message("assistant"):
                try:
                    # Create a spinner while processing
                    with st.spinner("Analyzing your query..."):
                        translator_prompt = f'''
                                You are an expert multilingual translator with deep knowledge of natural language processing. 
                                Your task is to translate the given text into English **only if it is in a different language**. 
                                If the text is already in English, return it as it is without any modifications.  

                                Ensure that the translation preserves the original meaning, intent, and context, especially for technical terms and database-related phrases.  

                                Text to process: {user_query}  
                                '''
                                
                        translated_response = llm.invoke(translator_prompt)
                        translated_query = translated_response.content

                        rephrase_prompt = f'''You are an expert database query assistant analyzing a user query against a specific database schema.

                                Context:
                                - YAML Schema: {yaml_output}
                                - Original User Query: {translated_query}

                                Your Tasks:
                                1. First, carefully assess if the user query is relevant to the database schema.

                                2. If the query is NOT relevant to the database:
                                - You should generate 3 sample questions that are:
                                    a) Contextually related to the user's original query
                                    b) Directly answerable using the current database schema
                                    c) Phrased in a general, informative manner
                                    d) Do NOT mention specific table or column names
                                    e) Closely aligned with the intent of the original query

                                3. If the query IS relevant to the database, no further action is needed.

                                Output Instructions:
                                - For irrelevant queries: Provide 3 generalized, schema-appropriate questions
                                - Ensure questions are clear, concise, and meaningful
                                - Avoid technical jargon
                                - Focus on the underlying information need

                                Example:
                                If user asks about "employee salaries in New York" and the schema is about product sales, 
                                suggested questions might be like:
                                - What insights can I get about our sales performance?
                                - How do our product categories compare in revenue?
                                - Can you show me trends in our sales data?
                                '''
                        suggested_response = llm.invoke(rephrase_prompt)
                        suggested_query = suggested_response.content

                        response = agent_executor.invoke({"input": translated_query})

                        # Extract the SQL query from the response
                        extracted_query = None
                        
                        if isinstance(response, dict) and "output" in response:
                            output_content = response["output"]
                            
                            # Handle different output formats
                            if isinstance(output_content, dict) and "action" in output_content:
                                if output_content["action"] == "Final Answer" and output_content.get('action_input') != "None":
                                    extracted_query = output_content.get("action_input")
                            elif isinstance(output_content, str) and output_content.strip() != "None":
                                extracted_query = output_content

                    # Store assistant's response in session state
                    response_message = "I've analyzed your query and generated SQL."
                    st.session_state.messages.append({"role": "assistant", "content": response_message})
                    st.write(response_message)

                    if not extracted_query or extracted_query == "None":
                        if "1." in suggested_query:
                            # Extract the database-related suggestions
                            suggestions_start = suggested_query.find("1.")
                            suggestions = suggested_query[suggestions_start:].split("\n")[0:3]
                            
                            st.warning("The query does not seem to be related to the current database schema. Here are some suggested questions:")
                            for suggestion in suggestions:
                                if suggestion.strip():
                                    st.write(f"- {suggestion.strip()}")
                        else:
                            st.warning("I couldn't generate a SQL query for this request. Please rephrase or try a different question.")
                    else:
                        with st.expander('Generated SQL Query'):
                            st.code(extracted_query, language='sql')

                        try:
                            # Execute SQL Query with progress indication
                            with st.spinner("Executing query..."):
                                df = execute_query(extracted_query, db_uri)

                            # Display results in Streamlit
                            if not df.empty:
                                result_count = len(df)
                                st.write(f"Found {result_count} results:")
                                
                                # Enable CSV download
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="Download results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv",
                                )

                                
                                # # Display results with appropriate visualizations
                                # if len(df.columns) > 1:
                                #     # If we have many rows, offer pagination
                                #     if result_count > 100:
                                #         page_size = 100
                                #         page_number = st.number_input("Page", min_value=1, max_value=(result_count // page_size) + 1, value=1)
                                #         start_idx = (page_number - 1) * page_size
                                #         end_idx = min(start_idx + page_size, result_count)
                                #         df_display = df.iloc[start_idx:end_idx]
                                #         st.write(f"Showing results {start_idx+1}-{end_idx} of {result_count}")
                                #     else:
                                        # df_display = df

                                    # Get column types for better visualization decisions
                                    
                            choose_visual_prompt = f'''You are a data visualization expert who analyzes dataframe to determine the optimal X and Y axis and suitable visualization for the dataframe for insightful visualizations.

                            ```Dataframe to analyze: {df}```

                            INSTRUCTIONS:
                            1. Carefully examine the data in the provided dataframe {df}
                            2. Select the most meaningful X-axis and Y-axis pairing that would create an insightful visualization
                            3. Return your selection in JSON format only

                            X-AXIS SELECTION PRIORITIES (in order):
                            1. Time-related columns (date, month, quarter, time, period, etc.)
                            2. Geographic columns (country, region, state, city, location, etc.)
                            3. Categorical variables (product, category, type, group, department, etc.)
                            4. Entity identifiers (name, ID, etc.) if the list is reasonably small
                            5. Ordinal variables with natural ordering (size_category, age_group, etc.)

                            Y-AXIS SELECTION PRIORITIES (in order):
                            1. Numerical metrics or KPIs (sales, revenue, profit, count, etc.)
                            2. Performance indicators (score, rating, performance, etc.)
                            3. Quantities or measurements (amount, volume, weight, duration, etc.)
                            4. Rates or percentages (rate, percentage, ratio, etc.)
                            5. Any other numerical columns that would provide meaningful insights

                            **VISUALIZATION TYPE SELECTION RULES:**
                            - Line chart: For time series data or showing trends over a continuous variable
                            - Bar chart: For comparing categorical data or showing ranking
                            - Scatter plot: For exploring relationships between two numerical variables
                            - Histogram: For distribution of a single numerical variable
                            - Heatmap: For correlation between variables or multi-dimensional categorical data
                            - Box plot: For showing distribution and outliers across categories
                            - Pie chart: Only for composition of a whole when there are few categories (<7)
                            - Area chart: For cumulative totals over time or stacked composition changes
                            - Bubble chart: When a third numerical dimension is important
                            - Grouped/stacked bar: For comparing multiple categories or subcategories

                            ADDITIONAL RULES:
                            - Never suggest the same column for both X and Y axes
                            - Always prefer numeric columns for Y-axis
                            - If multiple time columns exist, choose the one with appropriate granularity
                            - Consider natural relationships between variables (e.g., time and sales)
                            - Avoid using unique identifiers as X-axis if there are better categorical options

                            **Output format:**
                            **Output format (always return JSON):**
                            ```json
                                {{
                                "x_axis": <Predictred X-axis column name>,
                                "y_axis": <Predicted Y-axis column name>,   
                                "visualization": <Suggested visualization type>
                                }}
                    ```
                            '''

                            # Get the response from the LLM
                            identified_columns = llm.invoke(choose_visual_prompt)

                            # Extract JSON from the LLM response
                            import re
                            import json

                            def extract_json_from_text(text):
                                """Extract a JSON object from text that might contain additional content."""
                                # First try to parse the whole text as JSON
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    pass
                                
                                # Try to find JSON object within text
                                try:
                                    # Look for text between curly braces
                                    json_match = re.search(r'\{[\s\S]*?\}', text)
                                    if json_match:
                                        return json.loads(json_match.group(0))
                                except:
                                    pass
                                
                                # Try to find JSON object within code blocks (common in markdown responses)
                                try:
                                    # Look for text between ```json and ```
                                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                                    if json_match:
                                        return json.loads(json_match.group(1))
                                except:
                                    pass
                                
                                return None

                            # Try to extract JSON from the response
                            try:
                                response_text = identified_columns.content
                                x_y_columns = extract_json_from_text(response_text)
                                
                                if not x_y_columns:
                                    st.error(f"Could not extract valid JSON from the response. Raw response: {response_text}")
                                    st.stop()
                                    
                                x_axis = x_y_columns.get("x_axis")        
                                y_axis = x_y_columns.get("y_axis")
                                suggested_visual = x_y_columns.get("visualization")
                                
                                if not x_axis or not y_axis:
                                    st.warning("Unable to identify X and Y axis for visualization.")
                                    st.stop()
                            except Exception as e:
                                st.error(f"Error processing LLM response: {str(e)}")
                                st.stop()
                                
                            data_tab, visual_tab = st.tabs(["Data", "Visualization"])

                            with data_tab:
                                st.dataframe(df, use_container_width=True)
                                
                            # Prepare chart data
                            try:
                                with visual_tab:
                                    if df.shape[1] <= 10:
                                        fig = create_visualization(df, x_axis, y_axis, suggested_visual)
                                        if fig:
                                            st.pyplot(fig)
                                        else:
                                            st.warning("Unable to create visualization with the current data.")
                                    else:
                                        st.warning("Dataframe contains too many columns for visualization.")
                                                
                            except Exception as chart_error:
                                with visual_tab:
                                    st.info(f"Cannot visualize this data: {str(chart_error)}")
                        except SQLAlchemyError as e:
                            st.error(f"SQL execution error: {str(e)}")
                            st.error(traceback.format_exc())
                            st.warning("Unable to execute the SQL query. Please check the syntax.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(traceback.format_exc())
                    st.warning("Unable to process your request. Please try again.")

    # Sidebar usage instructions
    st.sidebar.markdown("### How to Use")
    st.sidebar.info("""
    - Ask questions about your database in natural language.
    - Review the generated SQL query.
    - Your Results will be generated in Table format.
    - Download results as CSV if needed.
    - Access previous queries from the history.
    """)

    # Update sidebar model information to correctly reference AWS Bedrock
    st.sidebar.markdown("### Model Information")
    st.sidebar.info("""
    Using AWS Bedrock
    - Model: Claude 3.5 Sonnet
    - Version: 2024-06-20
    """)

if __name__ == "__main__":
    sql_main()
