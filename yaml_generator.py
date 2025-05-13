import yaml
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, MetaData, select, func, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import datetime
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_database(connection_uri):
    """
    Establish connection to SQL Server database using the provided URI
    """
    try:
        # Check if the connection string already has the SQLAlchemy prefix
        if "mssql" not in connection_uri and "pyodbc" not in connection_uri:
            # If it doesn't have the SQLAlchemy prefix, add it
            connection_string = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_uri)}"
        else:
            connection_string = connection_uri
            
        engine = create_engine(connection_string)
        connection = engine.connect()
        logger.info(f"Successfully connected to database")
        return engine, connection
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to SQL Server database: {str(e)}")
        raise

def get_database_structure(engine):
    """
    Extract database structure including tables, columns, primary keys, and foreign keys
    """
    inspector = inspect(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    db_structure = {}
    
    # Get schema names (SQL Server specific)
    schemas = inspector.get_schema_names()
    
    for schema in schemas:
        # Skip system schemas
        if schema in ('sys', 'INFORMATION_SCHEMA', 'guest', 'db_owner', 'db_accessadmin', 
                      'db_securityadmin', 'db_ddladmin', 'db_backupoperator', 'db_datareader', 
                      'db_datawriter', 'db_denydatareader', 'db_denydatawriter'):
            continue
            
        for table_name in inspector.get_table_names(schema=schema):
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
    
    return db_structure

def get_sample_data(connection, schema, table_name, columns):
    """
    Get a sample row of data from the specified table
    """
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
        logger.warning(f"Could not get sample data for table {schema}.{table_name}: {str(e)}")
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

def generate_db_yaml(connection_uri, output_file=None):
    """
    Main function to generate YAML schema from SQL Server database
    Returns the db_structure dictionary by default
    If output_file is provided, also writes to that file
    If return_yaml_string=True, also returns the YAML as a string
    """
    try:
        # Connect to the database
        engine, connection = connect_to_database(connection_uri)
        
        # Get database structure
        db_structure = get_database_structure(engine)
        
        # Get sample data for each table
        for full_table_name, table_info in db_structure.items():
            schema = table_info['schema']
            table = table_info['table']
            columns = table_info['columns']
            
            sample_data = get_sample_data(connection, schema, table, columns)
            db_structure[full_table_name]['sample_data'] = sample_data
        
        # Generate YAML file if output_file is provided
        if output_file:
            success = generate_yaml_string(db_structure)
            logger.info(f"YAML file generated: {output_file}")
        
        # Close connection
        connection.close()
        engine.dispose()
        
        logger.info("SQL Server database schema extraction completed successfully.")
        
        # Return the db_structure dictionary
        return db_structure
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None

def generate_yaml(db_uri):

    # Call the function directly
    result = generate_db_yaml(db_uri)

    return result