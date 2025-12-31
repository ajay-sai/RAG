import streamlit as st

st.set_page_config(layout="wide", page_title="PostgreSQL Viewer")

st.title("PostgreSQL Database Schema Viewer")

# Initialize connection using st.connection
conn = st.connection("postgresql", type="sql")

# Function to get all schemas
@st.cache_data(ttl="10m")
def get_schemas():
    # Query information_schema.schemata to get all schema names
    df_schemas = conn.query("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema') AND schema_name NOT LIKE 'pg_toast%';")
    return df_schemas['schema_name'].tolist()

# Function to get tables for a selected schema
@st.cache_data(ttl="10m")
def get_tables(schema_name):
    # Query information_schema.tables to get tables in a specific schema
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = :schema_name;"
    df_tables = conn.query(query, params={"schema_name": schema_name})
    return df_tables['table_name'].tolist()

# Function to get columns (schema) for a selected table
@st.cache_data(ttl="10m")
def get_table_schema(schema_name, table_name):
    # Query information_schema.columns to get column details (name and data type)
    query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = :schema_name AND table_name = :table_name;"
    df_columns = conn.query(query, params={"schema_name": schema_name, "table_name": table_name})
    return df_columns

# Function to get row count for a selected table
@st.cache_data(ttl="10m")
def get_row_count(schema_name, table_name):
    query = f'SELECT count(*) FROM "{schema_name}"."{table_name}";'
    df_count = conn.query(query)
    return df_count.iloc[0, 0]

# --- Streamlit UI ---

schemas = get_schemas()
if schemas:
    selected_schema = st.sidebar.selectbox("Select a schema", schemas)
   
    tables = get_tables(selected_schema)
    if tables:
        selected_table = st.sidebar.selectbox("Select a table", tables)
       
        st.subheader(f"Schema for Table: {selected_schema}.{selected_table}")
        
        # Display Row Count
        row_count = get_row_count(selected_schema, selected_table)
        st.metric("Total Rows", row_count)

        table_schema_df = get_table_schema(selected_schema, selected_table)
        st.dataframe(table_schema_df, use_container_width=True)
       
        # Optional: Display some data from the table
        st.subheader(f"Sample Data from: {selected_table}")
        try:
            sample_data_query = f'SELECT * FROM "{selected_schema}"."{selected_table}" LIMIT 10;'
            sample_data_df = conn.query(sample_data_query, ttl="10m")
            st.dataframe(sample_data_df, use_container_width=True)
        except Exception as e:
            st.warning(
                f"Could not fetch sample data. This may be due to permission issues, "
                f"table access restrictions, or connectivity problems. Error details: {e}"
            )
           
    else:
        st.info("No tables found in this schema.")
else:
    st.warning("No available schemas found (excluding system schemas).")