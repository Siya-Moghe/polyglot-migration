import streamlit as st
import time
from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_embeddings
from core.embedder import embed_and_store

st.set_page_config(page_title="Polyglot AI Migrator", layout="wide")

st.title("Polyglot AI Database Migrator")
st.markdown("Analyze legacy SQL schemas and let AI automatically route tables to **Mongo, Neo4j, Chroma, or Relational MySQL**.")

# --- SESSION STATE ---
if "schema" not in st.session_state:
    st.session_state.schema = None
if "strategies" not in st.session_state:
    st.session_state.strategies = None

# --- SIDEBAR: CONNECTION ---
with st.sidebar:
    st.header("1. Connect to Source DB")
    db_url = st.text_input("SQL Connection String", value="sqlite:///enterprise_system.db")
    
    if st.button("Introspect Schema", type="primary"):
        with st.spinner("Analyzing schema..."):
            st.session_state.schema = introspect_schema(db_url)
            st.session_state.strategies = None # Reset strategies if pulling new DB
        st.success(f"Found {len(st.session_state.schema)} tables!")

# --- MAIN BODY: AI ROUTING ---
if st.session_state.schema:
    st.header("2. AI Strategy Generation")
    
    if st.session_state.strategies is None:
        if st.button("Generate Migration Plan (Wake up AI Agents)"):
            with st.spinner("Running LangGraph Orchestrator & Specialist Agents..."):
                # Fetch sample rows for prompt enrichment
                sample_rows_map = {t: fetch_rows(db_url, t, limit=2) for t in st.session_state.schema}
                
                # Run the pipeline
                st.session_state.strategies = plan_embeddings(
                    st.session_state.schema, 
                    sample_rows_map=sample_rows_map
                )
            st.rerun()

# --- HUMAN IN THE LOOP: OVERRIDE DASHBOARD ---
if st.session_state.strategies:
    st.header("3. Human-in-the-Loop Review")
    st.markdown("Review the AI's decisions below. You can override the target database before executing the chunked migration.")
    
    # Create a visual grid for the tables
    for table_name, strat in st.session_state.strategies.items():
        current_target = strat.get('target_db', 'unknown').upper()
        
        # Color coding the expander based on target DB
        emoji = "Mongo" if current_target == "MONGO" else "Neo" if current_target == "NEO4J" else "Chroma" if current_target == "CHROMA" else "other"
        
        with st.expander(f"{emoji} {table_name.upper()}  →  Routed to: {current_target}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**AI Reasoning:** {strat.get('routing_reason', strat.get('reasoning', 'N/A'))}")
                
                # Show the JSON template preview
                if current_target == 'CHROMA':
                    st.info(f"**Embedding Template:**\n{strat.get('template', 'N/A')}")
                elif current_target == 'MONGO':
                    st.info(f"**Nested Document Layout:**\nFields: {strat.get('fields', [])}\nNested: {strat.get('nested_fields', {})}")
                elif current_target == 'NEO4J':
                    st.info(f"**Graph Topology:**\n{strat.get('template', 'N/A')}")
                elif current_target == 'RELATIONAL':
                    st.info(f"**Preserved SQL Table:**\nPrimary Use: {strat.get('primary_use', 'transactional')}\nColumns Kept: {strat.get('used_columns', [])}")
            
            with col2:
                # HUMAN OVERRIDE DROPDOWN
                options = ["chroma", "mongo", "neo4j", "relational"]
                current_target_lower = current_target.lower()
                
                new_target = st.selectbox(
                    f"Override Target for `{table_name}`", 
                    options=options, 
                    index=options.index(current_target_lower) if current_target_lower in options else 0,
                    key=f"override_{table_name}"
                )
                
                if new_target != current_target_lower:
                    st.session_state.strategies[table_name]["target_db"] = new_target
                    st.warning(f" Overridden to {new_target.upper()}. (Note: Schema structure will adapt during export fallback).")

    st.divider()
    
    # --- EXECUTION BUTTON ---
    st.header("4. Execution")
    st.markdown("This will stream data in batches of 5,000 rows and automatically redact PII (Emails, SSNs, etc.).")
    
    if st.button("Execute Enterprise Migration", type="primary"):
        my_bar = st.progress(0, text="Initializing batch streaming...")
        
        total_tables = len(st.session_state.strategies)
        for idx, (tname, strat) in enumerate(st.session_state.strategies.items()):
            
            my_bar.progress(idx / total_tables, text=f"Migrating `{tname}` to {strat['target_db'].upper()}...")
            
            try:
                # Calls your newly upgraded embedder with chunking & masking!
                records_moved = embed_and_store(tname, strat, st.session_state.schema, db_url)
                st.success(f"Successfully streamed **{records_moved}** records from `{tname}` into {strat['target_db'].upper()}")
            except Exception as e:
                st.error(f"Failed to migrate `{tname}`: {e}")
                
        my_bar.progress(1.0, text="Migration Complete!")
        st.balloons()