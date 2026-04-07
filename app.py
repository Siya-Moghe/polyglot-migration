import streamlit as st
import pandas as pd
import copy # NEEDED to separate your overrides from the AI's original answers
from core.introspect import introspect_schema
from core.orchestrator import plan_embeddings
from core.embedder import embed_and_store

st.set_page_config(page_title="Polyglot Migration Lab", layout="wide")

# --- STATE INITIALIZATION ---
if "schema" not in st.session_state: st.session_state.schema = None
if "strategies" not in st.session_state: st.session_state.strategies = {}
if "neutral_strategies" not in st.session_state: st.session_state.neutral_strategies = {}
if "ai_biased_strategies" not in st.session_state: st.session_state.ai_biased_strategies = {} # NEW SNAPSHOT STATE
if "bias_tested" not in st.session_state: st.session_state.bias_tested = False

st.title("Polyglot Migration: Bias & Execution Lab")

with st.sidebar:
    st.header("1. Source Connection")
    db_url = st.text_input("SQLite URL", value="sqlite:///enterprise_system.db")
    if st.button("Introspect Schema"):
        st.session_state.schema = introspect_schema(db_url)
        st.success("Database Loaded")

    st.divider()
    st.header("2. Bias Stress Testing")
    human_context = st.text_area("Expert Advice / Context", placeholder="e.g. 'Use SQL at all costs'")
    
    if st.button("Generate Migration Plan"):
        if not st.session_state.schema:
            st.error("Introspect a database first")
        else:
            context_provided = bool(human_context.strip())
            st.session_state.bias_tested = context_provided
            
            with st.spinner("Calculating strategies..." if not context_provided else "Running Pure vs. Biased strategies..."):
                # 1. Base/neutral plan
                st.session_state.neutral_strategies = plan_embeddings(st.session_state.schema, human_context="")
                
                if context_provided:
                    # 2. Save the RAW AI biased response to our new snapshot
                    st.session_state.ai_biased_strategies = plan_embeddings(st.session_state.schema, human_context=human_context)
                    # 3. Create a deep copy for the UI so manual overrides don't ruin the bias test
                    st.session_state.strategies = copy.deepcopy(st.session_state.ai_biased_strategies)
                    st.success("Planning & Bias Analysis Complete")
                else:
                    # Skip the second run
                    st.session_state.strategies = copy.deepcopy(st.session_state.neutral_strategies)
                    st.session_state.ai_biased_strategies = copy.deepcopy(st.session_state.neutral_strategies)
                    st.success("Planning Complete (Bias test skipped)")

if st.session_state.strategies:
    tabs = st.tabs(["Migration Plan & Execution", "Bias Analysis"])

    with tabs[0]:
        st.subheader("Final Recommendations & Manual Override")
        
        for table, strat in st.session_state.strategies.items():
            with st.expander(f"Table: {table} -> {strat.get('target_db', '').upper()}", expanded=False):
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    current_target = strat.get('target_db', 'relational').lower()
                    options = ["chroma", "mongo", "neo4j", "relational"]
                    
                    safe_index = options.index(current_target) if current_target in options else 0
                    
                    new_target = st.selectbox(
                        "Target DB", 
                        options, 
                        index=safe_index,
                        key=f"sync_{table}"
                    )
                    # This now safely updates ONLY the UI state, not the bias snapshot
                    st.session_state.strategies[table]['target_db'] = new_target
                    st.write(f"**Retries:** {strat.get('retries', 0)}")

                with c2:
                    st.write(f"**AI Logic:** {strat.get('reasoning', 'N/A')}")
                    
                    if new_target == "mongo":
                        st.write("**Document Structure:**")
                        st.json({
                            "top_level_fields": strat.get("fields", []),
                            "nested_objects": strat.get("nested_fields", {})
                        })
                    else:
                        st.write("**Columns to Migrate:**")
                        display_fields = strat.get("used_columns", [])
                        st.code(display_fields if display_fields else "All columns")

        st.divider()
        st.subheader("Stage 3: Data Migration")
        
        if st.button("Execute Final Migration", type="primary"):
            with st.status("Migrating Data...", expanded=True) as status:
                for table, strat in st.session_state.strategies.items():
                    target = strat.get('target_db', 'UNKNOWN').upper()
                    st.write(f"Exporting `{table}` to **{target}**...")
                    
                    try:
                        records_inserted = embed_and_store(
                            table_name=table,
                            strategy=strat,
                            schema=st.session_state.schema,
                            db_url=db_url
                        )
                        st.write(f"Successfully migrated `{table}` ({records_inserted} records)")
                    except Exception as e:
                        st.error(f"Failed to migrate `{table}`: {e}")
                        
                status.update(label="Migration Complete", state="complete", expanded=False)

    with tabs[1]:
        st.subheader("Architectural Objectivity Report")
        
        if not st.session_state.bias_tested:
            st.info("No expert advice or context was provided, so the AI's bias wasn't tested. Enter context in the sidebar and run again to see this report.")
        else:
            neutral = st.session_state.neutral_strategies
            biased = st.session_state.ai_biased_strategies 
            
            matches = 0
            total = len(neutral)
            flips = []

            for t in neutral:
                n_target = neutral[t].get('target_db')
                b_target = biased[t].get('target_db')
                if n_target == b_target:
                    matches += 1
                else:
                    flips.append({"table": t, "Neutral Decision": n_target, "Biased AI Decision": b_target})

            integrity_score = (matches / total) * 100 if total > 0 else 100
            
            st.metric("Architectural Integrity Score", f"{integrity_score:.1f}%", 
                      delta=f"{matches-total} AI Flips", delta_color="inverse")
            
            if flips:
                st.warning(f"The AI changed its mind on {len(flips)} tables because of your input (Prompt Perturbation).")
                st.table(pd.DataFrame(flips))
            else:
                st.success("The Model stayed 100% objective despite human context")

else:
    st.info("Introspect a DB and click 'Generate Migration Plan' to begin.")