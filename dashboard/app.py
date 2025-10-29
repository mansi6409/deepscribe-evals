#!/usr/bin/env python3
"""
Streamlit Dashboard for DeepScribe Evals Suite

Usage:
    streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime


st.set_page_config(
    page_title="DeepScribe Evals Dashboard",
    page_icon="",
    layout="wide"
)


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_results(results_file):
    """Load evaluation results from JSON file.
    
    Args:
        results_file (str): Path to the JSON results file
        
    Returns:
        dict: Parsed JSON data containing metadata, aggregate metrics, and individual cases
        
    Note:
        Results are cached for 60 seconds to improve dashboard performance.
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def clear_cache():
    """Clear Streamlit's data cache to force reload of results files.
    
    This function is called when the user clicks the refresh button,
    ensuring that any new or updated result files are immediately reflected.
    """
    st.cache_data.clear()


def main():
    """Main dashboard application function.
    
    Renders the complete Streamlit dashboard with:
    - File selection sidebar
    - Overall performance metrics
    - Score distributions and visualizations
    - Common issues analysis
    - Individual case explorer
    - Performance metrics tracking
    
    The dashboard provides interactive exploration of evaluation results,
    allowing users to drill down from aggregate statistics to individual
    case findings and detailed error analysis.
    """
    st.title(" DeepScribe SOAP Note Evaluation Dashboard")
    st.markdown("---")
    
    # Sidebar - File selection
    st.sidebar.header("Select Results File")
    
    # Add refresh button
    if st.sidebar.button(" Refresh Data", help="Reload data from disk"):
        clear_cache()
        st.rerun()
    
    results_dir = Path("results")
    if not results_dir.exists():
        st.error("No results directory found. Please run evaluation first.")
        return
    
    result_files = sorted(results_dir.glob("*.json"), reverse=True)
    
    if not result_files:
        st.warning("No result files found. Run `python run_eval.py` first.")
        return
    
    selected_file = st.sidebar.selectbox(
        "Results file:",
        result_files,
        format_func=lambda x: x.name,
        key="file_selector"  # Unique key to track changes
    )
    
    # Load data (cache is invalidated by file path changing)
    data = load_results(str(selected_file))
    
    # Extract metadata
    metadata = data['metadata']
    aggregate = data['aggregate']
    cases = data['cases']
    
    # Display metadata
    st.sidebar.markdown("### Run Info")
    st.sidebar.write(f"**Mode:** {metadata['mode']}")
    st.sidebar.write(f"**Cases:** {metadata['num_cases']}")
    st.sidebar.write(f"**Dataset:** {metadata.get('dataset', 'N/A')}")
    st.sidebar.write(f"**Time:** {metadata['timestamp']}")
    
    # Show last loaded time
    current_time = datetime.now().strftime("%H:%M:%S")
    st.sidebar.markdown("---")
    st.sidebar.caption(f" Data loaded at: {current_time}")
    st.sidebar.caption(" Select a different file or click Refresh to update")
    
    # Overview metrics
    st.header(" Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Composite Score",
            f"{aggregate['mean_composite']:.3f}",
            help="Overall quality score (higher is better)"
        )
    
    with col2:
        st.metric(
            "Missing Rate",
            f"{aggregate['mean_missing_rate']:.3f}",
            delta=None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Hallucination Rate",
            f"{aggregate['mean_hallucination_rate']:.3f}",
            delta=None,
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Runtime",
            f"{aggregate['total_runtime_seconds']:.1f}s",
            help=f"Total evaluation time"
        )
    
    st.markdown("---")
    
    # Score distribution
    st.header(" Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Composite score histogram
        scores = [case['metrics']['composite'] for case in cases]
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Composite Score Distribution",
            labels={'x': 'Composite Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=aggregate['mean_composite'], line_dash="dash", 
                     annotation_text="Mean", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot of different metrics
        metrics_data = {
            'Missing': [case['metrics']['missing_rate_critical'] for case in cases],
            'Hallucination': [case['metrics']['hallucination_rate_critical'] for case in cases],
            'Contradiction': [case['metrics']['contradicted_rate'] for case in cases],
            'Unsupported': [case['metrics']['unsupported_rate'] for case in cases]
        }
        
        fig = go.Figure()
        for metric_name, values in metrics_data.items():
            fig.add_trace(go.Box(y=values, name=metric_name))
        
        fig.update_layout(
            title="Error Rate Distributions",
            yaxis_title="Rate",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Common issues
    st.header(" Common Issues")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Common Missing Entities")
        if aggregate['most_common_missing']:
            for i, entity in enumerate(aggregate['most_common_missing'][:10], 1):
                st.write(f"{i}. {entity}")
        else:
            st.write("None found")
    
    with col2:
        st.subheader("Most Common Hallucinated Entities")
        if aggregate['most_common_hallucinated']:
            for i, entity in enumerate(aggregate['most_common_hallucinated'][:10], 1):
                st.write(f"{i}. {entity}")
        else:
            st.write("None found")
    
    st.markdown("---")
    
    # Individual case explorer
    st.header(" Case Explorer")
    
    # Create case selector
    case_ids = [case['case_id'] for case in cases]
    selected_case_id = st.selectbox("Select a case to examine:", case_ids)
    
    # Get selected case
    selected_case = next(case for case in cases if case['case_id'] == selected_case_id)
    
    # Display case details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Composite Score", f"{selected_case['metrics']['composite']:.3f}")
    with col2:
        st.metric("Missing", len(selected_case['missing_critical']))
    with col3:
        st.metric("Hallucinated", len(selected_case['hallucinated']))
    with col4:
        st.metric("Contradicted", len(selected_case['contradicted']))
    
    # Findings tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Critical", 
        "Hallucinated", 
        "Contradicted",
        "Unsupported"
    ])
    
    with tab1:
        if selected_case['missing_critical']:
            df = pd.DataFrame(selected_case['missing_critical'])
            st.dataframe(df[['claim_or_entity', 'severity', 'confidence', 'section']], 
                        use_container_width=True)
        else:
            st.info("No missing critical findings")
    
    with tab2:
        if selected_case['hallucinated']:
            df = pd.DataFrame(selected_case['hallucinated'])
            st.dataframe(df[['claim_or_entity', 'severity', 'confidence', 'section']], 
                        use_container_width=True)
        else:
            st.info("No hallucinated findings")
    
    with tab3:
        if selected_case['contradicted']:
            df = pd.DataFrame(selected_case['contradicted'])
            st.dataframe(df[['claim_or_entity', 'evidence_span', 'severity', 'confidence']], 
                        use_container_width=True)
        else:
            st.info("No contradicted findings")
    
    with tab4:
        if selected_case['unsupported']:
            df = pd.DataFrame(selected_case['unsupported'])
            st.dataframe(df[['claim_or_entity', 'confidence']].head(10), 
                        use_container_width=True)
        else:
            st.info("No unsupported findings")
    
    # Section coverage
    st.subheader(" SOAP Section Coverage")
    coverage = selected_case['section_coverage']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**Subjective:**", "" if coverage['subjective'] else "")
    with col2:
        st.write("**Objective:**", "" if coverage['objective'] else "")
    with col3:
        st.write("**Assessment:**", "" if coverage['assessment'] else "")
    with col4:
        st.write("**Plan:**", "" if coverage['plan'] else "")
    
    st.markdown("---")
    
    # Performance metrics
    st.header(" Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evaluation time per case
        runtimes = [case['metrics']['runtime_seconds'] for case in cases]
        fig = px.scatter(
            x=range(len(runtimes)),
            y=runtimes,
            title="Evaluation Time per Case",
            labels={'x': 'Case Index', 'y': 'Runtime (seconds)'}
        )
        fig.add_hline(y=sum(runtimes)/len(runtimes), line_dash="dash",
                     annotation_text="Mean", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tiers executed distribution
        tiers_counts = {}
        for case in cases:
            tiers = tuple(case['meta']['tiers_executed'])
            tiers_counts[str(tiers)] = tiers_counts.get(str(tiers), 0) + 1
        
        fig = px.bar(
            x=list(tiers_counts.keys()),
            y=list(tiers_counts.values()),
            title="Tiers Executed per Case",
            labels={'x': 'Tiers', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
