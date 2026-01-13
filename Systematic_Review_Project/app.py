import streamlit as st
import pandas as pd
from config import Config
from state_manager import SessionState
from ui_components import UIComponents
from utils import AIService
from data_services import DataAggregator


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=Config.APP_TITLE,
        layout="wide",
        page_icon=Config.PAGE_ICON
    )
    
    SessionState.initialize()
    
    # Render sidebar and get settings
    model_name, active_sources, uploaded_files, num_per_source = (
        UIComponents.render_sidebar()
    )
    
    # Main content
    st.title("🔬 Global Epidemiology Evidence Agent")
    
    # Step 1: Research Goal
    st.header("Step 1: Research Goal")
    goal = st.text_input(
        "Research Question:",
        value=st.session_state.goal
    )
    
    if st.button("🪄 Auto-Infer PICO & Query"):
        if not goal:
            st.warning("Please enter a research question first.")
        else:
            with st.status("Generating PICO and Query...") as status:
                analysis = AIService.infer_pico_and_query(goal, model_name)
                
                if analysis:
                    pico = st.session_state.pico
                    pico.population = analysis.get('p', '')
                    pico.intervention = analysis.get('i', '')
                    pico.comparator = analysis.get('c', '')
                    pico.outcome = analysis.get('o', '')
                    st.session_state.query = analysis.get('query', '')
                    st.session_state.goal = goal
                    
                    status.update(
                        label="PICO and query generated!",
                        state="complete"
                    )
                    st.rerun()
                else:
                    status.update(
                        label="Failed to generate PICO",
                        state="error"
                    )
    
    st.session_state.query = st.text_area(
        "Search Query:",
        value=st.session_state.query,
        height=100
    )
    
    # Step 2: Fetch Evidence
    st.header("Step 2: Fetch Evidence")
    
    if st.button("🚀 Fetch Data from All Sources"):
        if not st.session_state.query:
            st.warning("Please enter a search query first.")
        else:
            with st.status("Aggregating Evidence...") as status:
                papers = DataAggregator.fetch_all(
                    st.session_state.query,
                    active_sources,
                    num_per_source,
                    uploaded_files
                )
                
                st.session_state.papers = papers
                status.update(
                    label=f"Total papers collected: {len(papers)}",
                    state="complete"
                )
    
    # Display paper preview
    if st.session_state.papers:
        st.subheader("📚 Evidence Preview")
        papers_df = pd.DataFrame([p.to_dict() for p in st.session_state.papers])
        st.dataframe(
            papers_df[['Source', 'ID', 'Title']],
            use_container_width=True
        )
    
    # Step 3: AI Screening
    st.header("Step 3: AI-Powered Screening")
    
    if st.button("🔍 Run Full AI Screening", type="primary"):
        if not st.session_state.papers:
            st.error("No papers to screen. Please fetch data first.")
        else:
            screened_results = []
            progress_bar = st.progress(0)
            
            with st.status("Analyzing Evidence...") as status:
                total = len(st.session_state.papers)
                
                for i, paper in enumerate(st.session_state.papers):
                    screening = AIService.screen_paper(
                        paper,
                        st.session_state.pico,
                        model_name
                    )
                    
                    result = {**paper.to_dict(), **screening.to_dict()}
                    screened_results.append(result)
                    
                    progress_bar.progress((i + 1) / total)
                
                st.session_state.results = pd.DataFrame(screened_results)
                status.update(
                    label="Screening complete!",
                    state="complete"
                )
    
    # Display results
    UIComponents.render_results(st.session_state.results)


if __name__ == "__main__":
    main()
