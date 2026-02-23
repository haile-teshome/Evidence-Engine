# Evidence Engine

Evidence Engine is a Streamlit-based research automation tool designed to streamline systematic reviews and literature evidence synthesis. It leverages AI (via LangChain and Ollama) to assist researchers in defining PICO criteria, screening abstracts, extracting data from full-text papers, and generating PRISMA flow diagrams.

## Key Features

- **PICO Analysis**: Automatically infer Population, Intervention, Comparator, and Outcome criteria from a research goal.
- **Multi-Source Search**: Aggregates literature from PubMed, arXiv, bioRxiv, Semantic Scholar, and CORE.
- **AI-Powered Screening**: Screens abstracts against specific inclusion/exclusion criteria using LLMs like GPT, Claude, or local Llama3 via Ollama.
- [cite_start]**Table Extraction**: Automatically detects and extracts data tables from HTML, PDF, and text-based research papers[cite: 1].
- **PRISMA Flow Visualization**: Generates dynamic PRISMA flow diagrams using Graphviz to track study selection progress.
- [cite_start]**Data Export**: Export extracted evidence and screening results to CSV or JSON formats[cite: 1].

## Tech Stack

- **Frontend**: Streamlit
- **AI Orchestration**: LangChain (OpenAI, Anthropic, Ollama)
- **Data Processing**: Pandas, Biopython (Entrez), PyPDF
- **Diagrams**: Graphviz

## Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) (for local AI features)
- Graphviz (for PRISMA diagrams)

### Quick Start
1. Clone the repository.
2. Run the automated setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh