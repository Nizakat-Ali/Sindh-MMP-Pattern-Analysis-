# Sindh MMP Analysis Dashboard

An interactive web dashboard for analyzing migration patterns and mobility profiles (MMP) in Sindh province, Pakistan. Built with Streamlit, this tool visualizes migration flows between districts using multiple visualization modes including Sankey diagrams, network graphs, heatmaps, and interactive maps.

## Features

- **🔍 Multi-Filter Interface**: Filter by destination district, origin district, MMP subtype, and weight type (children/families)
- **📊 Sankey Diagrams**: Bidirectional flow visualization with aggregation by MMP type
- **🕸️ Network Graphs**: Directed network visualization with node size/edge width scaling
- **🔥 Heatmap**: Origin-Destination heatmap with zero-value masking
- **📋 Data Tables**: Export filtered data and summary statistics as CSV
- **📈 Network Metrics**: Calculate in-strength, out-strength, and betweenness centrality
- **🗺️ Interactive Maps**: Visualize migration flows on interactive Folium maps with 6 different tile styles

## Project Structure

```
├── app.py                      # Main Streamlit dashboard
├── HelloWorld.py               # Static analysis script
├── coffee_machine.py           # Coffee machine CLI simulator
├── requirements.txt            # Python dependencies
├── Sindh MMP Analysis.xlsx     # Input data file
└── outputs/                    # Generated visualizations
```

## Installation

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd sindh-mmp-analysis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

## Dependencies

- pandas (2.3.3)
- numpy
- networkx (3.2.1)
- matplotlib (3.9.4)
- plotly (6.5.1)
- streamlit (1.50.0)
- folium (0.20.0)
- streamlit-folium (0.25.3)
- openpyxl (3.1.5)

## Usage

### Dashboard Filters
- **Weight Type**: Children or Families
- **Destination District**: Multi-select Sindh districts
- **Origin District**: Multi-select origin districts
- **MMP Subtype**: Filter by migration type
- **Top-N Corridors**: Display top 5-100 migration corridors
- **Map Style**: Choose from 6 different map tile styles
- **View Mode**: Sankey, Network, Heatmap, Tables, Metrics, or Map

### View Modes

- **Sankey**: Migration flows with bidirectional toggle
- **Network**: Directed graph visualization
- **Heatmap**: Origin-Destination matrix
- **Tables**: Data export with CSV download
- **Metrics**: Network centrality measures
- **Map**: Interactive geographic visualization with Pakistan-Afghanistan focus

## Data Format

Input Excel file requires:
- ORIGIN DISTRICT
- DISTRICT NAME
- # OF CHILDREN
- # OF FAMILIES
- ORIGIN COUNTRY
- MMP SUBTYPE

## Features Highlights

✅ 470+ lines of production code  
✅ 6 distinct visualization modes  
✅ Interactive map with geographic bounds  
✅ CSV export functionality  
✅ Network metrics calculation (in-strength, out-strength, betweenness centrality)  
✅ Performance optimized with caching  
✅ 150+ district coordinates (Pakistan & Afghanistan)  

## License

MIT

## Author

Created for analyzing migration patterns in Sindh, Pakistan
