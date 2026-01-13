#!/bin/bash
cd /Users/nizakatali/Emmyzcode
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[client]
showErrorDetails = true
[browser]
gatherUsageStats = false
EOF

/Users/nizakatali/Emmyzcode/.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
