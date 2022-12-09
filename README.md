# BraessTraficSim
A trafic simulator with focus on Braess's paradox

# Results
The simulation results gives that the paradox has a prelevance of 39.2% (392 of a thousand runs gives a paradox)

# Usage
To install and use run

    git clone https://github.com/Paulsson99/BraessTraficSim.git
    cd BraessTrafficSim
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    # Example command
    run-large-sim -N 1 2 1 -n 1 2 -e 1 -s