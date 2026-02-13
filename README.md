# BRAC: Byzantine-Resilient Agentic Consensus

A multimodal AI framework for B-cell Non-Hodgkin Lymphoma (NHL) subtyping with Byzantine fault tolerance, uncertainty quantification, and explainability.

## Overview

BRAC implements a Byzantine-resilient consensus mechanism for combining predictions from multiple diagnostic agents (pathology, radiology, laboratory, clinical) to achieve robust NHL subtype classification.

### Key Innovations

1. **Trust-Bootstrapped Reliability** - Anchors trust estimation to pathology as root-of-trust
2. **Riemannian Geometric Median** - Fisher-Rao geometry on probability simplex
3. **Byzantine Resilience** - Tolerates up to f < n/3 adversarial agents
4. **Conformal Prediction** - Distribution-free coverage guarantees
5. **Shapley Attribution** - Axiomatic explainability with inter-agent synergy analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/brac.git
cd brac

# Install in development mode
pip install -e ".[dev]"

# Or with all optional dependencies
pip install -e ".[dev,wandb,full]"
```

## Project Structure

```
brac/
├── brac/
│   ├── types.py              # Core type definitions
│   ├── hypothesis.py         # NHL hypothesis space
│   ├── agents/               # Diagnostic agents
│   ├── evidence/             # Evidence processing
│   ├── consensus/            # Byzantine consensus algorithms
│   ├── orchestrator.py       # Main BRAC algorithm
│   ├── attacks.py            # Byzantine attack models
│   └── visualization.py      # Plotting utilities
├── configs/                  # YAML configuration files
├── experiments/              # Experiment scripts
├── tests/                    # Unit tests
└── notebooks/                # Demo notebooks
```

## Quick Start

```python
from brac.orchestrator import BRACOrchestrator
from brac.agents.mock_agent import MockAgent
from brac.types import Modality, NHLSubtype

# Create orchestrator with default config
orchestrator = BRACOrchestrator.from_yaml('configs/default.yaml')

# Generate mock agent outputs
agents = {
    Modality.PATHOLOGY: MockAgent(Modality.PATHOLOGY),
    Modality.RADIOLOGY: MockAgent(Modality.RADIOLOGY),
    Modality.LABORATORY: MockAgent(Modality.LABORATORY),
    Modality.CLINICAL: MockAgent(Modality.CLINICAL),
}

# Run BRAC consensus
result = orchestrator.run(agent_outputs)

print(f"Diagnosis: {result.diagnosis}")
print(f"Prediction Set: {result.prediction_set}")
print(f"Shapley Values: {result.shapley_values}")
```

## Running Experiments

```bash
# Run all experiments
python -m experiments.run_all

# Run specific experiment
python -m experiments.exp1_byzantine_resilience
```

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{brac2026,
  title={A Byzantine-Resilient Agentic Framework for Robust Multimodal Subtyping of Non-Hodgkin Lymphoma},
  author={...},
  booktitle={Hybrid Artificial Intelligence Systems (HAIS)},
  year={2026},
  publisher={Springer}
}
```

## License

MIT License
