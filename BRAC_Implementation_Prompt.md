# BRAC Framework — Full Implementation Prompt for Claude Code

## PROJECT IDENTITY

**Paper**: "A Byzantine-Resilient Agentic Framework for Robust Multimodal Subtyping of Non-Hodgkin Lymphoma"
**Framework**: BRAC — Byzantine-Resilient Agentic Consensus
**Target venue**: HAIS 2026 (Springer LNCS) → Neurocomputing journal extension
**Language**: Python 3.10+
**Deep learning**: PyTorch 2.x
**Hardware**: GPU-aware (CUDA), must also run on CPU for unit tests

---

## HIGH-LEVEL OBJECTIVE

Implement the complete BRAC framework for B-cell Non-Hodgkin Lymphoma (NHL) subtyping from a multimodal AI system with 4 specialized agents. The system must be:
1. **Byzantine-resilient** — tolerates 1 out of 4 agents being adversarial/faulty
2. **Uncertainty-aware** — provides distribution-free coverage guarantees via conformal prediction
3. **Explainable** — produces axiomatic Shapley value attributions with inter-agent synergy analysis
4. **Reproducible** — deterministic seeds, full logging, experiment configs via YAML

---

## PROJECT STRUCTURE

```
brac/
├── README.md
├── pyproject.toml                  # Dependencies: torch, numpy, scipy, scikit-learn, matplotlib, seaborn, pyyaml, rich, wandb (optional)
├── configs/
│   ├── default.yaml                # Default hyperparameters
│   ├── experiment_byzantine.yaml   # Byzantine attack experiments
│   ├── experiment_conformal.yaml   # Conformal coverage experiments
│   └── experiment_shapley.yaml     # Shapley attribution experiments
├── brac/
│   ├── __init__.py
│   ├── types.py                    # Type definitions, dataclasses, enums
│   ├── hypothesis.py               # NHL hypothesis space (WHO classification)
│   │
│   ├── agents/                     # Layer 3: Agent Inference
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base agent
│   │   ├── pathology_agent.py      # ViT-based histopathology + IHC
│   │   ├── radiology_agent.py      # 3D CNN for PET/CT
│   │   ├── laboratory_agent.py     # Transformer for flow cytometry + labs
│   │   ├── clinical_agent.py       # Transformer for structured clinical data
│   │   └── mock_agent.py           # Deterministic mock agent for testing
│   │
│   ├── evidence/                   # Layers 1-2: Evidence Ingestion + Semantic Normalization
│   │   ├── __init__.py
│   │   ├── ingestion.py            # Raw data loading per modality
│   │   ├── semantic.py             # Semantic evidence tuples (SNOMED-CT, ICD-O-3)
│   │   └── quality.py              # Quality scoring (Q_i, C_i, S_i)
│   │
│   ├── consensus/                  # Layer 4: Byzantine Consensus
│   │   ├── __init__.py
│   │   ├── trust.py                # Innovation 1: Trust-bootstrapped reliability
│   │   ├── fisher_rao.py           # Fisher-Rao distance on probability simplex
│   │   ├── geometric_median.py     # Innovation 2: Riemannian Weiszfeld algorithm
│   │   ├── robustness.py           # Innovation 3: Empirical theorem verification
│   │   ├── conformal.py            # Innovation 4: Conformal prediction sets (APS)
│   │   ├── shapley.py              # Innovation 5: Exact Shapley values + interactions
│   │   └── aggregators.py          # Baselines: weighted avg, coordinate median, Krum, trimmed mean
│   │
│   ├── orchestrator.py             # Main BRAC orchestrator (Algorithm 1)
│   ├── attacks.py                  # Byzantine attack models (Type I, II, strategic)
│   └── visualization.py            # Simplex plots, Shapley bar charts, convergence curves
│
├── experiments/
│   ├── run_all.py                  # Master experiment runner
│   ├── exp1_byzantine_resilience.py
│   ├── exp2_conformal_coverage.py
│   ├── exp3_shapley_attribution.py
│   ├── exp4_convergence.py
│   ├── exp5_ablation.py
│   └── exp6_case_studies.py
│
├── tests/
│   ├── test_fisher_rao.py          # Metric axioms: symmetry, triangle inequality, d(p,p)=0
│   ├── test_geometric_median.py    # Convergence, breakdown point verification
│   ├── test_conformal.py           # Coverage guarantee holds empirically
│   ├── test_shapley.py             # Efficiency axiom: sum(phi_i) = v(A) - 1/K
│   ├── test_trust.py               # ReLU clipping, entropy distinction
│   └── test_orchestrator.py        # End-to-end BRAC pipeline
│
└── notebooks/
    └── demo.ipynb                  # Interactive demo with visualizations
```

---

## DETAILED SPECIFICATIONS PER MODULE

### 1. `brac/types.py` — Core Type Definitions

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch
import numpy as np

class NHLSubtype(Enum):
    """WHO-classified B-cell NHL subtypes. K=9 classes."""
    FL = 0          # Follicular Lymphoma
    MCL = 1         # Mantle Cell Lymphoma
    CLL_SLL = 2     # Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma
    DLBCL_GCB = 3   # Diffuse Large B-Cell Lymphoma, Germinal Center B-cell
    DLBCL_ABC = 4   # Diffuse Large B-Cell Lymphoma, Activated B-cell
    MZL = 5         # Marginal Zone Lymphoma
    BL = 6          # Burkitt Lymphoma
    LPL = 7         # Lymphoplasmacytic Lymphoma
    HCL = 8         # Hairy Cell Leukemia
    # OTHER = 9     # Optional catch-all

class Modality(Enum):
    PATHOLOGY = "pathology"     # Histopathology, IHC, molecular markers
    RADIOLOGY = "radiology"     # PET/CT imaging
    LABORATORY = "laboratory"   # Flow cytometry, biochemistry
    CLINICAL = "clinical"       # Demographics, staging, symptoms

class ByzantineType(Enum):
    HONEST = "honest"
    TYPE_I = "type_i"     # Data fault: uncertain, low quality evidence
    TYPE_II = "type_ii"   # Model fault: confident but wrong beliefs
    STRATEGIC = "strategic"  # Adversarial: optimally misleading

@dataclass
class SemanticEvidence:
    """Ontology-grounded evidence unit."""
    finding_type: str       # e.g., "morphology", "immunophenotype", "molecular"
    code: str               # SNOMED-CT or ICD-O-3 code
    value: str              # e.g., "CD20+", "Ki-67 > 90%"
    confidence: float       # [0, 1]
    provenance: str         # Source modality
    quality_score: float    # [0, 1]

@dataclass
class EvidenceQuality:
    """Per-agent evidence quality factors."""
    Q: float    # Data quality [0, 1]
    C: float    # Coverage [0, 1] — fraction of expected evidence present
    S: float    # Consistency [0, 1] — internal coherence

@dataclass 
class AgentOutput:
    """Output from a single diagnostic agent."""
    belief: torch.Tensor            # Shape: (K,) on probability simplex
    quality: EvidenceQuality
    evidence: list[SemanticEvidence]
    modality: Modality

@dataclass
class BRACResult:
    """Complete output of the BRAC framework."""
    diagnosis: NHLSubtype
    consensus_belief: torch.Tensor          # b* on simplex
    prediction_set: list[NHLSubtype]        # C_alpha
    prediction_set_size: int
    shapley_values: dict[Modality, float]   # phi_i per agent
    interaction_indices: dict[tuple[Modality, Modality], float]  # I_ij
    agent_reliabilities: dict[Modality, float]   # r_i
    agent_trusts: dict[Modality, float]          # tau_i
    convergence_rounds: int
    accepted: bool                          # True if |C_alpha| <= 2
    confidence: float                       # max(b*)
```

### 2. `brac/consensus/fisher_rao.py` — Fisher-Rao Geometry

This is the mathematical foundation. Must be numerically stable.

```
EQUATIONS TO IMPLEMENT:

1. Fisher-Rao distance (geodesic on probability simplex):
   d_FR(p, q) = 2 * arccos(sum_k sqrt(p_k * q_k))
   
   NUMERICAL STABILITY: Clamp the argument to arccos in [-1, 1].
   Add epsilon=1e-10 to avoid sqrt(0). Normalize inputs to sum to 1.

2. Square-root embedding:
   psi(p) = (sqrt(p_1), ..., sqrt(p_K))
   Maps Delta^{K-1} isometrically to the positive orthant of S^{K-1}
   
3. Inverse embedding:
   psi_inv(z) = z^2 / ||z||^2  (element-wise square, then normalize)

4. Exponential map on S^{K-1}_+ at point z in direction v:
   Exp_z(v) = cos(||v||) * z + sin(||v||) * v / ||v||
   
5. Logarithmic map on S^{K-1}_+ from z to w:
   Log_z(w) = (theta / sin(theta)) * (w - cos(theta) * z)
   where theta = arccos(clamp(<z, w>, -1, 1))
   
   EDGE CASE: If theta ≈ 0, return w - z (first-order approximation)
```

**Required functions:**
- `fisher_rao_distance(p, q)` → float. Batched version: `(B, K), (B, K) → (B,)`
- `sqrt_embedding(p)` → z on sphere
- `sqrt_embedding_inv(z)` → p on simplex
- `exp_map(z, v)` → point on sphere
- `log_map(z, w)` → tangent vector at z
- `frechet_mean_sphere(points, weights, max_iter=100, tol=1e-8)` → weighted Fréchet mean

**Unit tests** (`tests/test_fisher_rao.py`):
- `d_FR(p, p) == 0` for random p
- `d_FR(p, q) == d_FR(q, p)` (symmetry)
- `d_FR(p, r) <= d_FR(p, q) + d_FR(q, r)` (triangle inequality)
- `psi_inv(psi(p)) == p` (round-trip)
- `Exp_z(Log_z(w)) == w` (round-trip on sphere)
- Distances match known values: d_FR(uniform, one_hot) = 2*arccos(1/sqrt(K))

### 3. `brac/consensus/geometric_median.py` — Riemannian Weiszfeld Algorithm

```
ALGORITHM: Riemannian Weiszfeld iteration (Eq. 3 from paper)

Input: beliefs {b_i}, reliabilities {r_i}, max_iter L, tolerance eps
Output: consensus b* on simplex

1. Initialize: b^(0) = weighted Fréchet mean of {b_i} with weights {r_i}
              z^(0) = psi(b^(0))

2. For ell = 0, 1, ..., L-1:
   a. For each agent i:
      - d_i = d_FR(b^(ell), b_i)
      - If d_i < eps_numerical: skip (avoid division by zero)
      - w_i = r_i / d_i
   
   b. Compute weighted tangent mean:
      v = sum_i [ w_i * Log_{z^(ell)}(psi(b_i)) ] / sum_i [ w_i ]
   
   c. Step on sphere:
      z^(ell+1) = Exp_{z^(ell)}(v)
   
   d. Project back to simplex:
      b^(ell+1) = psi_inv(z^(ell+1))
   
   e. Convergence check:
      If d_FR(b^(ell+1), b^(ell)) < tolerance: break

3. Return b* = b^(ell+1)

IMPORTANT: Track convergence history {d_FR(b^(ell), b^(ell-1))} for plotting.
Return both the consensus and the convergence trace.
```

**Also implement the belief update rule for iterative consensus:**
```
For each agent i, after computing b*:
  lambda_i = lambda_0 * (1 - r_i)    # receptivity: unreliable agents defer more
  b_i^(t+1) = (1 - lambda_i) * b_i^(t) + lambda_i * b*^(t)
  b_i^(t+1) = b_i^(t+1) / sum(b_i^(t+1))   # re-normalize to simplex
```

### 4. `brac/consensus/trust.py` — Trust-Bootstrapped Reliability

```
EQUATION (Eq. 1 from paper):

r_i = tau_i * sigma(MLP_theta(Q_i, C_i, S_i, H(b_i)))

where:
  tau_i = ReLU(cosine_similarity(b_i, b_path))    # behavioral trust
  H(b_i) = -sum_k b_ik * log(b_ik + eps)          # belief entropy
  sigma = sigmoid
  MLP_theta: R^4 → R^1 (2 hidden layers, 32 units each, ReLU activation)

Special case: For pathology agent itself, tau_path = 1.0

The MLP is a LEARNABLE component trained during calibration.
For experiments without training data, provide a DEFAULT mode:
  r_i = tau_i * sigmoid(0.4*Q_i + 0.3*C_i + 0.2*S_i - 0.5*H(b_i))
```

**Class structure:**
```python
class TrustEstimator:
    def __init__(self, root_of_trust: Modality = Modality.PATHOLOGY, learnable: bool = False):
        ...
    
    def compute_behavioral_trust(self, beliefs: dict[Modality, Tensor]) -> dict[Modality, float]:
        """Cosine similarity of each agent's belief with root-of-trust belief, ReLU-clipped."""
        ...
    
    def compute_reliability(self, beliefs: dict[Modality, Tensor], 
                           qualities: dict[Modality, EvidenceQuality]) -> dict[Modality, float]:
        """Full reliability = tau_i * sigma(MLP(Q, C, S, H))"""
        ...
```

### 5. `brac/consensus/conformal.py` — Conformal Prediction Sets

```
EQUATIONS (Eq. 5-6 from paper):

CALIBRATION PHASE (offline, on held-out labeled data):
1. For each calibration case j = 1..N:
   s_j = 1 - p*(h_j^true)     # non-conformity score
2. q_hat = Quantile({s_j}, ceil((N+1)(1-alpha)) / N)

PREDICTION PHASE (online, for new case):
1. Sort hypotheses by decreasing consensus probability: pi(1), pi(2), ..., pi(K)
2. Find smallest k such that: sum_{i=1}^{k} p*(h_{pi(i)}) >= 1 - q_hat
3. Return C_alpha = {h_{pi(1)}, ..., h_{pi(k)}}

DECISION RULE:
- |C_alpha| = 1  → Accept diagnosis (confident, guaranteed coverage)
- |C_alpha| = 2-3 → Accept with differential note (genuine ambiguity)
- |C_alpha| > 3  → Escalate to human expert review
```

**Class structure:**
```python
class ConformalPredictor:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None
    
    def calibrate(self, consensus_beliefs: Tensor, true_labels: Tensor) -> float:
        """Compute q_hat from calibration data. Returns q_hat."""
        ...
    
    def predict_set(self, consensus_belief: Tensor) -> list[NHLSubtype]:
        """Return the adaptive prediction set C_alpha."""
        ...
    
    def marginal_coverage(self, beliefs: Tensor, labels: Tensor) -> float:
        """Empirical coverage on test set. Should be >= 1-alpha."""
        ...
```

### 6. `brac/consensus/shapley.py` — Exact Shapley Values

```
EQUATION (Eq. 7 from paper):

phi_i = sum_{S ⊆ A \ {a_i}} [ |S|! * (n - |S| - 1)! / n! ] * [v(S ∪ {a_i}) - v(S)]

where:
  A = {path, rad, lab, clin}  (n=4 agents)
  v(S) = p*_S(h_true)  = consensus confidence using only agents in S
  v(∅) = 1/K            = uniform prior (no information)

With n=4, enumerate ALL 2^4 = 16 coalitions exactly:
  ∅, {path}, {rad}, {lab}, {clin}, {path,rad}, {path,lab}, ..., {path,rad,lab,clin}

For each coalition S:
  1. Run geometric median consensus using only agents in S
  2. Record v(S) = max_h p*_S(h) or p*_S(h_true) if true label available

INTERACTION INDICES between agents i and j:
  I_ij = sum_{S ⊆ A \ {i,j}} [ |S|! * (n - |S| - 2)! / (n-1)! ] 
         * [v(S ∪ {i,j}) - v(S ∪ {i}) - v(S ∪ {j}) + v(S)]

  I_ij > 0 → Synergy (e.g., pathology + lab for DLBCL cell-of-origin)
  I_ij < 0 → Redundancy

SUBTYPE-SPECIFIC Shapley values:
  phi_i^(k) = Shapley value when the value function is v(S) = p*_S(h_k)
  This reveals which agents are most important for diagnosing each subtype.
```

**Class structure:**
```python
class ShapleyAttributor:
    def __init__(self, agents: list[Modality], geometric_median_fn: callable):
        self.agents = agents
        self.n = len(agents)
        self.gm_fn = geometric_median_fn
    
    def compute_coalition_values(self, beliefs: dict, reliabilities: dict, 
                                  true_label: Optional[int] = None) -> dict[frozenset, float]:
        """Compute v(S) for all 2^n coalitions."""
        ...
    
    def shapley_values(self, coalition_values: dict) -> dict[Modality, float]:
        """Exact Shapley values from coalition values."""
        ...
    
    def interaction_indices(self, coalition_values: dict) -> dict[tuple, float]:
        """Pairwise interaction indices I_ij."""
        ...
    
    def subtype_shapley(self, beliefs: dict, reliabilities: dict) -> dict[NHLSubtype, dict[Modality, float]]:
        """Shapley values per subtype phi_i^(k)."""
        ...

VERIFICATION: assert abs(sum(phi_i for phi_i in shapley_values.values()) - (v_grand - 1/K)) < 1e-6
```

### 7. `brac/consensus/aggregators.py` — Baselines for Comparison

Implement these aggregation baselines for Table 1 in the paper:

```python
def weighted_average(beliefs: list[Tensor], weights: list[float]) -> Tensor:
    """Standard weighted average. Breakdown point = 1/n."""

def coordinate_median(beliefs: list[Tensor], weights: list[float]) -> Tensor:
    """Coordinate-wise weighted median. Breakdown point = 50% but per-dimension only."""

def krum(beliefs: list[Tensor], f: int = 1) -> Tensor:
    """Krum selection (Blanchard et al. 2017). Selects the belief closest to its neighbors."""

def multi_krum(beliefs: list[Tensor], f: int = 1, m: int = 3) -> Tensor:
    """Multi-Krum: average of m Krum-selected beliefs."""

def trimmed_mean(beliefs: list[Tensor], trim_fraction: float = 0.25) -> Tensor:
    """Coordinate-wise trimmed mean."""
```

### 8. `brac/attacks.py` — Byzantine Attack Models

```python
class ByzantineAttack:
    """Models different Byzantine failure modes for experimental evaluation."""
    
    @staticmethod
    def type_i_data_fault(honest_belief: Tensor, noise_scale: float = 0.3) -> Tensor:
        """Type I: Uncertain agent. Add Dirichlet noise to honest belief."""
        noisy = honest_belief + torch.distributions.Dirichlet(
            torch.ones_like(honest_belief) * (1.0 / noise_scale)
        ).sample()
        return noisy / noisy.sum()
    
    @staticmethod
    def type_ii_model_fault(honest_belief: Tensor, K: int) -> Tensor:
        """Type II: Confident but wrong. Return peaked belief on wrong class."""
        true_class = honest_belief.argmax()
        wrong_class = (true_class + torch.randint(1, K, (1,)).item()) % K
        belief = torch.full((K,), 0.01)
        belief[wrong_class] = 0.91  # Very confident, totally wrong
        return belief / belief.sum()
    
    @staticmethod
    def strategic_attack(honest_beliefs: list[Tensor], target_class: int, K: int) -> Tensor:
        """Strategic: Knows other agents' beliefs, tries to maximally displace consensus."""
        # Compute current honest centroid, then push as far as possible toward target
        ...
    
    @staticmethod
    def label_flip(honest_belief: Tensor) -> Tensor:
        """Reverse the belief vector (highest becomes lowest)."""
        return honest_belief.flip(0)
        # Then renormalize
```

### 9. `brac/orchestrator.py` — Main BRAC Algorithm (Algorithm 1)

```
ALGORITHM 1: Byzantine-Resilient Agentic Consensus (BRAC)

Input: Evidence {E_i} for i=1..n, max_rounds T, convergence_threshold epsilon,
       calibration_scores {s_j}, confidence_level alpha, lambda_0

Output: BRACResult (diagnosis h*, prediction set C_alpha, Shapley values {phi_i})

STEPS:
1. AGENT INFERENCE: b_i^(0) = f_i(E_i) for each agent i

2. TRUST ESTIMATION (Innovation 1):
   tau_i = ReLU(cos(b_i^(0), b_path^(0)))   for i ≠ path
   tau_path = 1.0
   r_i = tau_i * sigmoid(MLP(Q_i, C_i, S_i, H(b_i)))

3. ITERATIVE CONSENSUS (Innovation 2):
   For t = 0, 1, ..., T-1:
     a. b*^(t) = RiemannianWeiszfeld({b_i^(t)}, {r_i})
     b. If max_i d_FR(b_i^(t), b*^(t)) < epsilon: break
     c. For each agent i:
        lambda_i = lambda_0 * (1 - r_i)
        b_i^(t+1) = (1 - lambda_i) * b_i^(t) + lambda_i * b*^(t)
        Normalize b_i^(t+1) to simplex

4. CONFORMAL PREDICTION (Innovation 4):
   C_alpha = APS(b*, {s_j}, alpha)

5. SHAPLEY ATTRIBUTION (Innovation 5):
   {phi_i} = ExactShapley(A, v)  over all 16 coalitions

6. DECISION:
   h* = argmax_h p*(h)
   If |C_alpha| <= 2: ACCEPT
   Else: ESCALATE to human review

Return BRACResult(...)
```

**Class:**
```python
class BRACOrchestrator:
    def __init__(self, config: dict):
        self.trust_estimator = TrustEstimator(...)
        self.conformal = ConformalPredictor(alpha=config['alpha'])
        self.shapley = ShapleyAttributor(...)
        self.max_rounds = config.get('max_rounds', 10)
        self.epsilon = config.get('epsilon', 1e-4)
        self.lambda_0 = config.get('lambda_0', 0.3)
    
    def run(self, agent_outputs: dict[Modality, AgentOutput]) -> BRACResult:
        """Execute full BRAC pipeline."""
        ...
    
    def calibrate(self, calibration_data: list[tuple[dict, NHLSubtype]]):
        """Calibrate conformal predictor on labeled data."""
        ...
```

### 10. `brac/agents/` — Agent Architectures

**For HAIS paper experiments, implement TWO modes:**

**Mode A: Mock agents (for consensus-layer experiments)**
Agents produce synthetic beliefs from known ground truth + controlled noise.
This is what we need for the paper experiments.

```python
class MockAgent:
    def __init__(self, modality: Modality, accuracy: float = 0.85, 
                 noise_type: str = 'dirichlet', concentration: float = 10.0):
        ...
    
    def generate_belief(self, true_label: int, K: int) -> AgentOutput:
        """Generate a belief vector centered on true_label with controlled noise."""
        # Dirichlet with high concentration on true class
        alpha = torch.ones(K) * 0.1
        alpha[true_label] = concentration
        belief = torch.distributions.Dirichlet(alpha).sample()
        ...
```

**Mode B: Real neural agents (for Neurocomputing extension)**
Stubs with proper interfaces, to be trained on real data later.

```python
class PathologyAgent(BaseAgent):
    """ViT-based agent for whole-slide histopathology images."""
    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, wsi_patches: Tensor) -> AgentOutput:
        # Multiple Instance Learning over patches
        ...

class RadiologyAgent(BaseAgent):
    """3D CNN for PET/CT volumes."""
    def __init__(self, num_classes: int = 9):
        self.encoder = nn.Sequential(...)  # 3D ResNet or similar
    ...

class LaboratoryAgent(BaseAgent):
    """Transformer for flow cytometry + biochemistry panels."""
    ...

class ClinicalAgent(BaseAgent):
    """Transformer for structured clinical data (demographics, staging, symptoms)."""
    ...
```

---

## EXPERIMENTS TO IMPLEMENT

### Experiment 1: Byzantine Resilience (`exp1_byzantine_resilience.py`)

**Goal**: Demonstrate that geometric median resists Byzantine attacks while weighted average fails.

**Setup**:
- K=9 subtypes, n=4 agents
- Generate 1000 synthetic cases
- For each case: 3 honest agents (Dirichlet noise around true label), 1 Byzantine
- Vary Byzantine attack: Type I, Type II, Strategic, Label-flip
- Compare aggregators: Weighted Average, Coordinate Median, Krum, Multi-Krum, **Ours (Geometric Median)**

**Metrics**:
- Accuracy (top-1 and top-3)
- Consensus displacement: d_FR(b*, b_oracle)
- Breakdown verification: increase f from 0 to 3, measure accuracy collapse

**Output**: Table + line plot showing accuracy vs. number of Byzantine agents.

### Experiment 2: Conformal Coverage (`exp2_conformal_coverage.py`)

**Goal**: Verify distribution-free coverage guarantee Pr(h_true ∈ C_alpha) ≥ 1-alpha.

**Setup**:
- Split synthetic data: 500 calibration + 500 test
- alpha ∈ {0.01, 0.05, 0.10, 0.20}
- Run with and without Byzantine agents

**Metrics**:
- Empirical coverage (should be ≥ 1-alpha)
- Average prediction set size |C_alpha|
- Set size distribution histogram
- Coverage conditional on subtype

**Output**: Table: alpha | target coverage | empirical coverage | avg set size

### Experiment 3: Shapley Attribution (`exp3_shapley_attribution.py`)

**Goal**: Show that Shapley values reveal clinically meaningful patterns.

**Setup**:
- For each of the 9 subtypes, generate 100 cases
- Compute Shapley values and interaction indices per subtype

**Expected patterns** (validate these):
- FL: pathology dominates (phi_path ≈ 0.45) — follicular pattern is morphologically distinctive
- BL: radiology high (phi_rad ≈ 0.30) — PET avidity is characteristic
- DLBCL: positive I_{path,lab} — IHC + flow cytometry synergy for GCB vs ABC
- CLL/SLL: laboratory dominates — flow cytometry is diagnostic
- MCL: path + lab synergy — cyclin D1 + morphology

**Output**: 
- Heatmap: subtypes × agents showing phi_i^(k)
- Bar chart of interaction indices per subtype
- Efficiency axiom verification: |sum(phi) - (v(A) - 1/K)| for each case

### Experiment 4: Convergence Analysis (`exp4_convergence.py`)

**Goal**: Verify Theorem 3 (convergence rate).

**Setup**:
- Run Weiszfeld algorithm on 1000 cases, log d_FR(b^(ell), b^(ell-1)) at each iteration
- Vary: number of agents, belief spread, Byzantine fraction

**Output**: 
- Convergence curve: log(d_FR) vs. iteration ell (should show linear convergence)
- Histogram of iterations to convergence
- Empirical contraction rate rho vs. theoretical bound

### Experiment 5: Ablation Study (`exp5_ablation.py`)

**Goal**: Show that each innovation contributes to overall performance.

**Ablation configurations**:
1. Full BRAC (all 5 innovations)
2. BRAC − trust (equal reliabilities r_i = 1)
3. BRAC − geometric median (use weighted average instead)
4. BRAC − conformal (use entropy threshold instead)
5. BRAC − Shapley (no attribution, just diagnosis)
6. BRAC − trust − geometric median (naive ensemble)

**Metrics**: Accuracy, coverage, set size, Shapley efficiency, computation time

**Output**: Ablation table (rows = configs, columns = metrics)

### Experiment 6: Case Studies (`exp6_case_studies.py`)

**Goal**: Generate the diagnostic report card shown in the paper.

For 3 representative cases (one easy, one ambiguous, one with Byzantine):
- Print full BRAC output: h*, C_alpha, {phi_i}, {tau_i}, {r_i}
- Show agent beliefs before and after consensus
- Visualize on simplex (3-class projection)
- Show Shapley bar chart

---

## VISUALIZATION REQUIREMENTS (`brac/visualization.py`)

1. **Simplex plot** (Fig. 2 in paper): 
   - Equilateral triangle with 3 subtypes at vertices
   - Plot agent beliefs as colored dots (red=path, blue=rad, green=lab, yellow=clin)
   - Plot geometric median (teal) and weighted average (red outline)
   - Dashed lines showing displacement

2. **Convergence curve**: 
   - x-axis: iteration, y-axis: log d_FR (log scale)
   - One line per experiment, shaded confidence interval

3. **Shapley heatmap**: 
   - Rows: subtypes, Columns: agents
   - Color: phi_i^(k) value
   - Annotated with numbers

4. **Shapley bar chart** (per case): 
   - Horizontal bars, one per agent, colored by modality
   - Show trust tau_i and reliability r_i alongside

5. **Conformal coverage plot**:
   - x-axis: alpha, y-axis: empirical coverage
   - Diagonal reference line (target = 1-alpha)
   - Error bars from multiple runs

6. **Byzantine resilience plot**:
   - x-axis: number of Byzantine agents (0-3), y-axis: accuracy
   - One line per aggregation method
   - Our method should maintain accuracy longer

**Style**: Use matplotlib with seaborn styling. Color palette:
- Pathology: #DC5050 (red)
- Radiology: #508CDC (blue) 
- Laboratory: #50B450 (green)
- Clinical: #DCB43C (yellow)
- Geometric Median: #3CB4B4 (teal)
- Weighted Average: #C85050 (dark red)
- Orchestrator: #A050C8 (purple)

All plots: 300 DPI, PDF + PNG output, white background, font size 12.

---

## CONFIGURATION (`configs/default.yaml`)

```yaml
# BRAC Framework Configuration
seed: 42

# Hypothesis space
num_subtypes: 9   # K

# Agents
num_agents: 4     # n
agent_modalities: [pathology, radiology, laboratory, clinical]
root_of_trust: pathology

# Trust estimation
trust:
  learnable: false   # Use default linear combination for now
  weights: [0.4, 0.3, 0.2, -0.5]  # Q, C, S, -H(b)

# Geometric median consensus
consensus:
  max_outer_rounds: 10     # T (outer belief refinement rounds)
  max_weiszfeld_iters: 50  # L (inner Weiszfeld iterations per round)
  convergence_threshold: 1e-4   # epsilon
  lambda_0: 0.3            # Base receptivity for belief update
  weiszfeld_tol: 1e-8      # Weiszfeld convergence tolerance
  numerical_eps: 1e-10     # Epsilon for numerical stability

# Conformal prediction
conformal:
  alpha: 0.05              # 95% coverage target
  calibration_size: 500    # N

# Shapley attribution
shapley:
  compute_interactions: true
  compute_subtype_specific: true

# Decision
decision:
  max_set_size_accept: 2   # Accept if |C_alpha| <= this value
  min_confidence: 0.0      # Optional minimum confidence threshold

# Experiments
experiments:
  num_synthetic_cases: 1000
  byzantine_fractions: [0.0, 0.25, 0.50, 0.75]  # 0, 1, 2, 3 out of 4
  attack_types: [type_i, type_ii, strategic, label_flip]
  aggregators: [weighted_average, coordinate_median, krum, multi_krum, trimmed_mean, geometric_median]
  num_runs: 10             # For confidence intervals

# Agent simulation (mock mode)
mock_agents:
  pathology:
    accuracy: 0.90
    concentration: 15.0    # Dirichlet concentration (higher = more confident)
  radiology:
    accuracy: 0.80
    concentration: 8.0
  laboratory:
    accuracy: 0.85
    concentration: 10.0
  clinical:
    accuracy: 0.70
    concentration: 5.0
```

---

## IMPLEMENTATION PRIORITIES

**Phase 1 — Core consensus (do this first)**:
1. `types.py` — All dataclasses and enums
2. `fisher_rao.py` — Distance, embeddings, Exp/Log maps
3. `geometric_median.py` — Riemannian Weiszfeld
4. `trust.py` — Reliability estimation
5. `aggregators.py` — Baselines
6. Unit tests for Phase 1

**Phase 2 — Uncertainty and explainability**:
7. `conformal.py` — APS calibration and prediction
8. `shapley.py` — Exact Shapley values and interactions
9. Unit tests for Phase 2

**Phase 3 — Integration**:
10. `orchestrator.py` — Full BRAC Algorithm 1
11. `mock_agent.py` — Synthetic belief generation
12. `attacks.py` — Byzantine attack models
13. End-to-end test

**Phase 4 — Experiments and visualization**:
14. `visualization.py` — All plot functions
15. Experiments 1-6
16. `run_all.py` — Master script producing all tables and figures

---

## QUALITY REQUIREMENTS

1. **Type hints everywhere** — use Python type hints on all function signatures
2. **Docstrings** — Google-style docstrings on every public function and class
3. **Logging** — use Python `logging` module, log all key intermediate values
4. **Determinism** — seed everything: `torch.manual_seed()`, `np.random.seed()`, `random.seed()`
5. **Assertions** — assert simplex constraints (sum=1, all>=0) at key checkpoints
6. **Numerical stability** — clamp arccos inputs, add epsilon to sqrt, check for NaN
7. **Tests** — every module must have corresponding tests; run with `pytest`
8. **No hardcoded paths** — use configs and relative paths only

---

## EXPECTED EXPERIMENTAL RESULTS (for validation)

When your code runs the experiments, verify these patterns:

| Metric | Expected |
|--------|----------|
| Geometric median accuracy (0 Byzantine) | ≈ 90-95% |
| Geometric median accuracy (1 Byzantine) | ≈ 85-92% (graceful degradation) |
| Weighted average accuracy (1 Byzantine) | ≈ 40-60% (collapse) |
| Conformal coverage (alpha=0.05) | ≥ 95.0% (by theory) |
| Average |C_alpha| (alpha=0.05) | ≈ 1.2-1.8 |
| Shapley efficiency error | < 1e-6 |
| Weiszfeld convergence | ≤ 10 iterations typically |
| Pathology Shapley (FL cases) | ≈ 0.40-0.50 (dominant) |
| Lab Shapley (CLL cases) | ≈ 0.35-0.45 (dominant) |
| I_{path,lab} for DLBCL | > 0 (positive synergy) |

---

## CRITICAL IMPLEMENTATION NOTES

1. **The probability simplex is NOT Euclidean.** Never average beliefs by arithmetic mean as the primary method. Always use Fisher-Rao geometry for the geometric median. Arithmetic averaging is only used as a BASELINE for comparison.

2. **Weiszfeld can fail if the median coincides with a data point.** Add a small perturbation or use the smoothed Weiszfeld variant (add epsilon to d_i in the denominator).

3. **Conformal prediction requires exchangeability.** The calibration and test data must be i.i.d. from the same distribution. In synthetic experiments this is trivially satisfied.

4. **Shapley computation is exact with n=4.** Do NOT use Monte Carlo sampling. Enumerate all 16 coalitions explicitly. The computational cost is negligible.

5. **Belief vectors must ALWAYS be on the simplex.** After any operation (averaging, updating, etc.), re-normalize: `b = b / b.sum()` and clamp to `[eps, 1]`.

6. **Fisher-Rao distance is bounded.** For K-simplex: d_FR ∈ [0, π]. Maximum distance is between opposite vertices (one-hot vectors of different classes).
