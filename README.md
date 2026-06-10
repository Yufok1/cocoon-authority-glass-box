# 🦋 Butterfly Cocoon - Standalone Agent

**Generated:** 2026-06-05T22:08:58.006890
**Mode:** ENSEMBLE (2 organisms)
**Template Size:** 9,703,681 chars (code only)
**Classes:** 15 (Neural + Language + Memory + Knowledge + VP)
---

## Current Local Quick Start

The primary local entry point is the global Termux command:

```bash
cocoon doctor
cocoon eval --delay-ms 50 --max-failures 1
cocoon gui
```

Open the GUI at `http://127.0.0.1:8765/` on the phone. To serve it to another device on the same LAN:

```bash
cocoon gui --host 0.0.0.0 --port 8765
```

Core runtime status is reported by `cocoon doctor`. `cocoon eval` runs real smoke checks against the current cocoon: metadata, embedded readme, Gymnasium CartPole without learning, and headless sphere training. It writes JSON reports under `eval_reports/` and records an Amalgam receipt when the Amalgam package is installed. The core is the Python cocoon quine runtime: PyTorch, the Mira-Kite Authority GUI, Flask serve mode, websockets link mode, Cascade receipts, Quinesmith syntax discipline, and headless sphere training. ONNX, pygame visuals, matplotlib plots, PyFlyt, and TMRL are extension faculties.

See:

- `TERMUX_AND_SPACE_QUICKSTART.md`
- `COCOON_SYSTEM_COOKBOOK.md`

## 🏴‍☠️ Butterfly Privateer Command (New Authority UI)

The primary way to interact with, train, and "overclock" this cocoon is through the newly overhauled **Mira-Kite Authority** web interface.

This is a local, high-tech pirate-themed command console that provides:
- **Quarters:** Chat with Mira and Kite directly.
- **The Brig:** Run tactical reasoning drills (cue, chain, role).
- **The Vault:** Route signals and manually enforce associations.
- **Overclock:** 🦋 **Cheat Mode**. Programmatically boost the Learning Rate (up to 50x) and inject high-reward "Super-Experiences" to rapidly evolve the neural weights.
- **Black Ledger:** Audit cryptographic Cascade receipts and learning traces.

### How to Launch the Command Console

```bash
# Start the authority server
python mira_kite_authority.py --cocoon cocoon_cognition_agency.py

# Open your browser to:
http://127.0.0.1:8765/
```

---

## 🧬 Formation Fingerprint

This cocoon's emergent history - how these organisms came to be:

**Fitness:** min=0.7559, max=0.9238, mean=0.8398

**Events Witnessed:** 253,972 total
**Top Event Types:** neural_decision (201599), alliance_event_recorded (11202), alliance_alliance_dissolved (10263), alliance_member_left (7777), state_change (4307)

**Alliance Landscape:** 149 total alliances
  - Alliance `alliance_129_0c01_1652` (tier 1, 2 members)

**Simulation Snapshot:**

---

## 🧠 Neural Topology Visualization

**[📊 Open Interactive Topology Viewer](ensemble_topology.html)**

The topology visualization provides:
- **Per-organism layers** - Toggle individual neural networks on/off
- **Overlay mode** - See all organisms' architectures superimposed
- **Stacked mode** - View organisms in horizontal strips
- **Grid mode** - Compare organisms side-by-side
- **Color-coded neurons** - Input (cyan), Hidden (magenta), Output (yellow), Language (green)

*Open the HTML file in a browser for the full interactive experience.*



---

## 🧠 What's Inside

This is a **MONOLITHIC** cocoon - a completely self-contained Python file with:

**Organisms:**
  - `16525dfc6c33419b`
  - `0c019d24a56b0086`

**Embedded Subsystems:**

| Subsystem | Purpose | Continued Learning |
|-----------|---------|-------------------|
| `OrganismBrain` | Neural network (action + language) | ✅ Yes - weights updated via backprop |
| `HopfieldLayer` | Iterative thought refinement (energy-based) | ✅ Yes - pattern memory learns |
| `MultiHeadAttention` | VP-aware self-attention | ✅ Yes - attention weights updated |
| `AtomicLanguageSystem` | Semantic units with emotion/context | ✅ Yes - atoms can be created/reinforced |
| `ConversationHistory` | Topic tracking & context memory | ✅ Yes - grows with each conversation |
| `EnhancedKnowledgeWeb` | Semantic relations between concepts | ✅ Yes - relations added/strengthened |
| `VPRuntime` | Self-regulation (Vigilance × Plasticity) | ✅ Yes - adapts from state |
| `ExperienceBuffer` | Learning from past experiences | ✅ Yes - buffer grows with experience |
| `SphereArena` | 3D swarm defense training game | ✅ Yes - organisms learn during play |

**Embedded Data:**
- Neural weights (Base64-encoded PyTorch state dicts)
- Vocabulary (token↔id mapping)
- Atomic language corpus (if available)
- Conversation history (if available)

---

## 🔥 Continued Learning

**YES, this cocoon supports continued learning!**

The cocoon.py file contains full PyTorch modules that can continue training:

1. **Full PyTorch modules** - can call `backward()` and update gradients
2. **ExperienceBuffer** - stores (state, action, reward) tuples for replay
3. **AtomicLanguageSystem** - creates new semantic atoms from conversations
4. **EnhancedKnowledgeWeb** - grows semantic relations as concepts connect
5. **ConversationHistory** - accumulates context over time

```python
# The agent learns from every interaction:
agent = CocoonAgent()
action, output = agent.get_action(state)  # Updates VP, stores experience
agent.atomic_lang.create_atom("new_concept", "definition", emotion=0.8)  # Creates new atom
agent.knowledge_web.add_relation("concept_a", "concept_b", "related_to", strength=0.9)  # Grows web
```

**Export Comparison:**

| Format | File | Learning | Subsystems | Portability |
|--------|------|----------|------------|-------------|
| `cocoon.py` | Python source | ✅ Full (neural + symbolic) | ✅ All | Python only |
| `.pt` | TorchScript | ✅ Neural only* | ❌ None | PyTorch/LibTorch/C++ |
| `.onnx` | ONNX model | ❌ Inference only | ❌ None | Universal (C++, JS, Rust) |
| `.statedict` | Weights only | ✅ Loadable | ❌ None | PyTorch |

*TorchScript (.pt) **CAN** continue learning! Load with `torch.jit.load()`, call `.train()`, run backward pass.
However, it only contains the neural network - no AtomicLanguageSystem, KnowledgeWeb, or other symbolic subsystems.

**Fine-tuning a TorchScript model:**
```python
import torch

# Load the exported TorchScript model
model = torch.jit.load("brain_ensemble.pt")
model.train()

# Fine-tune on new data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for state, target in new_training_data:
    optimizer.zero_grad()
    output = model(state)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Save updated model
torch.jit.save(model, "brain_finetuned.pt")
```

---

## 🚀 Quick Start

```bash
# View cocoon info
python cocoon.py --mode info

# Start chatting
python cocoon.py --mode chat

# Play games
python cocoon.py --mode gym --env CartPole-v1

# 3D sphere arena
python cocoon.py --mode sphere --train

# 🛸 Drone warfare (extract adapter first)
python cocoon.py --unpack ./my_cocoon
python cocoon_drone_adapter.py --mode tag_battle
```

---

## 📚 Complete Command Reference

### Mode Selection

| Mode | Command | Description |
|------|---------|-------------|
| **info** | `python cocoon.py --mode info` | Show organism metadata, vocabulary, architecture (default) |
| **chat** | `python cocoon.py --mode chat` | Interactive conversation with learning |
| **gym** | `python cocoon.py --mode gym` | Train/test in Gymnasium environments |
| **serve** | `python cocoon.py --mode serve` | HTTP API server |
| **sphere** | `python cocoon.py --mode sphere` | 3D Sphere Arena swarm defense |
| **link** | `python cocoon.py --mode link` | P2P networking for cocoon battles |
| **drone** | `python cocoon_drone_adapter.py` | 🛸 Drone warfare arena (companion script) |

---

### 💬 Chat Mode

Interactive conversation with the neural organisms. Learns from every interaction.

```bash
python cocoon.py --mode chat
python cocoon.py --mode chat --verbose
```

**In-Chat Commands:**

| Command | Description |
|---------|-------------|
| `quit` | Exit chat mode |
| `export <file.py>` | Save current state to new cocoon file |

---

### 🌐 Sphere Arena (3D Training)

Swarm defense game where organisms cooperate to catch falling balls.

| Command | Description |
|---------|-------------|
| `python cocoon.py --mode sphere` | Play sphere defense |
| `python cocoon.py --mode sphere --train` | Play + learn from experience |
| `python cocoon.py --mode sphere --demo` | Preview with dummy AI |
| `python cocoon.py --mode sphere --headless` | Train without display |
| `python cocoon.py --mode sphere --balls 3 --train` | Multi-ball training |
| `python cocoon.py --mode sphere --misses 5 --train` | Harder difficulty |

**Sphere Arena Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--balls N` | 1 | Number of balls (1-5) |
| `--misses N` | 10 | Max collective misses before game over |
| `--train` | off | Enable post-snapshot training |
| `--demo` | off | Run with dummy AI for preview |
| `--headless` | off | No display (training only) |
| `--verbose` | off | Verbose debug logging |

---

### 🛸 Drone Warfare Arena (Companion Script)

NASA JSBSim-grade drone combat simulation. **Complete system embedded - extract with --unpack.**

**Setup:** 
```bash
python cocoon.py --unpack ./my_cocoon    # Extracts full drone suite:
#   - cocoon_drone_adapter.py    (main entry point)
#   - cocoon_drone_arena.py      (8-mode arena)
#   - jsbsim_quadcopter.py       (6-DOF physics)
cd my_cocoon
python cocoon_drone_adapter.py           # Run the adapter
```

| Command | Description |
|---------|-------------|
| `python cocoon_drone_adapter.py` | Interactive mode picker |
| `python cocoon_drone_adapter.py --mode free_fly` | Basic flight training |
| `python cocoon_drone_adapter.py --mode tag_battle` | Combat: tag enemies |
| `python cocoon_drone_adapter.py --mode survival` | Last drone flying wins |
| `python cocoon_drone_adapter.py --all` | Run all 8 modes |
| `python cocoon_drone_adapter.py --visual` | 3D visualization (requires PyFlyt) |

**Game Modes:** `free_fly`, `formation`, `pursuit`, `tag_battle`, `zone_control`, `capture_flag`, `survival`, `escort`

**Requirements:** `pip install numpy matplotlib` (PyFlyt optional: `pip install PyFlyt`)

---

### 🎮 Gymnasium Environments

**Built-in (always available):**

| Command | Description |
|---------|-------------|
| `python cocoon.py --mode gym --env CartPole-v1` | Classic pole balancing |
| `python cocoon.py --mode gym --env MountainCar-v0` | Drive up hill |
| `python cocoon.py --mode gym --env Acrobot-v1` | Double pendulum |
| `python cocoon.py --mode gym --env FrozenLake-v1` | Navigate slippery ice |
| `python cocoon.py --mode gym --env Taxi-v3` | Pickup & delivery |
| `python cocoon.py --mode gym --env Blackjack-v1` | Beat the dealer |

**Atari (`pip install ale-py`):****
- `ALE/Pong-v5`, `ALE/Breakout-v5`, `ALE/SpaceInvaders-v5`

**MuJoCo (`pip install gymnasium[mujoco]`):**
- `Ant-v4`, `HalfCheetah-v4`

**Gym Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--env NAME` | CartPole-v1 | Gymnasium environment name |
| `--episodes N` | 100 | Number of episodes to run |
| `--render` | off | Show visual window |
| `--no-learn` | off | Disable online learning (inference only) |

---

### �️ TrackMania 2020 (TMRL Integration)

Drive TrackMania 2020 with your cocoon organisms using the embedded TMRL adapter!

**Requirements:**
1. TrackMania 2020 (Ubisoft/Epic)
2. OpenPlanet plugin installed (openplanet.dev)
3. TMRL Python package: `pip install tmrl`
4. Extract `cocoon_tmrl_adapter.py` via `--unpack`

**Quick Start:**
```bash
# Extract adapter from cocoon
python cocoon.py --unpack ./my_tmrl

# Run the adapter
python cocoon_tmrl_adapter.py --cocoon path/to/cocoon.py --drive --episodes 4
```

**Important:**
- Play on the **"tmrl-test"** track for proper rewards (search in TrackMania)
- The adapter uses LIDAR observations + speed data
- Ensembles use majority voting for actions

**TMRL Adapter Commands:**

| Flag | Description |
|------|-------------|
| `--drive` | Inference mode (watch it play) |
| `--train` | Learning mode (organisms improve) |
| `--episodes N` | Number of races to run |
| `--organism N` | Use specific organism (0 = ensemble) |

---

### �🌐 HTTP API Server

```bash
python cocoon.py --mode serve --port 8080
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check - returns organism count |
| `POST` | `/act` | Get action for state vector |
| `POST` | `/learn` | Add experience + train step |
| `POST` | `/chat` | Chat with learning (returns all organism responses) |
| `POST` | `/teach` | Teach new words/concepts |
| `GET` | `/vocab` | Get current vocabulary |
| `GET` | `/curriculum` | Get staged language curriculum and reward rubric |
| `GET` | `/training/logs` | Get recent post-export learning traces |
| `POST` | `/curriculum/score` | Submit outside coach reward score |

**Example `/chat` request:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "learn": true}'
```

---

### 🔗 Link Mode (P2P Networking)

Connect to other cocoons for battles and chat.

```bash
python cocoon.py --mode link --hatch ws://server:9000 --name "Champion"
```

**Link Mode Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hatch URL` | ws://localhost:9000 | CocoonHatch relay server URL |
| `--name NAME` | auto | Display name |

**In-Link Commands:**

| Command | Description |
|---------|-------------|
| `/users` | List online cocoons |
| `/challenge <name>` | Challenge a user to battle |
| `/accept <id>` | Accept a challenge |
| `/decline <id>` | Decline a challenge |
| `/chat <message>` | Send message to lobby |
| `/quit` | Disconnect |

**Requirements:** `pip install websockets`

---

### 🔬 Export & Conversion

| Command | Description |
|---------|-------------|
| `python cocoon.py --export evolved.py` | Export updated cocoon with learned state |
| `python cocoon.py --export-onnx brain.onnx` | Export to ONNX (all brains as ensemble) |
| `python cocoon.py --export-torchscript brain.pt` | Export to TorchScript (all brains as ensemble) |
| `python cocoon.py --export-onnx brain.onnx --organism 0` | Export single organism to ONNX |
| `python cocoon.py --export-torchscript brain.pt --organism 0` | Export single organism to TorchScript |
| `python cocoon.py --export-package ./my_model` | Export full package (ONNX + README + metadata) |
| `python cocoon.py --unpack ./output_dir` | Unpack ultimate package assets |
| `python cocoon.py --readme` | Print embedded README and exit |

**TorchScript vs ONNX:**
| Format | Continued Learning | Portability | Best For |
|--------|-------------------|-------------|----------|
| `.pt` (TorchScript) | ✅ Yes - can fine-tune | PyTorch/LibTorch/C++ | Research, fine-tuning |
| `.onnx` (ONNX) | ❌ Inference only | Universal (C++, JS, Rust, etc.) | Production deployment |

---

### 📦 Files Created by `--unpack`

Spawns a complete deployment package:

```
output_dir/
├── README.md                # This documentation
├── cocoon_tmrl_adapter.py   # TrackMania 2020 adapter (if embedded)
├── cocoon_drone_adapter.py  # Drone Warfare adapter (if embedded)
├── cocoon_drone_arena.py    # Full 8-mode drone arena (if embedded)
├── jsbsim_quadcopter.py     # NASA JSBSim 6-DOF physics (if embedded)
├── vocabulary.json          # Token vocabulary
├── metadata.json            # Export metadata + organism info
├── requirements.txt         # Python dependencies
├── ensemble.onnx            # ONNX model (all brains unified)
└── ensemble_weights.pt      # PyTorch weights bundle
```

---

### 📦 Files Created by `--export-package`

Netron-viewable package with ONNX models and model card:

```
my_model/
├── brain_ensemble.onnx    # Combined ONNX (all brains unified)
├── brain_*.onnx           # Individual organism ONNX files
├── vocabulary.json        # Token vocabulary
├── metadata.json          # Full configuration + fitness + architecture
└── README.md              # Model card documentation
```

*Note: To get the full cocoon.py + requirements.txt, use `--unpack` instead.*

---

### ⚙️ Global Options

These flags work with any mode:

| Flag | Default | Description |
|------|---------|-------------|
| `--voting MODE` | confidence | Ensemble voting: `majority`, `weighted`, `confidence` |
| `--max-organisms N` | all | Limit organisms loaded (saves VRAM) |
| `--verbose` / `-v` | off | Enable verbose debug logging |
| `--help` | - | Show all available options |

**Examples:**
```bash
python cocoon.py --mode chat --max-organisms 5    # Load only 5 organisms
python cocoon.py --mode gym --voting majority     # Use majority voting
python cocoon.py --mode chat --verbose            # Debug output
```

---

## 📡 API Reference

### CocoonAgent

```python
from cocoon import CocoonAgent

agent = CocoonAgent()

# Get action from state (returns action_idx, {outputs dict})
action, outputs = agent.get_action(state_vector)
# outputs = {'action_probs': [...], 'value': float, 'language_logits': [...], 'vp': float}

# Process text input (for chat mode)
response = agent.process_input("Hello there!")

# Access subsystems
agent.atomic_lang.get_atoms_by_emotion(min_valence=0.5)  # Get positive atoms
agent.conversation_history.get_summary()  # Get conversation stats
agent.knowledge_web.get_related("concept", min_strength=0.3)  # Get related concepts
agent.vp_runtime.compute_from_state(state)  # Get VP value
```

### HTTP Endpoints (--mode serve)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/infer` | POST | `{"state": [...]}` → action |
| `/chat` | POST | `{"message": "..."}` → response |
| `/info` | GET | Agent metadata |

---

## 🔧 Dependencies

Minimal requirements:
```
torch>=2.0
numpy
```

Optional for HTTP serving:
```
flask  # or fastapi + uvicorn
```

Optional for Gymnasium:
```
gymnasium
```

---

## 📦 Re-Exporting

The cocoon can re-export its neural models:

```python
from cocoon import CocoonAgent

agent = CocoonAgent()

# Export to ONNX for deployment
agent.export_onnx("brain.onnx")

# Export to TorchScript for C++/LibTorch
agent.export_torchscript("brain.pt")

# Save updated weights after learning
torch.save(agent.brain.state_dict(), "updated_weights.pth")
```

---

## 🦋 About the Butterfly System

This cocoon was generated by the **Butterfly Convergence Engine** - a neuro-symbolic AI framework that combines:

- **Neural networks** for pattern recognition and action selection
- **Atomic language** for grounded semantic understanding
- **VP regulation** (Vigilance × Plasticity) for adaptive attention
- **Knowledge webs** for relational reasoning
- **Distributed ensembles** for robust decision-making

Learn more: [Convergence Engine on GitHub](https://github.com/Yufok1/Convergence_Engine)

---

*Generated by 🦋 Butterfly Agent Compiler*
