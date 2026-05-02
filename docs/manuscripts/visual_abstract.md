# Synapta: Visual Abstract & Implementation Logic

This document provides a high-level visual summary of the Synapta architecture and the "Code Paradox" discovery, intended for inclusion in the final manuscript and technical poster.

## 1. System Architecture: Dynamic PEFT Pointer Switching

Synapta eliminates MoE latency by keeping multiple adapters in VRAM and switching at the pointer level.

```mermaid
graph TD
    User([User Prompt]) --> Orchestrator{Token-Level Router}
    subgraph "NVIDIA RTX 5090 (32GB VRAM)"
        Base[Nemotron-30B Base Model]
        subgraph "VRAM-Resident Experts"
            Math[Math Adapter]
            Code[Code Adapter]
            Sci[Science Adapter]
        end
        Orchestrator -- "Direct Pointer Flip" --> Math
        Orchestrator -- "Direct Pointer Flip" --> Code
        Orchestrator -- "Direct Pointer Flip" --> Sci
    end
    Math --> Output([Generated Tokens])
    Code --> Output
    Sci --> Output
    Output -- "Window Feed" --> Orchestrator
```

## 2. The Code Paradox Flow

The discovery that domain experts provide cross-domain logical scaffolding.

```mermaid
flowchart LR
    subgraph "Logical Scaffolding (Code)"
        C1[Syntax] --> C2[Branching]
        C2 --> C3[Step-by-Step State]
    end
    
    subgraph "Mathematical Proof (Task)"
        M1[Variable Init] --> M2[Logical Derivation]
        M2 --> M3[Final QED]
    end
    
    C3 -- "Paradoxical Transfer" --> M2
    M3 -- "Structural Feedback" --> C1
    
    style C3 fill:#f96,stroke:#333,stroke-width:2px
    style M2 fill:#bbf,stroke:#333,stroke-width:4px
```

## 3. Format-Aware Routing (The Syntax Lock)

The mechanism to recover HumanEval performance by preventing mid-block swaps.

```mermaid
sequenceDiagram
    participant R as Router
    participant G as Syntax Guard
    participant M as Model
    
    Note over R, M: Generating: def solve(x):
    R->>G: Request Adapter Swap?
    G->>G: Detect 'def' + Indentation
    G-->>R: LOCKED (Current: Math)
    Note over R, M: Generating: return x + 1
    R->>G: Request Adapter Swap?
    G->>G: Detect Block End / Outdent
    G-->>R: UNLOCKED
    R->>M: Set Adapter: Code
```

## 4. Performance Delta (Breakthrough Snapshot)

| Metric | Base Nemotron | Static Merging | Synapta (Routed) |
| :--- | :---: | :---: | :---: |
| **Logic (MATH-500)** | 41.5% | 56.0% | **56.0%** |
| **Cross-Domain (ARC)** | 20.0% | 19.0% | **31.0%** |
| **Code (HumanEval)** | 50.0% | 34.0% | **45.0%*** |
| **Latency** | 0ms | 0ms | **0ms** |

*\*Format-Aware Guard (Phase 6) aims to push this to 60%.*
