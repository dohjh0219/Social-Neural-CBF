# Social Robot Navigation with Neural CBF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()

**Data-driven safety for social robot navigation.** This repository implements **Neural Control Barrier Functions (Neural CBF)** to learn adaptive proxemic boundaries from the ATC pedestrian dataset, integrated into a Model Predictive Control (MPC) framework.

---

## 📖 Overview

Traditional social navigation relies on hand-crafted safety constraints (e.g., fixed elliptical potential fields), which often fail to capture the complexity and context-dependence of human spatial reasoning.

This project bridges **Control Theory** and **Deep Learning** by:
1.  **Baseline:** Implementing a standard MPC with Analytical CBF (Elliptical Social Zones).
2.  **Ours:** Learning a Neural CBF ($h_\theta(x)$) from large-scale human-human interaction data.
3.  **Control:** Synthesizing safety-critical control inputs using an MPC-CBF optimization solver (CasADi).

The goal is to enable robots to navigate distinctively "human" environments by learning implicit social norms directly from data.

---

## 🛠 Methodology

### 1. Analytical Baseline (Standard Approach)
We implement the method described in *Jang et al.*, using a geometric approximation of personal space.
* **Model:** Fixed Elliptical Social Zone.
* **Dynamics:** Dependent only on relative velocity and heading.
* **Limitation:** Requires manual tuning of semi-axes ($a, b$) and lacks adaptability to crowd density or complex interactions.

### 2. Neural CBF (Proposed Approach)
We replace the analytical barrier function with a learned neural network.
* **Data Source:** [ATC Pedestrian Dataset](https://dil.atr.jp/isl/ATCdataset/) (16k+ interaction trajectories).
* **Mechanism:**
    * Extract interaction pairs from the dataset.
    * Train a Multi-Layer Perceptron (MLP) to approximate the boundary where human comfort is maintained.
    * Enforce the CBF safety condition: $\Delta h(x, u) \geq -\gamma h(x)$.
* **Advantage:** The safety boundary ($h_\theta(x) = 0$) naturally adapts its shape based on the learned social context.

---