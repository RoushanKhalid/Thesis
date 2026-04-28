# A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning

Welcome to the main repository for the thesis project on **Retrieval-Based Multi-Modal Learning**. This project implements a novel framework designed for continuous, incremental learning without the costly process of repeatedly fine-tuning model weights. 

By utilizing an **Intelligent Memory Orchestrator (IMO)** and an external vector database, this system provides near-real-time knowledge integration, high robustness, and hardware efficiency compared to traditional neural network fine-tuning.

---

## 📖 Abstract

Modern multi-modal models achieve strong performance but struggle with continuous knowledge integration. Traditional adaptation depends on fine-tuning, which introduces high computational costs, risks of catastrophic forgetting, and slow deployment cycles.

This thesis proposes a **non-parametric, retrieval-centric framework**. Instead of fine-tuning the backbone network, new data is encoded into embeddings and stored in an external vector memory (ChromaDB). A dedicated IMO layer controls memory quality via confidence gating, prototype consistency, and outlier filtering. During inference, retrieval evidence is dynamically fused with the base model to form the final prediction.

---

## 🗂️ Repository Structure

* `tcontext/` - The core implementation, pipeline, and web demo for the "Cats vs Dogs" incremental learning experiment.
  * *`quick500_experiment.py`*: Full evaluation pipeline (training, testing, report creation).
  * *`web_app.py`*: Streamlit interactive dashboard.
  * *`query_demo.py`*: CLI utility for incremental memory updates.
  * *`vector_db/`*: The persistent ChromaDB vector store.
* `logs/` - Execution logs and script outputs.
* `proposal/` - Initial thesis outlines and research proposals.
* `thesis_enrichment.md` - Core theoretical formulation and architecture rules (IMO design, mathematical models).
* `run_demo.ps1` - PowerShell script to initialize the demo and web app environments.

---

## ⚡ Core Concepts

### 1. Separation of Memory
The framework strictly separates knowledge into two layers:
- **Parametric Memory**: A frozen encoder (pre-trained, such as `EfficientNetB0` or `CLIP`) that extracts rich feature embeddings.
- **Non-parametric Memory**: A dynamic external vector database ($V$) storing embeddings, labels, and quality scores.

### 2. Intelligent Memory Orchestrator (IMO)
When new data arrives, instead of directly appending to the database, the IMO acts as a quality gate:
- **Consistency Check**: Ensures the new sample aligns with the prototype cluster.
- **Conflict Detection**: Quarantines noisy labels or conflicting data points.
- **Temporal Weighting**: Prevents stale information from skewing new predictions.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed and the necessary pip dependencies for the project:
```bash
pip install tensorflow keras chromadb streamlit openai-clip pandas matplotlib
```

### Running the Full Comparative Demo
To run the automated setup and launch the Streamlit dashboard comparing the fixed base classifier against the dynamic retrieval framework:

```powershell
# From the repository root:
.\run_demo.ps1
```
The script will ensure all assets are prepared and automatically launch the Web UI on `http://localhost:8501`. 

### Using the Streamlit Dashboard
Inside the dashboard, you can:
1. Examine the **Cost VS Accuracy** differences between parametric fine-tuning & vector indexing.
2. View the full **Exploratory Data Analysis (EDA)**.
3. Use the **Live Inference module** to upload an image and compare the Base Model prediction against the Retrieval Memory prediction.
4. **Demonstrate Incremental Learning**: Upload a *new* labeled sample to instantly update the vector database and immediately query it again, demonstrating 0-latency adaptation without restarting or retraining the model.

---

## 🖥️ Standalone CLI Commands

You can run experiments or query operations manually from the CLI inside the `tcontext` folder:

**Run the static model vs retrieval benchmark:**
```bash
python tcontext/quick500_experiment.py --seed 777
```

**Query an image directly from the CLI:**
```bash
python tcontext/query_demo.py --query-image "path/to/cat.jpg" --top-k 5
```

**Add a new concept/image to the database instantly:**
```bash
python tcontext/query_demo.py --add-image "path/to/new_dog.png" --label dogs
```

---

## 📊 Experimental Results & Benchmarks

Our latest benchmarks (available under `tcontext/reports/`) test the system on the Microsoft Cats vs Dogs dataset using a 25% stratified sampling setup to control overfitting. 

Using **CLIP (ViT-B/32) + ChromaDB** against a baseline **EfficientNetB0**:
- **Retrieval Test Accuracy**: Matches or slightly outperforms the trained parameterization (e.g., ~98.8% vs ~98.0%).
- **Adaptation Time**: Instantaneous index operation (O(1) seconds) vs iterative backpropagation (O(N) minutes/hours).
- **Forgetting Risk**: 0% (Controlled purely via Explicit Document Management).

Please refer to the `thesis_enrichment.md` file for full mathematical formulations and the ablation study blueprint.
