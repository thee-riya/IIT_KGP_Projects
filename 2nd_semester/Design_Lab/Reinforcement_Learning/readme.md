# 🧹🤖 Q-Learning & DQN Cleaning Agent – 24CS60R70

Welcome to the **Grid Cleaning Challenge**! This project features a smart little agent 🧠 navigating a grid 🟦 to clean dirt 💩 using **Q-Learning** and **Deep Q-Networks (DQN)**.

It includes:
- 📄 Well-documented code
- 🧠 Smart heuristics
- 🧪 Evaluation & tuning
- ✨ Bonus multi-dirt support!

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `24CS60R70.QLearning.py` | 🧠 Q-Learning agent (single dirt cell) |
| `24CS60R70_DQN.py` | 🤖 DQN agent using PyTorch |
| `24CS60R70_Bonus.py` | 🧹 Bonus version with multiple dirts |
| `repoort.pdf` | 📊 Project report |
| `README.md` | 📘 You're reading it! |

---

## ⚙️ Requirements

Make sure you have Python ≥ 3.6 and install the dependencies:

```bash
pip install numpy matplotlib tqdm joblib gym torch
```

---

## ▶️ How to Run

### 🧠 Q-Learning Agent (Single Dirt)

```bash
python 24CS60R70.QLearning.py --grid_size 10 --episodes 1000
```

🔍 **Hyperparameter tuning:**

```bash
python 24CS60R70.QLearning.py --hyperparam
```

📈 **Evaluation mode:**

```bash
python 24CS60R70.QLearning.py --eval
```

---

### 🤖 DQN Agent (Deep Q-Learning)

```bash
python 24CS60R70_DQN.py --grid_size 10 --episodes 200
```

⚙️ **Tuning for best performance:**

```bash
python 24CS60R70_DQN.py --hyperparam
```

🧪 **Evaluate across environments:**

```bash
python 24CS60R70_DQN.py --eval
```

💾 **Save your trained model:**

```bash
python 24CS60R70_DQN.py --save my_dqn_model.pth
```

---

### ✨ Bonus: Multiple Dirt Cells

```bash
python 24CS60R70_Bonus.py --grid_size 10 --episodes 1000 --num_dirts 3
```

⚙️ **Tune for multiple dirts:**

```bash
python 24CS60R70_Bonus.py --hyperparam
```

🧪 **Evaluate:**

```bash
python 24CS60R70_Bonus.py --eval
```

---

## 🧾 Project Report

Check out the detailed report in [`repoort.pdf`](repoort.pdf) 📑.  
It includes methodology, experiments, results, and cool visualizations 📊.

---

## 🧠 Smart Agent Logic

🚀 **Heuristic-based Action Selection:**
- The agent moves toward the **nearest dirt cell** using **Manhattan distance** 🧮.
- If the move is invalid (wall/obstacle), fallback to Q-values or random exploration 🌀.
- Heuristics applied in both Q-Learning and DQN for smarter behavior!

📌 Modifications are clearly marked with `# 🧠 Modified` comments in each script.

---

## 📂 File Structure

```
.
├── 24CS60R70.QLearning.py       # 🧠 Q-Learning (Single Dirt)
├── 24CS60R70_DQN.py             # 🤖 DQN (Deep Q-Learning)
├── 24CS60R70_Bonus.py           # ✨ Q-Learning with Multiple Dirt Cells
├── report.pdf                  # 📊 Project Report
└── README.md                    # 📘 Interactive Guide
```

---

## 💡 Highlights

✅ Uses OpenAI Gym-style environment  
✅ Renders grid world with Twemoji CDN-based emoji support 🖼️  
✅ Heuristic-driven for realistic movement  
✅ Supports both single & multiple dirt modes  
✅ Includes post-training generalization tests across multiple environments

