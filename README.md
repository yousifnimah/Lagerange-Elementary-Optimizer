# LEO - Lagrange Elementary Optimization for Cloud Task Scheduling

This project is developed as part of the Master's studies in **Computer Science** to support research in **metaheuristic optimization algorithms**.  
It implements the **Lagrange Elementary Optimization (LEO)** algorithm for solving **Cloud Task Scheduling** and related optimization problems.

---

## 🎓 Author & Acknowledgement 
- **Project Owner (M.Sc. Research Student)**: *Hassanein Jameel*
- **Project / Collaboration**: *Yousif N. Abbas* (Refactoring and Code Optimizations)
- Based on the work of **Aso M. Aladdin & Tarik A. Rashid**

## 📂 Project Structure
```
.
├── datasets.py                        # Population / Dataset initialization
├── lagrange_elementary_optimizer.py   # LEO optimizer implementation
├── train.py                           # Training / running examples
├── readme.md                          # Project documentation
└── .venv                              # Virtual environment
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/leo-optimizer.git
   cd leo-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

If you don't have `requirements.txt`, create a file with:
```
numpy
matplotlib
```

---

## Usage

Run training example:
```bash
python train.py
```

Inside `train.py`, you can configure:
```python
from lagrange_elementary_optimizer import LEOOptimizer
from datasets import Population

# Example
pop = Population(size=50)
pop.initialize(lowB=-5, upB=15, numGs=2)

leo = LEOOptimizer(epochs=50, pop_size=50, num_g=2)
best, history = leo.fit()
print("Best solution:", best)
```

---

## Output

- Console log shows **best cost per iteration**  
- Final **best solution** position and cost are printed  
- Convergence curve plotted using Matplotlib  

---

## References

- Hassanein Jameel (Based on Aso M. Aladdin & Tarik A. Rashid)  
- Original paper on Lagrange Elementary Optimization  

---

## 📝 License

MIT License.  
Feel free to use, modify, and distribute.
