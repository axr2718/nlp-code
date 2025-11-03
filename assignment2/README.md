# starter code for a2

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

One example on running the code:

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json``


**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``

---

## Hyperparameter Search

The FFNN implementation includes automated hyperparameter search functionality.

### Usage

```bash
python ffnn.py --train_data training.json --val_data validation.json --hyperparameter_search
```

Optional: specify epochs with `-e` flag (default is 10)

### Search Space

Tests **48 configurations** across:
- Hidden dimensions: [32, 64, 128, 256]
- Learning rates: [0.001, 0.005, 0.01, 0.05]
- Momentum: [0.85, 0.9, 0.95]

### Output Files

- `ffnn_search_results.pkl` - All configuration results with metrics
- `ffnn_best_model.pt` - Best performing model
- `ffnn_final_results.pkl` - Best configuration summary
- `ffnn_learning_curves.png` - Training/validation plots

