import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden = self.activation(self.W1(input_vector))

        output = self.W2(hidden)

        predicted_vector = self.softmax(output)
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


def train_model(model, optimizer, train_data, valid_data, epochs, return_metrics=False):
    """Training loop extracted from main block, optionally returns metrics"""
    train_losses = [] if return_metrics else None
    train_accuracies = [] if return_metrics else None
    val_losses = [] if return_metrics else None
    val_accuracies = [] if return_metrics else None
    
    print("========== Training for {} epochs ==========".format(args.epochs))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        epoch_loss = 0
        loss_count = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            if return_metrics:
                epoch_loss += loss.item()
                loss_count += 1
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        
        if return_metrics:
            train_losses.append(epoch_loss / loss_count)
            train_accuracies.append(correct / total)

        loss = None
        correct = 0
        total = 0
        val_epoch_loss = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        with torch.no_grad():
            minibatch_size = 16
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                    if return_metrics:
                        val_epoch_loss += example_loss.item()
                loss = loss / minibatch_size
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Validation time for this epoch: {}".format(time.time() - start_time))
            
            if return_metrics:
                val_losses.append(val_epoch_loss / total)
                val_accuracies.append(correct / total)
    
    if return_metrics:
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    return None

# ============================================================================
# HYPERPARAMETER SEARCH AND FINAL TRAINING
# ============================================================================

def hyperparameter_search(train_data, valid_data, vocab_size, search_epochs=10):
    """Search for best hyperparameters"""
    from itertools import product
    import pickle
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Define search grid
    hidden_dims = [32, 64, 128, 256]
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    momentums = [0.85, 0.9, 0.95]
    
    total_configs = len(hidden_dims) * len(learning_rates) * len(momentums)
    print(f"Testing {total_configs} configurations with {search_epochs} epochs each...")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Learning rates: {learning_rates}")
    print(f"Momentums: {momentums}")
    print("="*80 + "\n")
    
    results = []
    best_val_acc = 0
    best_config = None
    best_model = None
    config_num = 0
    start_time = time.time()
    
    for hidden, lr, momentum in product(hidden_dims, learning_rates, momentums):
        config_num += 1
        print(f"[{config_num}/{total_configs}] hidden={hidden}, lr={lr}, momentum={momentum}")
        
        random.seed(42)
        torch.manual_seed(42)
        
        model = FFNN(input_dim=vocab_size, h=hidden)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        metrics = train_model(model, optimizer, train_data, valid_data, search_epochs, return_metrics=True)
        
        max_val_acc = max(metrics['val_accuracies'])
        final_val_acc = metrics['val_accuracies'][-1]
        
        result = {
            'hidden_dim': hidden,
            'lr': lr,
            'momentum': momentum,
            'max_val_acc': max_val_acc,
            'final_val_acc': final_val_acc,
            'metrics': metrics
        }
        results.append(result)
        
        print(f"  → Max Val Acc: {max_val_acc*100:.2f}%, Final Val Acc: {final_val_acc*100:.2f}%")
        
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_config = result
            best_model = model
            print(f"  ✓ New best!")
        print()
    
    elapsed = time.time() - start_time
    print("="*80)
    print(f"Search complete in {elapsed/60:.1f} minutes")
    print(f"Best: hidden={best_config['hidden_dim']}, lr={best_config['lr']}, momentum={best_config['momentum']}")
    print(f"Best final validation accuracy: {best_val_acc*100:.2f}%")
    print("="*80 + "\n")
    
    # Save search results
    with open('ffnn_search_results.pkl', 'wb') as f:
        pickle.dump({'best_config': best_config, 'all_results': results}, f)
    print("Saved: ffnn_search_results.pkl\n")
    
    return best_model, best_config, results


def plot_results(metrics, best_config):
    """Generate plots for train/val loss and accuracy"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Run: pip install matplotlib")
        return
    
    print("="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Create 2-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, metrics['train_losses'], 'b-', linewidth=2, marker='o', 
             markersize=6, label='Training Loss')
    ax1.plot(epochs, metrics['val_losses'], 'r-', linewidth=2, marker='s', 
             markersize=6, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('FFNN Loss by Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, [acc * 100 for acc in metrics['train_accuracies']], 'b-', 
             linewidth=2, marker='o', markersize=6, label='Training Accuracy')
    ax2.plot(epochs, [acc * 100 for acc in metrics['val_accuracies']], 'r-', 
             linewidth=2, marker='s', markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('FFNN Accuracy by Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ffnn_learning_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: ffnn_learning_curves.png")
    print("="*80 + "\n")


def run_hyperparameter_search_pipeline(train_data_file, val_data_file, epochs=10):
    """Complete pipeline: search, train, save, plot"""
    import pickle
    
    print("\n" + "="*80)
    print("FFNN HYPERPARAMETER SEARCH PIPELINE")
    print("="*80)
    print(f"Training data: {train_data_file}")
    print(f"Validation data: {val_data_file}")
    print(f"Epochs: {epochs}")
    print("="*80 + "\n")
    
    # Load and prepare data
    print("Loading data...")
    train_data_raw, valid_data_raw = load_data(train_data_file, val_data_file)
    vocab = make_vocab(train_data_raw)
    vocab, word2index, index2word = make_indices(vocab)
    
    print(f"Loaded {len(train_data_raw)} training samples")
    print(f"Loaded {len(valid_data_raw)} validation samples")
    print(f"Vocabulary size: {len(vocab)}\n")
    
    print("Vectorizing data...")
    train_data = convert_to_vector_representation(train_data_raw, word2index)
    valid_data = convert_to_vector_representation(valid_data_raw, word2index)
    print("Data vectorized\n")
    
    # Hyperparameter search
    best_model, best_config, all_results = hyperparameter_search(
        train_data, valid_data, len(vocab), epochs)
    
    # Save best model
    torch.save(best_model.state_dict(), 'ffnn_best_model.pt')
    print("Saved model: ffnn_best_model.pt")
    
    # Save results
    final_results = {
        'best_config': best_config,
        'metrics': best_config['metrics'],
        'vocab_size': len(vocab)
    }
    with open('ffnn_final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    print("Saved results: ffnn_final_results.pkl\n")
    
    # Plot
    plot_results(best_config['metrics'], best_config)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - ffnn_search_results.pkl (all search results)")
    print("  - ffnn_best_model.pt (trained model)")
    print("  - ffnn_final_results.pkl (best config + metrics)")
    print("  - ffnn_learning_curves.png (plots)")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, default=10, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--hyperparameter_search', action='store_true', help = "run hyperparameter search")
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    if args.hyperparameter_search:
        # Run hyperparameter search pipeline
        run_hyperparameter_search_pipeline(
            train_data_file=args.train_data,
            val_data_file=args.val_data,
            epochs=args.epochs
        )
    else:
        # Regular training mode
        if args.hidden_dim is None:
            parser.error("--hidden_dim is required when not using --hyperparameter_search")
        
        # load data
        print("========== Loading data ==========")
        train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
        vocab = make_vocab(train_data)
        vocab, word2index, index2word = make_indices(vocab)

        print("========== Vectorizing data ==========")
        train_data = convert_to_vector_representation(train_data, word2index)
        valid_data = convert_to_vector_representation(valid_data, word2index)
        

        model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
        optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
        train_model(model, optimizer, train_data, valid_data, args.epochs, return_metrics=False)
        # write out to results/test.out
