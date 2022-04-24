from argparse import ArgumentParser, Namespace
from pathlib import Path

import os
import json
import matplotlib.pyplot as plt

def main(args):

    training_logs = json.loads(Path(args.log_path).read_text())
    train = training_logs['train']
    eval = training_logs['eval']
    print("="*100, "\n", f"=> training_logs.keys()")
    print(training_logs['train'][0].keys())
    print(training_logs['eval'][0].keys(), "\n")
    
    #print(f"batch_size = {batch_size}")
    print("="*100, "\n", f"total epochs = {len(train)}", "\n")
    print(f"total Dataloader load times(step) = {len(train[0]['cum_avg_batch_loss'])}")
    print(f"total Optimizer update times（step / gradient_accumulation_step） = {train[0]['total_completed_steps (optimizer update)']}")
    
    for i in range(len(train)):
        print("="*100, "\n", f"epoch = {train[i]['epoch']}")
        # Info: Train
        print( "=> train:")
        print(f"    epoch_loss = {train[i]['epoch_loss']}")
        print(f"    epoch_acc = {train[i]['epoch_acc']}")
        print(f"    epoch_best_acc = {train[i]['epoch_best_acc']}, epoch_best_loss = {train[i]['epoch_best_loss']}", "\n")
        # Info: Eval
        print( "=> eval（per epoch）:")
        print(f"    epoch_acc = {eval[i]['epoch_acc']}")
        print(f"    epoch_best_acc = {eval[i]['epoch_best_acc']}", "\n")
    
    cum_avg_batch_loss = []
    cum_avg_batch_acc = []
    for i in range(len(train)):
        cum_avg_batch_loss += train[i]['cum_avg_batch_loss']
        cum_avg_batch_acc += train[i]['cum_avg_batch_acc']
    
    # Plot: cum_avg_batch_loss
    plt.figure("cum_avg_batch_loss")
    plt.title(f"Accumulate Average Batch Loss")
    plt.plot(cum_avg_batch_loss, label="train")
    plt.savefig(f"__Train__cum_avg_batch_loss.png")
    plt.close()
    # Plot: cum_avg_batch_acc
    plt.figure("cum_avg_batch_acc")
    plt.title(f"Accumulate Average Batch Accuracy")
    plt.plot(cum_avg_batch_acc, label="train")
    plt.savefig(f"__Train__cum_avg_batch_acc.png")
    plt.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to training_logs.json",
        required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()      
    main(args)






