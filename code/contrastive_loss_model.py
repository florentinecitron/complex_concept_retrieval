from code.data import get_data

import numpy as np
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader, Dataset


class ManifestoDataset(Dataset):
    def __init__(self):
        self.texts, labels, self.queries = get_data()

        self.int2label = {idx: label for idx, label in enumerate(labels)}
        self.label2int = {label: idx for idx, label in enumerate(labels)}

        self.labels = [self.label2int[label] for label in labels]
        self.classes = list(set(self.labels))

    def __getitem__(self, index):
        anchor, target = self.texts[index], self.labels[index]
        # now pair this up with the correct query
        posneg = self.queries[self.int2label[target]]
        #return anchor, posneg, target
        return posneg, anchor, target

    def __len__(self):
        return self.texts.shape[0]


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X


class LinearProjection(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.hidden = nn.Linear(embedding_dim, embedding_dim,)

    def forward(self, X):
        return self.hidden(X)


class NonLinearProjection(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.hidden = nn.Linear(embedding_dim, embedding_dim)
        self.dropout_p = 0.5

    def forward(self, X):
        h = self.hidden(X)
        return nn.Dropout(self.dropout_p)(nn.ReLU()(h))


class MLNonLinearProjection(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        dropout_p = 0.3
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, X):
        return self.layers(X)


def main():
    manifesto_data = ManifestoDataset()
    lengths = [int(len(manifesto_data)*0.8), int(len(manifesto_data)*0.2)+1]
    manifesto_train, manifesto_test = torch.utils.data.random_split(manifesto_data, lengths)

    #dataloader = DataLoader(manifesto_train, batch_size=64,)

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=0.2)

    # Set the mining function
    miner = miners.TripletMarginMiner(margin=0.2)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(manifesto_data.classes, m=1, length_before_new_iter=len(manifesto_train))

    # Set other training parameters
    batch_size = 128

    # the metric learning training requires a "trunk" model,
    # so just create an n by n linear projection here
    trunk_model = LinearProjection(384)
    trunk_model_optimizer = optim.Adadelta(trunk_model.parameters())

    projection_model = NonLinearProjection(384)
    projection_model_optimizer = optim.Adadelta(projection_model.parameters())

    models = {"embedder": projection_model, "trunk": trunk_model}
    optimizers = {"embedder_optimizer": projection_model_optimizer, "trunk_optimizer": trunk_model_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}

    record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": manifesto_test}
    model_folder = "example_saved_models"

    # Create the tester
    tester = testers.GlobalTwoStreamEmbeddingSpaceTester(
        end_of_testing_hook = hooks.end_of_testing_hook,
        #visualizer = umap.UMAP(n_neighbors=50),
        #visualizer_hook = visualizer_hook,
        data_device=torch.device("cpu"),
        dataloader_num_workers = 32,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"))

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)

    trainer = trainers.TwoStreamMetricLoss(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        mining_funcs,
        manifesto_train,
        sampler=sampler,
        dataloader_num_workers=1,
        data_device=torch.device("cpu"),
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook
    )
    trainer.train(num_epochs=1)
    print(tester.all_accuracies)