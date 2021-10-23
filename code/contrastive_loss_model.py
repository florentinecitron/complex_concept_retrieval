from data import ManifestoDataset

import numpy as np
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning.distances import CosineSimilarity



class Identity(nn.Module):
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
        dropout_p = 0.2
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, X):
        return self.layers(X)


def train(network_type, epochs=2):
    assert network_type in ("linear", "nn")

    manifesto_data = ManifestoDataset.load()

    # TODO make sure that there should be some labels that are held out
    manifesto_train, manifesto_test = manifesto_data.split_by_holdout_labels()

    #dataloader = DataLoader(manifesto_train, batch_size=64,)

    # Set the loss function
    loss = losses.TripletMarginLoss(distance=CosineSimilarity())

    # Set the mining function
    miner = miners.TripletMarginMiner()

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(
        list(set(manifesto_data.labels)),
        m=1,
        length_before_new_iter=len(manifesto_train)
    )

    # Set other training parameters
    batch_size = 128

    # the metric learning training requires a "trunk" model,
    # so just create an n by n linear projection here;
    # the composition of these models will be linear too.
    trunk_model = LinearProjection(384)
    trunk_model_optimizer = optim.Adadelta(trunk_model.parameters())

    if network_type == "nn":
        embed_model = MLNonLinearProjection(384, 100)
    elif network_type == "linear":
        embed_model = LinearProjection(384)
    embed_model_optimizer = optim.Adadelta(embed_model.parameters())

    models = {"embedder": embed_model, "trunk": trunk_model}
    optimizers = {
        "embedder_optimizer": embed_model_optimizer,
        "trunk_optimizer": trunk_model_optimizer
    }
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
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count")
    )

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
    trainer.train(num_epochs=epochs)
    print(tester.all_accuracies)

    return tester, trunk_model, embed_model

def eval_cosine():
    manifesto_data = ManifestoDataset.load()

    # TODO make sure that there should be some labels that are held out
    manifesto_train, manifesto_test = manifesto_data.split_by_holdout_labels()


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
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count")
    )

    tester.test(dataset_dict, 1, Identity())



    print(tester.all_accuracies)

    return tester, trunk_model, embed_model


def results():
    cosine= {'val': {'epoch': 1,
              'AMI_level0': 0.24873027842900694,
              'NMI_level0': 0.24948181982787582,
              'mean_average_precision_level0': 0.30689698276907584,
              'mean_average_precision_at_r_level0': 0.26387371807075183,
              'precision_at_1_level0': 0.21179001721170396,
              'r_precision_level0': 0.295923781713968}}
    nn = {'val': {'epoch': 2, 'AMI_level0': 0.0923391042352795, 'NMI_level0': 0.0929133084283648, 'mean_average_precision_level0': 0.3813426309955886, 'mean_average_precision_at_r_level0': 0.31152548248617495, 'precision_at_1_level0': 0.21730165424809036, 'r_precision_level0': 0.39693066580539244}}
    linear = {'val': {'epoch': 1, 'AMI_level0': 0.13866463869321424, 'NMI_level0': 0.13971167740765608, 'mean_average_precision_level0': 0.41829824386550946, 'mean_average_precision_at_r_level0': 0.3354986656817992, 'precision_at_1_level0': 0.23833412619469793, 'r_precision_level0': 0.42161760661906034}}

    results = []
    for x in [cosine, linear, nn]:
        d = x["val"]
        del d["epoch"]
        results.append(d)
    df = pd.DataFrame(results).T
    df.columns = ["cosine", "linear", "nn"]
    print(tabulate.tabulate(df, floatfmt=".2f", tablefmt="latex", headers="keys"))
