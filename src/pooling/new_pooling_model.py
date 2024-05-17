"""The file with the pooling model."""

import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch
from ptls.data_load.utils import collate_feature_dict
from torch import nn
from tqdm import tqdm

class SimpleNN(nn.Module):
    """Fully-connected neural network model with ReLU activations"""
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int],
                 output_size: int = None
                 ) -> None:
        """Initialize method for SimpleNN.
        
        Args:
        ----
            input_size (int): Input size
            hidden_sizes (list[int]): Hidden sizes (if output_size is None then the last size is output size)
            output_size (int): Output
         """
        super().__init__()
        layers = nn.ModuleList([])
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        if output_size is not None:
            layers.append(nn.Linear(prev_size, output_size))
        self.layers = layers
    
    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = nn.functional.relu(out)
            out = layer(out)
        return out

class PoolingModel(nn.Module):
    """PytorchLightningModule for local validation of backbone with pooling of information of different users."""

    def __init__(
        self,
        train_data: list[dict],
        backbone: nn.Module,
        backbone_embd_size: int,
        pooling_type: str = "mean",
        max_users_in_train_dataloader: int = 3000,
        min_seq_length: int = 15,
        max_seq_length: int = 1000,
        max_embs_per_user: int = 1000,
        init_device: str = "cuda",
        freeze: bool = True,
        additional_config: dict = None
    ) -> None:
        """Initialize method for PoolingModel.

        Args:
        ----
            train_data (list[dict]): Dataset for calculating embedings
            backbone (nn.Module):  Local embeding model
            pooling_type (str): "max", "mean", "attention" or "learnable_attention", type of pooling
            backbone_embd_size (int): Size of local embedings from backbone model
            max_users_in_train_dataloader (int): Maximum number of users to save
                in self.embegings_dataset
            min_seq_length (int): Minimum length of sequence for user
                in self.embegings_dataset preparation
            max_seq_length (int): Maximum length of sequence for user
                in self.embegings_dataset preparation
            max_embs_per_user (int): How many datapoints to take from one user
            in self.embegings_dataset preparation
            init_device (str): Name of device to use during initialization
            freeze (bool): Flag
        """
        super().__init__()
        if pooling_type not in ["mean", "max", "attention", "learnable_attention", 
                                "symmetrical_attention", "kernel_attention", "exp_hawkes",
                                "learnable_hawkes", "exp_learnable_hawkes", "attention_hawkes"]:
            raise ValueError("Unknown pooling type.")

        self.backbone = backbone.to(init_device)
        self.backbone.eval()
        self.backbone_embd_size = backbone_embd_size

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pooling_type = pooling_type
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.max_embs_per_user = max_embs_per_user

        self.embegings_dataset = self.make_pooled_embegings_dataset(
            train_data, max_users_in_train_dataloader, init_device
        )

        if pooling_type == "learnable_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.backbone_embd_size, self.backbone_embd_size
            )
        elif pooling_type == "symmetrical_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.backbone_embd_size, additional_config["hidden_size"]
            )
        elif pooling_type == "kernel_attention":
            self.attention_kernel = SimpleNN(
                input_size=self.backbone_embd_size,
                hidden_sizes=additional_config["hidden_sizes"]
            )
        elif pooling_type == "learnable_hawkes":
            self.hawkes_nn = SimpleNN(
                input_size=self.backbone_embd_size * 2,
                hidden_sizes=additional_config["hawkes_nn"]["hidden_sizes"],
                output_size=self.backbone_embd_size
            )
            self.hawkes_time_nn = SimpleNN(
                input_size=1,
                hidden_sizes=additional_config["hawkes_time_nn"]["hidden_sizes"],
                output_size=1
            )
        elif pooling_type == "exp_learnable_hawkes":
            self.hawkes_nn = SimpleNN(
                input_size=self.backbone_embd_size * 2,
                hidden_sizes=additional_config["hidden_sizes"],
                output_size=self.backbone_embd_size
            )
            
    def prepare_data_for_one_user(self, x, device):
        """Prepare one user's embedings and last times for this embedings.

        Args:
        ----
            x (dict): Data from one user
            device (str): Name of device to use

        Return:
        ------
            embs (np.array): Embeddings of slices of one user
            times (np.array): Times of last transaction in slices of ine user
        """
        resulting_user_data = []
        indexes = np.arange(self.min_seq_length, len(x["event_time"]))
        times = x["event_time"][indexes - 1].cpu().numpy()
        for i in indexes:
            start_index = 0 if i < self.max_seq_length else i - self.max_seq_length
            data_for_timestamp = FeatureDict.seq_indexing(x, range(start_index, i))
            resulting_user_data.append(data_for_timestamp)
        resulting_user_data = collate_feature_dict(resulting_user_data)
        embs = self.backbone(resulting_user_data.to(device))
        if self.backbone.is_reduce_sequence:
            embs = embs.detach().cpu().numpy()
        else:
            embs = embs.payload[:, -1, :].detach().cpu().numpy()
        return embs, times

    def make_pooled_embegings_dataset(
        self, train_data: list[dict], max_users_in_train_dataloader: int, device: str
    ) -> dict[int, torch.Tensor]:
        """Create pooled embeding dataset.

        This function for each timestamp get sequences in dataset which ends close to this timestamp,
        make local embedding out of them and pool them together

        Args:
        ----
            train_data (list[dict]): data for calculating global embedings
                from local sequences
            max_users_in_train_dataloader (int): max dataset size.
            device (str): name of device to use

        Return:
        ------
            embegings_dataset(dict): dictionary containing timestamps and pooling
                vectors for that timestamps
        """
        data = []
        all_times = set()
        for x in tqdm(train_data):
            # check if the user's sequence is not long enough
            if len(x["event_time"]) <= self.min_seq_length:
                continue

            data.append({})
            embs, times = self.prepare_data_for_one_user(x, device)
            all_times.update(times)
            for emb, time in zip(embs, times):
                data[-1][time] = emb

            # check if the number of users is enough
            if len(data) >= max_users_in_train_dataloader:
                break

        all_times = list(all_times)
        all_times.sort()
        self.min_time = np.min(all_times)
        embegings_dataset = self.reshape_time_user_dataset(all_times, data)
        self.times = np.sort(list(embegings_dataset.keys()))

        return embegings_dataset

    def reshape_time_user_dataset(
        self, all_times: list, data: list[dict]
    ) -> dict[int, torch.Tensor]:
        """Reshaping of time-users-embeddings data.

        from list of users with Dicts
        with time keys to dict with time keys and values with concatenated users
        embeddings

        Args:
        ----
            all_times (list): list of all time points that can be found in data
            data (list[dict]): list of users containing the time as a keys and
            embeddings as values

        Return:
        ------
            embegings_dataset(dict): dictionary containing timestamps as keys and pooling
                vectors and timestamps as values for that timestamps {timestamp: {"embs": ..., "times": ...,}, ...}
        """
        embegings_dataset = {}
        for time in tqdm(all_times):
            embs_for_this_time = []
            times_for_this_time = []
            for user_data in data:
                user_times = list(user_data.keys())
                user_times.sort()
                user_times = np.array(user_times)

                index = np.searchsorted(user_times, time, "right") - 1
                if index < 0:
                    continue

                closest_time = user_times[index]
                closest_emb = user_data[closest_time]
                embs_for_this_time.append(closest_emb)
                times_for_this_time.append(closest_time)
            if len(embs_for_this_time) > 0:
                embegings_dataset[time] = {"embs": np.stack(embs_for_this_time),
                                           "times": np.stack(times_for_this_time)}

        return embegings_dataset

    def local_forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Local forward method (just pass batch through the backbone).

        Args:
        ----
            batch (PaddedBatch): batch of data

        Return:
        ------
            out (torch.Tensor): embedings of data in batch
        """
        out = self.backbone(batch)
        return out

    def global_forward(
        self, batch: PaddedBatch, batch_of_local_embedings: torch.Tensor
    ) -> torch.Tensor:
        """Global forward method ().

        Args:
        ----
            batch (PaddedBatch): batch of data
            batch_of_local_embedings (torch.Tensor): embedings of data in batch

        Return:
        ------
            batch_of_global_poolings (torch.Tensor): global embedings for
                last times for data in batch
        """
        if self.backbone.is_reduce_sequence == False:
            batch_of_local_embedings = batch_of_local_embedings.payload
            bs, ts, ls = batch_of_local_embedings.shape
            batch_of_local_embedings = batch_of_local_embedings.reshape(-1, ls)
            batch_event_times = batch.payload["event_time"].reshape(-1)
        else: 
            batch_event_times = batch.payload["event_time"]

        batch_of_global_poolings = []
        prev_time = 0

        for event_time_seq, user_emb in zip(
            batch_event_times, batch_of_local_embedings
        ):
            
            max_time = event_time_seq.max().item()
            if max_time == 0: 
                max_time = prev_time
            prev_time = max_time

            local_pooled_emb = self.make_local_pooled_embedding(
                max_time, user_emb
            )
            batch_of_global_poolings.append(local_pooled_emb)
        batch_of_global_poolings = torch.stack(batch_of_global_poolings)

        if self.backbone.is_reduce_sequence == False:
            ls = batch_of_global_poolings.shape[-1]
            batch_of_global_poolings = batch_of_global_poolings.reshape(bs, ts, ls)

        return batch_of_global_poolings

    def forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Forward method (makes local and global forward and concatenate them).

        Args:
        ----
            batch (PaddedBatch): batch of data

        Return:
        ------
            out (torch.Tensor): concatenated local and global forward outputs
        """
        batch_of_local_embedings = self.local_forward(batch)
        batch_of_global_poolings = self.global_forward(batch, batch_of_local_embedings)
        
        if self.backbone.is_reduce_sequence == False:
            out = batch_of_local_embedings
            out._payload = torch.cat(
                (
                    batch_of_local_embedings.payload,
                    batch_of_global_poolings.to(batch_of_local_embedings.payload.device),
                ),
                dim=-1,
            )
        else:
            out = torch.cat(
                (
                    batch_of_local_embedings,
                    batch_of_global_poolings.to(batch_of_local_embedings.device),
                ),
                dim=-1,
            )
            

        return out
    

    def make_local_pooled_embedding(
        self, time: float, user_emb: torch.Tensor
    ) -> torch.Tensor:
        """Find the closest timestamp in self.embegings_dataset and return the pooling vector at this timestamp.

        Args:
        ----
            time (float): timepoint for which we are looking for pooling vector
            user_emb (torch.Tensor): the desired user embedding.

        Return:
        ------
            pooled_vector (torch.Tensor): pooling vector for given timepoint

        """
        if time < self.min_time:
            warnings.warn(
                "Attention! Given data was before than any in train dataset. Pooling vector is set to random."
            )

            pooled_vector = torch.rand(self.backbone_embd_size)

        else:
            index = np.searchsorted(self.times, time, "right") - 1
            closest_time = self.times[index]
            vectors = self.embegings_dataset[closest_time]["embs"]
            times = self.embegings_dataset[closest_time]["times"]
            vectors = np.stack(vectors)
            times = np.stack(times)

            if len(vectors) >= self.max_embs_per_user:
                indexes = np.argsort(times)[-self.max_embs_per_user:]
                # indexes = np.random.choice(np.arange(len(vectors)), self.max_embs_per_user, replace=False)
                vectors = vectors[indexes]
                times = times[indexes]

            
            if self.pooling_type == "mean":
                pooled_vector = torch.Tensor(np.mean(vectors, axis=0))

            elif self.pooling_type == "max":
                pooled_vector = torch.Tensor(np.max(vectors, axis=0))

            elif "attention" in self.pooling_type:
                vectors = torch.Tensor(vectors).to(user_emb.device)

                if self.pooling_type == "learnable_attention":
                    if not self.learnable_attention_matrix:
                        raise ValueError("Learnable attention matrix wasn't initialized!")
                    vectors_prep = self.learnable_attention_matrix(vectors)
                elif self.pooling_type == "symmetrical_attention":
                    if not self.learnable_attention_matrix:
                        raise ValueError("Learnable attention matrix wasn't initialized!")
                    vectors_prep = self.learnable_attention_matrix(vectors)
                    user_emb = self.learnable_attention_matrix(user_emb)
                elif self.pooling_type == "kernel_attention":
                    if not self.attention_kernel:
                        raise ValueError("Attention kernel wasn't initialized!")
                    vectors_prep = self.attention_kernel(vectors)
                    user_emb = self.attention_kernel(user_emb)
                else: vectors_prep = vectors
                dot_prod = vectors_prep @ user_emb.unsqueeze(0).transpose(1, 0)
                softmax_dot_prod = nn.functional.softmax(dot_prod, 0)
                if self.pooling_type == "attention_hawkes":
                    times = torch.Tensor(times - time).to(user_emb.device)
                    times = times.unsqueeze(-1)
                    time_part = torch.exp(times)
                    softmax_dot_prod = softmax_dot_prod * time_part
                pooled_vector = (softmax_dot_prod * vectors).sum(dim=0)
            
            elif self.pooling_type == "exp_hawkes":
                pooled_vector = np.mean(vectors * np.exp(times - time).reshape(-1,1), axis=0)
                pooled_vector = torch.Tensor(pooled_vector).to(user_emb.device)
                
            elif "hawkes" in self.pooling_type:
                if not self.hawkes_nn:
                    raise ValueError("Hawkes NN wasn't initialized!")
                
                vectors = torch.Tensor(vectors).to(user_emb.device)
                concated = torch.cat((vectors, user_emb.tile((len(vectors), 1))), axis=1)
                emb_part = self.hawkes_nn(concated)
                times = torch.Tensor(times - time).to(user_emb.device)
                times = times.unsqueeze(-1)
                if self.pooling_type == "learnable_hawkes":
                    if not self.hawkes_time_nn:
                        raise ValueError("Hawkes NN for time wasn't initialized!")
                    time_part = self.hawkes_time_nn(times)
                elif self.pooling_type == "exp_learnable_hawkes":
                    time_part = torch.exp(times)
                else:
                    raise ValueError("Unsupported pooling type.")
                pooled_vector = torch.sum(emb_part * time_part, axis=0) 

            else:
                raise ValueError("Unsupported pooling type.")
            
        device = next(self.parameters()).device
        return pooled_vector.to(device)

    @property
    def embedding_size(self) -> int:
        """Function that return the output size of the model.

        Return: output_size (int): the output size of the model
        """
        return 2 * self.backbone_embd_size  # type: ignore

    def set_pooling_type(self, pooling_type: str, additional_config: dict = None) -> None:
        """Change pooling type of the model."""
        if pooling_type not in  ["mean", "max", "attention", "learnable_attention", 
                                "symmetrical_attention", "kernel_attention", "exp_hawkes",
                                "learnable_hawkes", "exp_learnable_hawkes", "attention_hawkes"]:
            raise ValueError("Unknown pooling type.")
        if pooling_type == self.pooling_type :
            raise ValueError("Current pooling type is equal to the new pooling type.")
    
        if pooling_type == "learnable_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.backbone_embd_size, self.backbone_embd_size
            )
        elif pooling_type == "symmetrical_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.backbone_embd_size, additional_config["hidden_size"]
            )
        elif pooling_type == "kernel_attention":
            self.attention_kernel = SimpleNN(
                input_size=self.backbone_embd_size,
                hidden_sizes=additional_config["hidden_sizes"]
            )
        elif pooling_type == "learnable_hawkes":
            self.hawkes_nn = SimpleNN(
                input_size=self.backbone_embd_size * 2,
                hidden_sizes=additional_config["hawkes_nn"]["hidden_sizes"],
                output_size=self.backbone_embd_size
            )
            self.hawkes_time_nn = SimpleNN(
                input_size=1,
                hidden_sizes=additional_config["hawkes_time_nn"]["hidden_sizes"],
                output_size=1
            )
        elif pooling_type == "exp_learnable_hawkes":
            self.hawkes_nn = SimpleNN(
                input_size=self.backbone_embd_size * 2,
                hidden_sizes=additional_config["hidden_sizes"],
                output_size=self.backbone_embd_size
            )

        self.pooling_type = pooling_type
