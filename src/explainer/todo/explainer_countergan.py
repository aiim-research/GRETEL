from src.core.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance

import numpy as np
from torch.utils.data import TensorDataset
import os

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import wandb


class CounteRGANExplainer(Explainer):

    def __init__(self,
                 id,
                 explainer_store_path,
                 n_nodes,
                 n_labels=2,
                 batch_size_ratio=0.1,
                 training_iterations=20000,
                 n_discriminator_steps=2,
                 n_generator_steps=3,
                 ce_binarization_threshold=.5,
                 fold_id=0,
                 device='cpu',
                 config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.name = 'countergan'

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # activate cuda later
        self.device = 'cpu'
        
        self.batch_size_ratio = batch_size_ratio
        self.n_labels = n_labels
        self.n_nodes = n_nodes
        self.training_iterations = training_iterations
        self.n_discriminator_steps = n_discriminator_steps
        self.n_generator_steps = n_generator_steps
        self.explainer_store_path = explainer_store_path
        self.ce_binarization_threshold = ce_binarization_threshold
        self.fold_id = fold_id
        self._fitted = False
        
        # multi-class support
        self.explainers = [
            CounteRGAN(n_nodes,
                       residuals=True,
                       ce_binarization_threshold=ce_binarization_threshold).to(self.device) for _ in range(n_labels)
        ]
        
    def _get_softmax_label(self, desired_label=0):
        y = -1
        while True:
            y = np.random.randint(low=0, high=self.n_labels)
            if y != desired_label:
                break
        return y
    """def _get_explainer(self, ignore_class):
        y = -1
        while True:
            y = np.random.randint(low=0, high=self.n_labels)
            if y != ignore_class:
                break
        return self.explainers[y].generator"""

    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        if(not self._fitted):
            self.fit(oracle, dataset, self.fold_id)

        # Getting the scores/class of the instance
        pred_scores = oracle.predict_proba(instance)

        with torch.no_grad():
            torch_data_instance = torch.from_numpy(instance.to_numpy_array())[None, None, :, :]
            torch_data_instance = torch_data_instance.to(torch.float)
            explainer = self.explainers[np.argmax(pred_scores)].generator
            """ignore_index = np.argmax(pred_scores)"""
            explainer.eval()
            counterfactual = explainer(torch_data_instance).squeeze().cpu().numpy()
            cf_instance = DataInstance(instance.id)
            cf_instance.from_numpy_array(counterfactual, store=True)
            
            return cf_instance
    

    def save_explainers(self):
        for i, explainer in enumerate(self.explainers):
            torch.save(explainer.state_dict(),
                       os.path.join(self.explainer_store_path, self.name, f'explainer_{i}'))

    def load_explainers(self):
        for i in range(self.n_labels):
            self.explainers[i].load_state_dict(
                torch.load(
                    os.path.join(self.explainer_store_path, self.name, f'explainer_{i}')
                )
            )


    def fit(self, oracle: Oracle, dataset : Dataset, fold_id=0):
        explainer_name = f'countergan_fit_on_{dataset.name}_fold_id_{fold_id}'\
        + f'_batch_ratio_{self.batch_size_ratio}_training_iter_{self.training_iterations}'\
        + f'gen_steps_{self.n_generator_steps}_disc_steps_{self.n_discriminator_steps}'\
        + f'bin_threshold_{self.ce_binarization_threshold}'

        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)

        if os.path.exists(explainer_uri):
            # Load the weights of the trained model
            self.name = explainer_name
            self.load_explainers()

        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(explainer_uri)        
            self.name = explainer_name
            
            for i in range(self.n_labels):
                self.__fit(self.explainers[i],
                          oracle,
                          dataset,
                          fold_id,
                          desired_label=i)

            self.save_explainers()        

        # setting the flag to signal the explainer was already trained
        self._fitted = True

    def _check_divergence(self, generated_graph: torch.Tensor):
        return torch.all(torch.isnan(generated_graph))
    
    def _infinite_data_stream(self, loader: DataLoader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch
    
    def __fit(self, countergan, oracle : Oracle, dataset : Dataset, fold_id, desired_label=0):
        generator_loader, discriminator_loader = self.transform_data(dataset, fold_id, class_to_explain=desired_label)
        generator_loader = self._infinite_data_stream(generator_loader)
        discriminator_loader = self._infinite_data_stream(discriminator_loader)
     
        discriminator_optimizer = torch.optim.Adam(countergan.discriminator.parameters(),
                                                    lr=0.0002,
                                                    weight_decay=1e-4,
                                                    betas=(0.5, 0.999))
        
        countergan_optimizer = torch.optim.NAdam(countergan.parameters(),
                                                    lr=5e-4,
                                                    weight_decay=1e-4,
                                                    betas=(0.9, 0.5))
        
        loss_discriminator = nn.BCELoss(reduction='none')
        loss_countergan = nn.BCELoss()
        
        for iteration in range(self.training_iterations):
            G_losses, D_losses = [], []

            if iteration > 0:
                generated_graph = countergan.generator(graph)
                if self._check_divergence(generated_graph):
                    break
            
            discriminator_optimizer.zero_grad()
            countergan.set_training_generator(False)
            countergan.generator.train(False)
            countergan.set_training_discriminator(True)
            countergan.discriminator.train(True)
            
            for _ in range(self.n_discriminator_steps):
                # get the next batch of data
                graph, _ = next(discriminator_loader)
                graph = graph.to(self.device)
                # add the channels' dimension
                graph = graph[:,None,:,:]
                # generate the fake graph
                graph_fake, _ = next(generator_loader)
                graph_fake = graph_fake.to(self.device)
                graph_fake = graph_fake[:,None,:,:]
                graph_fake = countergan.generator(graph_fake)
                
                graph_batch = torch.cat([graph, graph_fake], dim=0)
                y_batch = torch.cat(
                    [
                        torch.ones((self.discrimnator_batch,)),
                        torch.zeros((self.generator_batch,))
                    ],
                    dim=0)
                
                # shuffle real and fake examples
                p = torch.randperm(len(y_batch))
                graph_batch, y_batch = graph_batch[p], y_batch[p]
                
                oracle_scores = []
                temp_instance = DataInstance(-1)
                for i in range(len(graph_batch)):
                    temp_instance.from_numpy_array(
                        graph_batch[i].to("cpu").detach().numpy().squeeze()
                    )
                    oracle_scores.append(
                        oracle.predict_proba(temp_instance)[self._get_softmax_label(desired_label)]
                    )
                # The following update to the oracle scores is needed to have
                # the same order of magnitude between real and generated sample
                # losses
                oracle_scores = np.array(oracle_scores, dtype=float).squeeze()
                real_samples = torch.where(y_batch == 1.)
                average_score_real_samples = np.mean(oracle_scores[real_samples])
                if average_score_real_samples != 0:
                    oracle_scores[real_samples] /= average_score_real_samples
                
                fake_samples = torch.where(y_batch == 0.)
                oracle_scores[fake_samples] = 1.
                
                y_pred = countergan.discriminator(graph_batch)
                
                loss = torch.mean(loss_discriminator(y_pred.squeeze(), y_batch.float())\
                    * torch.tensor(oracle_scores, dtype=torch.float))
                        
                D_losses.append(loss.item())
                loss.backward()
                discriminator_optimizer.step()
            
            countergan_optimizer.zero_grad() 
            countergan.set_training_generator(True)
            countergan.generator.train(True)
            countergan.set_training_discriminator(False)
            countergan.discriminator.train(False)
            
            for _ in range(self.n_generator_steps):
                graph, _ = next(generator_loader)
                graph = graph[:,None,:,:]
                graph = graph.to(self.device)
                y_fake = torch.ones((self.generator_batch, ))
                
                output = countergan(graph)
                
                loss = loss_countergan(output.squeeze(), y_fake.float())
                    
                loss.backward()
                G_losses.append(loss.item())
                countergan_optimizer.step()
                
            print(f'Iteration [{iteration}/{self.training_iterations}]'\
                    +f'\tLoss_D: {np.mean(D_losses)}\tLoss_G: {np.mean(G_losses)}')

           # wandb.log({
           # f'iteration_cls={desired_label}': iteration,
           # f'loss_d_cls={desired_label}_{self.fold_id}': np.mean(D_losses),
           # f'loss_g_cls={desired_label}_{self.fold_id}': np.mean(G_losses)
           # })
    

    def transform_data(self, dataset: Dataset, fold_id=0, class_to_explain=0):
        X  = np.array([i.to_numpy_array() for i in dataset.instances])
        y = np.array([i.graph_label for i in dataset.instances])

        X_train = X[dataset.get_split_indices()[fold_id]['train']]
        y_train = y[dataset.get_split_indices()[fold_id]['train']]
        
        class_to_explain_indices = np.where(y_train == class_to_explain)[0]
        generator_X = X_train[class_to_explain_indices]
        generator_y = y_train[class_to_explain_indices]
        
        class_to_not_explain_indices = np.where(y_train != class_to_explain)[0]
        discriminator_X = X_train[class_to_not_explain_indices]
        discriminator_y = y_train[class_to_not_explain_indices]
        
        self.generator_batch = int(len(generator_X) * self.batch_size_ratio)
        self.discrimnator_batch = int(len(discriminator_X) * self.batch_size_ratio)

        generator_dataset = TensorDataset(
            torch.tensor(generator_X, dtype=torch.float),
            torch.tensor(generator_y, dtype=torch.float)
            )
        
        discriminator_dataset = TensorDataset(
            torch.tensor(discriminator_X, dtype=torch.float),
            torch.tensor(discriminator_y, dtype=torch.float)
        )

        generator_loader = DataLoader(generator_dataset,
                                  batch_size=self.generator_batch,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
        
        discriminator_loader = DataLoader(discriminator_dataset,
                                                   batch_size=self.discrimnator_batch,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   drop_last=True)

        return generator_loader, discriminator_loader
    

class CounteRGAN(nn.Module):
    
    def __init__(self, n_nodes=28,
                 residuals=True,
                 ce_binarization_threshold=None):
        super(CounteRGAN, self).__init__()
        
        self.n_nodes = n_nodes
        self.residuals = residuals
        
        self.generator = ResidualGenerator(n_nodes=n_nodes,
                                           residuals=residuals,
                                           threshold=ce_binarization_threshold)
        
        self.discriminator = Discriminator(n_nodes=n_nodes)
        
    def set_training_discriminator(self, training):
        self.discriminator.set_training(training)
        
    def set_training_generator(self, training):
        self.generator.set_training(training)
        
    def forward(self, graph):
        graph = self.generator(graph)
        return self.discriminator(graph)

class ResidualGenerator(nn.Module):

    def __init__(self,
                 n_nodes=28,
                 residuals=True,
                 threshold=None):
        super(ResidualGenerator, self).__init__()

        self.n_nodes = n_nodes
        self.residuals = residuals
        self.threshold = threshold
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # activate cuda later
        self.device = 'cpu'

        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        self.fc = nn.Linear(in_features=64 * self.fc_len**2,
                            out_features=128 * (self.n_nodes // 4)**2)
        
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=128,
                                        out_channels=64,
                                        kernel_size=(4,4),
                                        stride=(2,2),
                                        padding=1)
        
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv2 = nn.ConvTranspose2d(in_channels=64,
                                        out_channels=64,
                                        kernel_size=(4,4),
                                        stride=(2,2),
                                        padding=1)
        
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.2)

        self.final_conv = nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size=(7,7),
                                    stride=(1,1),
                                    padding=self.__get_same_padding(7,1)
                                    )
        
        
    def init_run(self):
        with torch.no_grad():
            dummy_input = torch.randn((1,1, self.n_nodes, self.n_nodes)).to(self.device)
            x = self.conv2(self.conv1(dummy_input))

        return x.shape[-1]


    def forward(self, graph):
        x = self.conv1(graph)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.leaky_relu3(x)
        x = x.view((-1, 128, self.n_nodes//4, self.n_nodes//4))
        x = self.deconv1(x)
        x = self.leaky_relu4(x)
        x = self.deconv2(x)
        x = self.leaky_relu5(x)
        x = self.final_conv(x)
        x = torch.tanh(x) # Tanh values are in [-1, 1] so allow the residual to add or remove edges

        # building the counterfactual example from the union of the residuals and the original instance
        if self.residuals:
            x = torch.add(graph, x)
        # delete self loops
        for i in range(x.shape[0]):
            x[i][0].fill_diagonal_(0)
            
        mask = ~torch.eye(self.n_nodes, dtype=bool)
        mask = torch.stack([mask] * x.shape[0])[:,None,:,:]
        x[mask] = torch.sigmoid(x[mask])
        
        x = (torch.rand_like(x) < x).to(torch.float)
                      
        return x


    def __get_same_padding(self, kernel_size, stride):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    def set_training(self, training):
        self.training = training
    


class Discriminator(nn.Module):

    def __init__(self, n_nodes):
        super(Discriminator, self).__init__()

        self.n_nodes = n_nodes
        self.training = False
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # activate cuda later
        self.device = 'cpu'
        
        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2)
                            )
        
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout1 = nn.Dropout2d(p=0.4)

        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout2 = nn.Dropout2d(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
            
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout3 = nn.Dropout2d(p=0.4)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        self.fc = nn.Linear(64*self.fc_len**2, 1)



    def init_run(self):
        with torch.no_grad():
            dummy_input = torch.randn((1,1, self.n_nodes, self.n_nodes)).to(self.device)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))

        return x.shape[-1]


    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise


    def get_same_padding(self, kernel_size, stride, input_dim):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    def set_training(self, training):
        self.training = training

    def forward(self, graph):
        x = self.conv1(graph)
        if self.training:
            x = self.add_gaussian_noise(x)

        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = torch.sigmoid(self.fc(x))

        return x
    
