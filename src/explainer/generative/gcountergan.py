import torch

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer
from src.utils.samplers.abstract_sampler import Sampler
from src.utils.cfg_utils import init_dflts_to_of

class GCounteRGAN(PerClassExplainer):
    """
    GCounteRGAN: Generative Counterfactual Explanations using GANs

    This class extends the PerClassExplainer to provide counterfactual explanations
    using a generative model based on GANs.

    Attributes:
    - `sampler`: Sampler
        A sampler instance used for generating counterfactual instances.

    Methods:
    - `init(self) -> None`:
        Initializes the fields of the GCounteRGAN instance by setting up the 
        sampler and sending the models to the device.

    - `explain(self, instance) -> torch.Tensor`:
        Generates counterfactual explanations for the given input instance.

    - `check_configuration(self) -> None`:
        Checks the configuration of the GCounteRGAN instance, setting the prototype
        class and ensuring the presence of a sampler.

    """

    def init(self) -> None:
        """
        Initializes the fields of the instance.

        This method sets up the sampler and sends the models to the device.
        """
        super().init()
        self.sampler: Sampler = get_instance_kvargs(
            self.local_config['parameters']['sampler']['class'],
            self.local_config['parameters']['sampler']['parameters']
        )
        self.send_to_device()

    def explain(self, instance) -> torch.Tensor:
        """
        Generates counterfactual explanations for the given input instance.

        Args:
        - `instance`: torch.Tensor
            The input instance for which counterfactual explanations are generated.

        Returns:
        - `cf_instance`: torch.Tensor
            The counterfactual instance, or the original instance if no counterfactual
            instance is generated.
        """
        with torch.no_grad():
            res = super().explain(instance)

            embedded_features, edge_probs = dict(), dict()
            for key, prob_adj_matrix in res.items():
                embedded_features[key] = torch.from_numpy(instance.node_features)
                edge_probs[key] = prob_adj_matrix.squeeze()

            cf_instance = self.sampler.sample(
                instance, self.oracle, **{'embedded_features': embedded_features, 'edge_probabilities': edge_probs}
            )
        return cf_instance if cf_instance else instance

    def check_configuration(self) -> None:
        """
        Checks the configuration of the GCounteRGAN instance.

        This method sets the prototype class and ensures the presence of a sampler
        in the configuration.
        """
        self.set_proto_kls('src.explainer.generative.gans.image.model.GAN')
        super().check_configuration()
        # The sampler must be present in any case
        init_dflts_to_of(
            self.local_config, 'sampler', 'src.utils.samplers.bernoulli.Bernoulli', sampling_iterations=1
        )