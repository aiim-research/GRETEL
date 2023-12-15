from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset
class OracleFactory(Factory):      
    def get_oracle(self, oracle_snippet, dataset):
        inject_dataset(oracle_snippet, dataset)
        return self._get_object(oracle_snippet)
            
    def get_oracles(self, config_list, dataset):
        return [self.get_oracle(obj, dataset) for obj in config_list]
    
    """def get_oracle_by_name(self, oracle_dict, dataset: Dataset, emb_factory: EmbedderFactory) -> Oracle:

        oracle_name = oracle_dict['name']
        oracle_parameters = oracle_dict['parameters']

        # Check if the oracle is a KNN classifier
        if oracle_name == 'knn':
            if not 'k' in oracle_parameters:
                raise ValueError('''The parameter "k" is required for knn''')
            if not 'embedder' in oracle_parameters:
                raise ValueError('''knn oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_knn(dataset, emb, oracle_parameters['k'], -1, oracle_dict)

        # Check if the oracle is an SVM classifier
        elif oracle_name == 'svm':
            if not 'embedder' in oracle_parameters:
                raise ValueError('''svm oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_svm(dataset, emb, -1, oracle_dict)

        # Check if the oracle is an ASD Custom Classifier
        elif oracle_name == 'asd_custom_oracle':
            return self.get_asd_custom_oracle(oracle_dict)

        # Check if the oracle is an ASD Custom Classifier
        elif oracle_name == 'gcn-tf':
            return self.get_gcn_tf(dataset, -1, oracle_dict)

        elif oracle_name == 'gcn_synthetic_pt':
            return self.get_pt_syn_oracle(dataset, -1, oracle_dict)
            
        # Check if the oracle is a Triangles-Squares Custom Classifier
        elif oracle_name == 'trisqr_custom_oracle':
            return self.get_trisqr_custom_oracle(oracle_dict)

        # Check if the oracle is a Tree-Cycles Custom Classifier
        elif oracle_name == 'tree_cycles_custom_oracle':
            return self.get_tree_cycles_custom_oracle(oracle_dict) 
        
        elif oracle_name == 'cf2':
            if not 'converter' in oracle_parameters:
                raise ValueError('''The parameter "converter" is required for cf2''')
            
            converter_name = oracle_parameters['converter'].get('name')
            if not converter_name:
                raise ValueError('''The parameter "name" for the converter is required for cf2''')
            
            converter = None
            feature_dim = oracle_parameters.get('feature_dim', 36)
            weight_dim = oracle_parameters.get('weight_dim', 28)
            if converter_name == 'tree_cycles':
                converter = CF2TreeCycleConverter(feature_dim=feature_dim)
            else:
                converter = DefaultFeatureAndWeightConverter(feature_dim=feature_dim,
                                                              weight_dim=weight_dim)
            lr = oracle_parameters.get('lr', 1e-3)
            batch_size_ratio = oracle_parameters.get('batch_size_ratio', .1)
            weight_decay = oracle_parameters.get('weight_decay', 5e-4)
            epochs = oracle_parameters.get('epochs', 100)
            fold_id = oracle_parameters.get('fold_id', 0)
            threshold = oracle_parameters.get('threshold', .5)
            
            return self.get_cf2(dataset, converter, feature_dim, weight_dim, lr,
                                      weight_decay, epochs, batch_size_ratio,
                                      threshold, fold_id, oracle_dict)
        # If the oracle name does not match any oracle in the factory
        else:
            raise ValueError('''The provided oracle name does not match any oracle provided by the factory''')

    def get_cf2(self, dataset: Dataset, converter: ConverterAB, in_dim: int, h_dim: int, lr: float,
                weight_decay: float, epochs: int, batch_size_ratio: float, threshold: float, 
                fold_id: int, config_dict=None) -> Oracle:
        clf = CF2Oracle(id=self._oracle_id_counter,
                        oracle_store_path=self._oracle_store_path,
                        converter=converter,
                        in_dim=in_dim,
                        h_dim=h_dim,
                        lr=lr,
                        weight_decay=weight_decay,
                        epochs=epochs,
                        threshold=threshold,
                        batch_size_ratio=batch_size_ratio,
                        fold_id=fold_id,
                        config_dict=config_dict)
        self._oracle_id_counter += 1
        clf.fit(dataset, split_i=fold_id)
        return clf

    def get_knn(self, data: Dataset, embedder: Embedder, k, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = KnnOracle(id=self._oracle_id_counter,oracle_store_path=self._oracle_store_path,  emb=embedder, k=k, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_svm(self, data: Dataset, embedder: Embedder, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = SvmOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, emb=embedder, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_asd_custom_oracle(self, config_dict=None) -> Oracle:
        clf = ASDCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf

    def get_gcn_tf(self, data: Dataset, split_index=-1, config_dict=None) -> Oracle:
        clf = TfGCNOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(data, split_index)
        return clf
    
    def get_pt_syn_oracle(self, data: Dataset, split_index=-1, config_dict=None) -> Oracle:
        clf = SynNodeOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(data, split_index)
        return clf
        
    def get_trisqr_custom_oracle(self, config_dict=None) -> Oracle:
        clf = TrianglesSquaresCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf

    def get_tree_cycles_custom_oracle(self, config_dict=None) -> Oracle:
        clf = TreeCyclesCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf"""