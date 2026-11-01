"""Save complete first-pass records without modifying Evaluator's old dump."""
import hashlib
import json
import os
import tempfile

import numpy as np

from src.evaluation.future.stages.stage import Stage


class ProbeDump(Stage):

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters'].setdefault('narratives_stage', 'src.evaluation.future.stages.probe_narratives.ProbeNarratives')

    def init(self):
        super().init()

    def process(self, explanation):
        saved = explanation.stages_info[self.local_config['parameters']['narratives_stage']]
        identity_config = {key: saved[key] for key in ('generator', 'schema_version', 'node_matching', 'feature_columns', 'atol')}
        config_hash = hashlib.sha256(json.dumps(identity_config, sort_keys=True, default=self._json_default).encode()).hexdigest()[:16]
        scope = self.context.conf['experiment'].get('scope', 'default_scope')
        run = getattr(self.context, 'run_number', -1)
        fold = explanation.explainer.fold_id
        directory = os.path.join(self.context.output_store_path, scope, explanation.dataset.name,
                                 explanation.oracle.name, explanation.explainer.name, 'probe_inputs',
                                 config_hash, f'run_{run}', f'fold_{fold}')
        os.makedirs(directory, exist_ok=True)
        identity = str(explanation.input_instance.id)
        filename = 'cf_' + hashlib.sha256(identity.encode()).hexdigest()[:16] + '.json'
        path = os.path.join(directory, filename)
        payload = {'id': identity, 'fold_id': fold, 'run_id': run, 'data': saved}
        # Atomic per-sample output; a failed write must fail the run visibly.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=directory, delete=False, encoding='utf-8') as f:
                temporary = f.name
                json.dump(payload, f, default=self._json_default, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.write_into_explanation(explanation, {'path': path, 'counterfactuals': len(saved['counterfactuals'])})
        return explanation

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot serialize {type(value).__name__}')
