"""Shared configuration of independent probes, using the standard Stage contract."""
from src.core.factory_base import get_instance_kvargs
from src.evaluation.future.stages.stage import Stage
from src.utils import probe_inputs
from src.utils.probe_common import check_judge


class ProbeStage(Stage):
    probe = None

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'judge' not in p:
            raise ValueError('A probe requires an independently configured judge')
        p.setdefault('narratives_stage', probe_inputs.NARRATIVES_STAGE)
        p.setdefault('context', 'off')
        p.setdefault('mode', 'dict')
        p.setdefault('feature_match', 'transition')
        p.setdefault('control', True)
        p.setdefault('success', 'target')
        p.setdefault('edge_defaults', None)
        p.setdefault('preprocess', True)
        p.setdefault('judge_family', None)
        if p['context'] not in ('on', 'off') or p['mode'] not in ('full', 'dict'):
            raise ValueError('Invalid context or mode')
        if p['feature_match'] not in ('identity', 'transition') or p['success'] not in ('flip', 'target'):
            raise ValueError('Invalid feature-match or success criterion')
        if type(p['control']) is not bool or type(p['preprocess']) is not bool:
            raise ValueError('control and preprocess must be booleans')
        if self.probe == 'recourse' and p['context'] != 'off':
            raise ValueError('Recourse always receives the graph; its control is separate from CTX')

    def init(self):
        super().init()
        self.judge = None

    def process(self, explanation):
        p = self.local_config['parameters']
        if self.judge is None:
            self.judge = get_instance_kvargs(p['judge']['class'], {'context': self.context, 'local_config': p['judge']})
        saved = explanation.stages_info[p['narratives_stage']]
        check_judge(self.judge.local_config, saved, p['judge_family'])
        def predictor(instance):
            try:
                return explanation.oracle.predict(instance)
            finally:
                explanation.oracle._call_counter -= 1
        records = probe_inputs.run_saved(self.probe, self.judge, saved, p['context'], p['mode'],
                                        p['feature_match'], p['control'], predictor,
                                        explanation.dataset, p['success'], p['edge_defaults'],
                                        explanation.dataset.manipulate if p['preprocess'] else None)
        self.write_into_explanation(explanation, {'context': p['context'], 'mode': p['mode'],
                                                 'counterfactuals': records,
                                                 'summary': probe_inputs.summarize(self.probe, records)})
        return explanation
