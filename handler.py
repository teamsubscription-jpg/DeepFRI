"""
RunPod serverless handler for DeepFRI.

Example job input:
    {"input": {"seq": "SMTDLLSAEDIKK...", "ontology": ["mf", "ec"]}}
    {"input": {"pdb": "<contents of a .pdb file>", "ontology": ["mf"], "saliency": true}}
"""
import os
import json
import tempfile

import runpod

from deepfrier.Predictor import Predictor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONFIG = os.environ.get('DEEPFRI_MODEL_CONFIG', os.path.join(APP_DIR, 'trained_models', 'model_config.json'))

# model paths in model_config.json are relative to the repo root
os.chdir(APP_DIR)

with open(MODEL_CONFIG) as json_file:
    PARAMS = json.load(json_file)

_predictors = {}


def get_predictor(net, ont):
    key = (net, ont)
    if key not in _predictors:
        params = PARAMS[net]
        _predictors[key] = Predictor(params['models'][ont], gcn=params['gcn'])
    return _predictors[key]


def handler(job):
    job_input = job.get('input', {})
    seq = job_input.get('seq')
    pdb = job_input.get('pdb')
    ontologies = job_input.get('ontology', ['mf'])
    if isinstance(ontologies, str):
        ontologies = [ontologies]
    saliency = bool(job_input.get('saliency', False))
    use_guided_grads = bool(job_input.get('use_guided_grads', False))

    if (seq is None) == (pdb is None):
        return {'error': "Provide exactly one of 'seq' (protein sequence) or 'pdb' (PDB file contents)."}
    invalid = [ont for ont in ontologies if ont not in ('mf', 'bp', 'cc', 'ec')]
    if invalid:
        return {'error': "Invalid ontology: %s. Choose from mf, bp, cc, ec." % invalid}

    net = 'cnn' if seq is not None else 'gcn'
    pdb_fn = None
    if pdb is not None:
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as f:
            f.write(pdb)
            pdb_fn = f.name

    try:
        output = {}
        for ont in ontologies:
            predictor = get_predictor(net, ont)
            predictor.predict(seq if seq is not None else pdb_fn)
            rows = sorted(predictor.prot2goterms['query_prot'], key=lambda x: x[2], reverse=True)
            result = {'predictions': [{'term': str(term), 'name': str(name), 'score': float(score)}
                                      for term, name, score in rows]}
            if saliency and ont in ('mf', 'ec'):
                predictor.compute_GradCAM(layer_name=PARAMS[net]['layer_name'], use_guided_grads=use_guided_grads)
                result['saliency_maps'] = predictor.pdb2cam.get('query_prot', {})
            output[ont] = result
        return output
    finally:
        if pdb_fn is not None:
            os.remove(pdb_fn)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
