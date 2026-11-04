# Server queue: PROTEINS 10 folds, selector with a uniform fallback

The selector with uniformly random blocks in the overshoot fallback (plain-LBS blocks while the instance has had no improvement), arm `dce-lcls-sel-uniform-fallback` of `scripts/gen_selector_proteins_folds.py`. Its target: the two instances where the selector never left its initial counterfactual while plain LBS did (fold 1 id 275, fold 2 id 236), whose near-miss weighted fallback blocks all failed. Compared with the selector arm over the 10 folds, to check it fixes them without costing elsewhere.

- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-uniform-fallback/generate_minimize9.jsonc
