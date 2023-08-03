1. install miniconda

[Link to website](https://docs.conda.io/en/latest/miniconda.html#)

2.  build and activate conda env
```
conda env create -f env.yml
conda activate render_tracebot
```

3. setup render_cfg.yaml

4. if models_info.json does not exist call
```
python create_bop_models_info_json.py render_cfg.yaml 
```

5. start rendering

 - Only rgb and depth
```
blenderproc run render_base.py render_cfg.yaml
```

- full bop_pipeline
```
./run.sh
```
