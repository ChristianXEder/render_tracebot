#!/bin/sh
for i in {1..1} #{1..k} k*1000=how many images are rendered
do
  blenderproc run render_base.py render_cfg.yaml
done
python proc_to_bop_pyrender.py render_cfg.yaml