# Dual Hypergraph Contrastive Learning for Network Alignment


#### Requirements
`python==3.9.12`

`networkx==2.6.3`

`numpy==1.22.4`

`pytorch==1.12.1`

`scikit-learn==1.5.2`

`torch-geometric ==2.6.1`

`gensim==4.3.3`

#### Examples
If you want to run DualHCL algorithm on Douban online-offline dataset with training ratio 0.8, run the following command in the home directory of this project:

`run_DualHCL.sh`

or

`python DualHCL.py --s_edge data/douban/online.txt --t_edge data/douban/offline.txt --gt_path data/douban/node,split=0.8.test.dict --train_path data/douban/node,split=0.8.train.dict`
