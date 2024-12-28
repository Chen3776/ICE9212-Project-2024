
Code Structure:

  --datafiles

    --caltech101

      --test

      --train

      --valid

      --cat-to-name.json

    --flowers102

      --test

      --train

      --valid

      --cat-to-name.json

  --datasets

    --caltech101.py  

    --flowers102.py  

    --process_caltech.py  

    --process_flowers102.py  

  --preprocess  

    --DatasetExtendWithStableDiffusion.py  


Usage:  

  Import torch-based dataset from /datasets/caltech101.py and /datasets/flowers102.py  
  
  Extend the original dataset by /preprocess/DatasetExtendWithStableDiffusion.py  
  
  Datafiles is zipped and saved in jbox: https://jbox.sjtu.edu.cn/l/c1O2Nw
