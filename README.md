# 3dcv final
our work is based on the Gen6D github repo

## Gen6D

Gen6D is able to estimate 6DoF poses for unseen objects like the following video.

![](assets/example.gif)

Required packages are list in `requirements.txt`. To determine how to install PyTorch along with CUDA, please refer to the [pytorch-documentation](https://pytorch.org/get-started/locally/)


## File structure
Down load the pretrained models from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EkWESLayIVdEov4YlVrRShQBkOVTJwgK0bjF7chFg2GrBg?e=Y8UpXu) and organize files like this
```
Gen6D
|-- data
    |-- model
        |-- detector_pretrain
            |-- model_best.pth
        |-- selector_pretrain
            |-- model_best.pth
        |-- refiner_pretrain
            |-- model_best.pth
```

## Custom object
follow [custom_object.md](custom_object.md)

## How to execute
Once the data is present, run the command like,
```
python python my_predict.py --cfg configs/gen6d_pretrain.yaml --database custom/${database_name} --num 1 --std 2.5
```
