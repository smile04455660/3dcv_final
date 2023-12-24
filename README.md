# 3dcv final
our work is based on the Gen6D github repo

## Gen6D

Gen6D is able to estimate 6DoF poses for unseen objects like the following video.

![](assets/example.gif)

Required packages are list in `requirements.txt`. To determine how to install PyTorch along with CUDA, please refer to the [pytorch-documentation](https://pytorch.org/get-started/locally/)


## File structure
Organize files like
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
