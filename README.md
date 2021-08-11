# BSP-NET-2D-Pytorch

This code is a Pytorch implementation of BSP-Net 2D version that borrows from [bsp_2d](https://github.com/czq142857/BSP-NET-original/blob/master/bsp_2d/) and [BSP-NET-pytorch](https://github.com/czq142857/BSP-NET-pytorch/).

To run the code, install Python3.6, pytorch 1.2.0, h5py and opencv-python on either Ubuntu or Windows, then run *train_with_L_overlap.sh* or *train_without_L_overlap.sh* to train the model.
The results will be written to subfolder *samples* periodically.

You can use *data/create_dataset.py* to create a toy 2D dataset.