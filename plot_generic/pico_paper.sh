#!/bin/bash

rm -rf plot_generic/leonardo/
rm -rf plot_generic/lumi/
rm -rf plot_generic/mare_nostrum/

# Ring algorithm
./plot_generic/heatmap.py --system leonardo --collective allreduce --nnodes 32,64,128,256,512,1024,2048 --target_algo ring_ompi --exclude bine,ring_over
./plot_generic/heatmap.py --system lumi --collective allreduce --nnodes 8,16,32,64,128,256,1024 --target_algo ring_over --exclude bine
./plot_generic/heatmap.py --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target_algo ring_ompi --exclude bine,ring_over --notes UCX_MAX_RNDV_RAILS=1

# Untuned default algorithm
./plot_generic/heatmap.py --system leonardo --collective allreduce --nnodes 32,64,128,256,512,1024,2048 --target_algo default_ompi --exclude bine,over
./plot_generic/heatmap.py --system lumi --collective allreduce --nnodes 8,16,32,64,128,256,1024 --target_algo default_mpich --exclude bine,over
./plot_generic/heatmap.py --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target_algo default_ompi --exclude bine,over --notes UCX_MAX_RNDV_RAILS=1