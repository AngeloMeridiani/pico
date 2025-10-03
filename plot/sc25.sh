#!/bin/bash
echo "Generating tables ..."
./plot/tables.sh &> /dev/null

echo "Generating boxplots ..."
./plot/boxplots.sh &> /dev/null

echo "Generating heatmaps ..."
./plot/heatmaps.sh &> /dev/null

echo "Generating fig. 5 ..."
pushd tracer/sinfo
python3 plot.py &> /dev/null
popd

mkdir -p plot/paper

# Fig. 5
cp tracer/sinfo/multi_box_min_None_allreduce_rabenseifner_vs_bine_bandwidth.pdf plot/paper/fig5.pdf

# Fig. 8a
cp plot/lumi/heatmaps/allreduce/tasks_per_node_1_metric_mean_base_all_y_no_False.pdf plot/paper/fig8a.pdf

# Fig. 8b
cp plot/lumi/boxplot.pdf plot/paper/fig8b.pdf

# Fig. 9a
cp plot/leonardo/heatmaps/allreduce/tasks_per_node_1_metric_mean_base_all_y_no_False.pdf plot/paper/fig9a.pdf

# Fig. 9b
cp plot/leonardo/boxplot.pdf plot/paper/fig9b.pdf

# Fig. 10a
cp plot/mare_nostrum/boxplot.pdf plot/paper/fig10a.pdf

# Fig. 10b
cp plot/fugaku/boxplot.pdf plot/paper/fig10b.pdf

# Table 3
cp plot/lumi/bine_vs_binomial.txt plot/paper/table3.txt

# Table 4
cp plot/leonardo/bine_vs_binomial.txt plot/paper/table4.txt

# Table 5
cp plot/mare_nostrum/bine_vs_binomial.txt plot/paper/table5.txt
