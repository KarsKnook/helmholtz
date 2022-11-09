#!/bin/bash

problem="constant"
mesh_refinement="k^(3/2)"
np=1
k_arr=( 1 2 4)
delta_arr=( 1)

echo "GMRES iterations for $problem with $mesh_refinement for k in [${k_arr[@]}] and delta in [${delta_arr[@]}]"

echo rm total_iterations.dat 
#rm total_iterations.dat
echo touch total_iterations.dat
#touch total_iterations.dat

for k in ${k_arr[@]}
do 
for delta in ${delta_arr[@]}
do
dirname=plots/${k}_${delta}
echo mkdir $dirname
#mkdir $dirname

args="--problem $problem --mesh_refinement $mesh_refinement --k $k --delta $delta --show_args --plot"
echo "mpiexec -np $np python testing.py $args | tee $dirname/resid.dat"
#mpiexec -np $np python testing.py $args | tee $dirname/resid.dat

echo "cat $dirname/iterations.dat >> total_iterations.dat"
#cat $dirname/iterations.dat >> total_iterations.dat
done
done