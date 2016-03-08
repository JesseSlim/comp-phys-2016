# Molecular Dynamics project code
This folder contains the code for the molecular dynamics argon simulation project. It also contains the raw data from our simulations in the various results*/-folders and some scripts used for data analysis in the analysis/ folder. The final simulation results are referred to as 'longrun' in all filenames, as they were run for a longer time than the initial simulations.

## Simulation code
The development of the simulation code was of course an iterative process and remnants of that process are strewn all over this folder. The files were developed in the following order:

TestSimulation.ipynb -> ArgonSimulation.ipynb -> ArgonSimulation.py (final code that was actually used)
The final simulations were performed batchwise, controlled by the run_simulation_long.py script

Other interesting files include:
- CalculateDistancesTiming.ipynb: used for benchmarking different methods of calculating simulation quantities. The fastest method was eventually implemented in the simulation code
- PlotResults.ipynb: preliminary display of simulation results for verification purposes
- transcribe_results.py: removes the distance histogram from the final simulation result files (which was by far the largest array saved and not necessary as the correlation function calculated from it was also saved) and counteracted some initial bugs in the measurement code.

Some code was actually written twice as a matter of internal competition: these files may also be found in this folder

## Data analysis
The final result quantities were extracted from the 'longrun' simulation results. Code for calculating these and plotting may be found in analysis/LongRun*.ipynb

WARNING: Note that pulling this entire repository might be a bad idea, because we also used it to share raw simulation data (>15GB)