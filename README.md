# TransportEquationTVDStabilisation
Simulation of the reaction-diffusion-advection equation with the TVD stabilisation of the convective term.
The current setting is the "solid body rotation" benchmark:  simulation of the pure transport equation on the unit square [0,1]^2, where the initial condition is transported in the counter-clockwise direction. 
The code is based on the dealii finite element library. 
To let it run, one has to install dealii and then to include the *.cc source-files into the project (see dealii/examples/, e.g. the example 12 of the heat equation).
