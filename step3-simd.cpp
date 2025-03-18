#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#define DEFAULT_BLOCK_SIZE 128
#include "NBodySimulationSIMD.hpp"
#include "NBodySimulation.hpp"

template <typename T>
void run(T *nbs, int argc, char **argv, int block_size)
{
    // Code that initialises and runs the simulation.
    nbs->setUp(argc,argv,block_size);
    nbs->openParaviewVideoFile();
    nbs->takeSnapshot();
    while (!nbs->hasReachedEnd()) {
        nbs->updateBody();
        nbs->takeSnapshot();
    }
    nbs->printSummary();
    nbs->closeParaviewVideoFile();
}

int main(int argc, char** argv) {



    bool sp = false;
    bool avx2 = false;
    int block_size = DEFAULT_BLOCK_SIZE;

    int k = 0;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--sp")
        {
            sp = true;
            k++;
        }
        if (arg == "--avx2")
        {
            avx2 = true;
            k++;
        }
        if (arg == "--bs")
        {
            block_size = std::stoi(argv[i+1]);
            //i++;
            k+=2;
        }
    }
    argc -= k;
    argv += k;

    if (argc==1) {
        std::cerr << "usage: " << std::string(argv[0])
                << " [options] plot-time final-time dt objects" << std::endl
                << " Details:" << std::endl
                << " ----------------------------------" << std::endl
                << " [options] Optional switches:" << std::endl
                << "  --sp            Single precision computation (instead of double) " << std::endl;

            #ifdef __AVX2__
                std::cerr << "  --avx2          Compute using explicit AVX2 implementation" << std::endl;
            #endif
                
        std::cerr << " ----------------------------------" << std::endl
                << "  plot-time:        interval after how many time units to plot."
                    " Use 0 to switch off plotting" << std::endl
                << "  final-time:      simulated time (greater 0)" << std::endl
                << "  dt:              time step size (greater 0)" << std::endl
                << "  objects:         any number of bodies, specified by position, velocity, mass" << std::endl
                << std::endl
                << "Examples of arguments:" << std::endl
                << "+ One body moving form the coordinate system's centre along x axis with speed 1" << std::endl
                << "    0.01  100.0  0.001    0.0 0.0 0.0  1.0 0.0 0.0  1.0" << std::endl
                << "+ One body spiralling around the other" << std::endl
                << "    0.01  100.0  0.001    0.0 0.0 0.0  1.0 0.0 0.0  1.0     0.0 1.0 0.0  1.0 0.0 0.0  1.0" << std::endl
                << "+ Three-body setup from first lecture" << std::endl
                << "    0.01  100.0  0.001    3.0 0.0 0.0  0.0 1.0 0.0  0.4     0.0 0.0 0.0  0.0 0.0 0.0  0.2     2.0 0.0 0.0  0.0 0.0 0.0  1.0" << std::endl
                << "+ Five-body setup" << std::endl
                << "    0.01  100.0  0.001    3.0 0.0 0.0  0.0 1.0 0.0  0.4     0.0 0.0 0.0  0.0 0.0 0.0  0.2     2.0 0.0 0.0  0.0 0.0 0.0  1.0     2.0 1.0 0.0  0.0 0.0 0.0  1.0     2.0 0.0 1.0  0.0 0.0 0.0  1.0" << std::endl
                << std::endl;

        return -1;
    }

    if ( (argc-4)%7!=0 ) {
        std::cerr << "error in arguments: each planet is given by seven entries"
                    " (position, velocity, mass)" << std::endl;
        std::cerr << "got " << argc << " arguments"
                    " (three of them are reserved)" << std::endl;
        std::cerr << "run without arguments for usage instruction" << std::endl;
        return -2;
    }

    std::cout << std::setprecision(15);

    
    if (sp){
        #ifdef __AVX2__
            if (avx2){
                NBodySimulationSIMD<float, __m256> nbs;
                run(&nbs,argc,argv,block_size);
            }
            else{
                NBodySimulation<float> nbs;
                run(&nbs,argc,argv,block_size);
            }
        #else
            NBodySimulation<float> nbs;
            run(&nbs,argc,argv,block_size);
        #endif 

    }
    else{
        #ifdef __AVX2__
            if (avx2){
                NBodySimulationSIMD<double, __m256d> nbs;
                run(&nbs,argc,argv,block_size);
            }
            else{
                NBodySimulation<double> nbs;
                run(&nbs,argc,argv,block_size);
            }
        #else
            NBodySimulation<double> nbs;
            run(&nbs,argc,argv,block_size);
        #endif 
    }
    
    return 0;
}
