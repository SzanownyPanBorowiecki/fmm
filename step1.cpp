#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// You can compile this file with
// g++ -O3 assignment-code.cpp -o assignment-code
// or with the Makefile  and run it with
// ./assignment-code

// Results will be added to the paraview-output directory. In it you will find
// a result.pvd file that you can open with ParaView. To see the points you will
// need to look a the properties of result.pvd and select the representation
// "Point Gaussian". Pressing play will play your time steps.
// #define CUTOFF 10

template<typename T>
class NBodySimulation {

  T t;
  T tFinal;
  T tPlot;
  T tPlotDelta;

  int NumberOfBodies;
  int InitialNumberOfBodies;

  /**
   * Pointer to pointers. Each pointer in turn points to three coordinates, i.e.
   * each pointer represents one molecule/particle/body.
   */
  T** x;

  /**
   * Equivalent to x storing the velocities.
   */
  T** v;

  /**
   * One mass entry per molecule/particle.
   */
  T*  mass;

  /**
   * Global time step size used.
   */
  T timeStepSize;

  /**
   * Maximum velocity of all particles.
   */
  T maxV;

  /**
   * Minimum distance between two elements.
   */
  T minDx;
  T minCollisionParam;
  T collisionConstant;


  /**
   * Stream for video output file.
   */
  std::ofstream videoFile;

  /**
   * Output counters.
   */
  int snapshotCounter;
  int timeStepCounter;


public:
  /**
   * Constructor.
   */
  NBodySimulation () :
    t(0), tFinal(0), tPlot(0), tPlotDelta(0), NumberOfBodies(0),
    x(nullptr), v(nullptr), mass(nullptr),
    timeStepSize(0), maxV(0), minDx(0), videoFile(nullptr),
    snapshotCounter(0), timeStepCounter(0) {};

  /**
   * Destructor.
   */
  ~NBodySimulation () {
    if (x != nullptr) {
      for (int i=0; i<NumberOfBodies; i++)
        delete [] x[i];
      delete [] x;
    }
    if (v != nullptr) {
      for (int i=0; i<NumberOfBodies; i++)
        delete [] v[i];
      delete [] v;
    }
    if (mass != nullptr) {
      delete [] mass;
    }
  }

  /**
   * Set up scenario from the command line.
   *
   * If you need additional helper data structures, you can initialise them
   * here. Alternatively, you can introduce a totally new function to initialise
   * additional data fields and call this new function from main after setUp().
   * Either way is fine.
   *
   * This operation's semantics is not to be changed in the assignment.
   */
  void setUp(int argc, char** argv) {
    NumberOfBodies = (argc-4) / 7;
    collisionConstant = 1.0/(100*NumberOfBodies);

    InitialNumberOfBodies = NumberOfBodies;
    x    = new T*[NumberOfBodies];
    v    = new T*[NumberOfBodies];
    mass = new T [NumberOfBodies];

    int readArgument = 1;

    tPlotDelta   = std::stof(argv[readArgument]); readArgument++;
    tFinal       = std::stof(argv[readArgument]); readArgument++;
    timeStepSize = std::stof(argv[readArgument]); readArgument++;

    for (int i=0; i<NumberOfBodies; i++) {
      x[i] = new T[3];
      v[i] = new T[3];

      x[i][0] = std::stof(argv[readArgument]); readArgument++;
      x[i][1] = std::stof(argv[readArgument]); readArgument++;
      x[i][2] = std::stof(argv[readArgument]); readArgument++;

      v[i][0] = std::stof(argv[readArgument]); readArgument++;
      v[i][1] = std::stof(argv[readArgument]); readArgument++;
      v[i][2] = std::stof(argv[readArgument]); readArgument++;

      mass[i] = std::stof(argv[readArgument]); readArgument++;

      if (mass[i]<=0.0 ) {
        std::cerr << "invalid mass for body " << i << std::endl;
        exit(-2);
      }
    }

    std::cout << "created setup with " << NumberOfBodies << " bodies"
              << std::endl;

    if (tPlotDelta<=0.0) {
      std::cout << "plotting switched off" << std::endl;
      tPlot = tFinal + 1.0;
    }
    else {
      std::cout << "plot initial setup plus every " << tPlotDelta
                << " time units" << std::endl;
      tPlot = 0.0;
    }
  }

  void force_calculation(int i, int j, T &fx, T &fy, T &fz, T &distance){
   T dx = x[j][0]-x[i][0];
   T dy = x[j][1]-x[i][1];
   T dz = x[j][2]-x[i][2];
   T distance2 = dx*dx + dy*dy + dz*dz;
   distance = std::sqrt(distance2);
   T c = mass[i]*mass[j]/(distance2*distance);
   fx = c*dx;
   fy = c*dy;
   fz = c*dz;
  }

  /**
   * This is where the timestepping scheme and force updates are implemented
   */
  void updateBody() {

    timeStepCounter++;
    maxV   = 0.0;
    minDx  = std::numeric_limits<T>::max();
    // force0 = force along x direction
    // force1 = force along y direction
    // force2 = force along z direction 
    T* force0 = new T[NumberOfBodies];
    T* force1 = new T[NumberOfBodies];
    T* force2 = new T[NumberOfBodies];
    for (int k = 0; k<NumberOfBodies; ++k)
    {
      force0[k] = 0.0;
      force1[k] = 0.0;
      force2[k] = 0.0;
    }

    for (int k = 0; k<NumberOfBodies; ++k)
    {
      for (int l=k+1; l<NumberOfBodies; ++l)
      {
        // x,y,z forces acting on particle k due to particle l
        T fx, fy, fz, distance;
        force_calculation(k,l,fx,fy,fz,distance);
        minDx = std::min( minDx,distance );

        if (distance/(mass[k]+mass[l]) <= collisionConstant)
        {
          x[k][0] = (mass[k]*x[k][0]+mass[l]*x[l][0])/(mass[k]+mass[l]);
          x[k][1] = (mass[k]*x[k][1]+mass[l]*x[l][1])/(mass[k]+mass[l]);
          x[k][2] = (mass[k]*x[k][2]+mass[l]*x[l][2])/(mass[k]+mass[l]);
          v[k][0] = (mass[k]*v[k][0]+mass[l]*v[l][0])/(mass[k]+mass[l]);
          v[k][1] = (mass[k]*v[k][1]+mass[l]*v[l][1])/(mass[k]+mass[l]);
          v[k][2] = (mass[k]*v[k][2]+mass[l]*v[l][2])/(mass[k]+mass[l]);
          mass[k] += mass[l];

          // remove body l by replacing it with the last one 
          x[l] = x[NumberOfBodies-1];
          v[l] = v[NumberOfBodies-1];
          mass[l] = mass[NumberOfBodies-1];
          
          delete [] x[NumberOfBodies-1];
          delete [] v[NumberOfBodies-1];
          mass[NumberOfBodies-1] = 0.0;
          NumberOfBodies--;

          // will have to calculate this frame again
          return;
        }

        force0[k] += fx;
        force1[k] += fy;
        force2[k] += fz;

        force0[l] -= fx;
        force1[l] -= fy;
        force2[l] -= fz;
      }
    }


    for (int k = 0; k<NumberOfBodies; ++k)
    {
      x[k][0] = x[k][0] + timeStepSize * v[k][0];
      x[k][1] = x[k][1] + timeStepSize * v[k][1];
      x[k][2] = x[k][2] + timeStepSize * v[k][2];
      v[k][0] = v[k][0] + timeStepSize * force0[k] / mass[k];
      v[k][1] = v[k][1] + timeStepSize * force1[k] / mass[k];
      v[k][2] = v[k][2] + timeStepSize * force2[k] / mass[k];
      maxV = std::max(maxV, std::sqrt( v[k][0]*v[k][0] + v[k][1]*v[k][1] + v[k][2]*v[k][2] ));

    }

    t += timeStepSize;

    delete[] force0;
    delete[] force1;
    delete[] force2;      
  }

  /**
   * Check if simulation has been completed.
   */
  bool hasReachedEnd() {
    return t > tFinal;
  }

  /**
   * This operation is not to be changed in the assignment.
   */
  void openParaviewVideoFile() {
    videoFile.open("paraview-output/result.pvd");
    videoFile << "<?xml version=\"1.0\"?>" << std::endl
              << "<VTKFile type=\"Collection\""
                 " version=\"0.1\""
                 " byte_order=\"LittleEndian\""
                 " compressor=\"vtkZLibDataCompressor\">" << std::endl
              << "<Collection>";
  }

  /**
   * This operation is not to be changed in the assignment.
   */
  void closeParaviewVideoFile() {
    videoFile << "</Collection>"
              << "</VTKFile>" << std::endl;
    videoFile.close();
  }

  /**
   * The file format is documented at
   * http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
   *
   * This operation is not to be changed in the assignment.
   */
  void printParaviewSnapshot() {
    static int counter = -1;
    counter++;
    std::stringstream filename, filename_nofolder;
    filename << "paraview-output/result-" << counter <<  ".vtp";
    filename_nofolder << "result-" << counter <<  ".vtp";
    std::ofstream out( filename.str().c_str() );
    out << "<VTKFile type=\"PolyData\" >" << std::endl
        << "<PolyData>" << std::endl
        << " <Piece NumberOfPoints=\"" << NumberOfBodies << "\">" << std::endl
        << "  <Points>" << std::endl
        << "   <DataArray type=\"Float64\""
                  " NumberOfComponents=\"3\""
                  " format=\"ascii\">";

    for (int i=0; i<NumberOfBodies; i++) {
      out << x[i][0]
          << " "
          << x[i][1]
          << " "
          << x[i][2]
          << " ";
    }

    out << "   </DataArray>" << std::endl
        << "  </Points>" << std::endl
        << " </Piece>" << std::endl
        << "</PolyData>" << std::endl
        << "</VTKFile>"  << std::endl;

    out.close();

    videoFile << "<DataSet timestep=\"" << counter
              << "\" group=\"\" part=\"0\" file=\"" << filename_nofolder.str()
              << "\"/>" << std::endl;
  }

  /**
   * This operations is not to be changed in the assignment.
   */
  void printSnapshotSummary() {
    
    std::cout << "plot next snapshot"
              << ",\t time step=" << timeStepCounter
              << ",\t t="         << t
              << ",\t dt="        << timeStepSize
              << ",\t v_max="     << maxV
              << ",\t dx_min="    << minDx
              << std::endl;
    
  }

  /**
   * This operations is not to be changed in the assignment.
   */
  void takeSnapshot() {
    if (t >= tPlot) {
      printParaviewSnapshot();
      printSnapshotSummary();
      tPlot += tPlotDelta;
    }
  }

  /**
   * This operations is not to be changed in the assignment.
   */
  void printSummary() {
    std::cout << "Number of remaining objects: " << NumberOfBodies << std::endl;
    std::cout << "Position of first remaining object: "
              << x[0][0] << ", " << x[0][1] << ", " << x[0][2] << std::endl;
  }
};


template <typename T>
void run(T *nbs, int argc, char **argv)
{
    // Code that initialises and runs the simulation.
    nbs->setUp(argc,argv);
    nbs->openParaviewVideoFile();
    nbs->takeSnapshot();
    while (!nbs->hasReachedEnd()) {
        nbs->updateBody();
        nbs->takeSnapshot();
    }
    nbs->printSummary();
    nbs->closeParaviewVideoFile();
}

/**
 * Main routine.
 *
 * No major changes in assignment. You can add a few initialisation
 * or stuff if you feel the need to do so. But keep in mind that you
 * may not alter what the program plots to the terminal.
 */
int main(int argc, char** argv) {
    bool sp = false;

    int k = 0;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--sp")
        {
            sp = true;
            k++;
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
        NBodySimulation<float> nbs;
        run(&nbs,argc,argv);
    }
    else{
        NBodySimulation<double> nbs;
        run(&nbs,argc,argv);
    }
    return 0;
}


