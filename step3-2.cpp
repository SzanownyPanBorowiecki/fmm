#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <type_traits>
#include <immintrin.h>

#define DEFAULT_BLOCK_SIZE 128

// avx2 intrinsics
inline __m256   load(const float *addr){return _mm256_load_ps(addr); }
inline void     store(float *addr, const __m256 val){ _mm256_store_ps(addr, val); }
inline __m256   add(const __m256 a, const __m256 b){ return _mm256_add_ps(a, b); }
inline __m256   sub(const __m256 a, const __m256 b){ return _mm256_sub_ps(a, b); }
inline __m256   mul(const __m256 a, const __m256 b){ return _mm256_mul_ps(a, b); }
inline __m256   div(const __m256 a, const __m256 b){ return _mm256_div_ps(a, b); }
inline __m256   sqrt(const __m256 a){ return _mm256_sqrt_ps(a); }
inline void     setzero(__m256 &a){ a = _mm256_setzero_ps(); }
inline void     setconst(__m256 &a, const float c){ a = _mm256_set1_ps(c); }
inline void     rotate_halves(__m256 &a){ a = _mm256_permute_ps(a, 0b00111001); }
inline void     swap_halves(__m256 &a){ a = _mm256_permute2f128_ps(a, a, 1); }

inline __m256d  load(const double *addr){return _mm256_load_pd(addr); }
inline void     store(double *addr, const __m256d val){ _mm256_store_pd(addr, val); }
inline __m256d  add(const __m256d a, const __m256d b){ return _mm256_add_pd(a, b); }
inline __m256d  sub(const __m256d a, const __m256d b){ return _mm256_sub_pd(a, b); }
inline __m256d  mul(const __m256d a, const __m256d b){ return _mm256_mul_pd(a, b); }
inline __m256d  div(const __m256d a, const __m256d b){ return _mm256_div_pd(a, b); }
inline __m256d  sqrt(const __m256d a){ return _mm256_sqrt_pd(a); }
inline void     setzero(__m256d &a){ a = _mm256_setzero_pd(); }
inline void     setconst(__m256d &a, const double c){ a = _mm256_set1_pd(c); }
inline void     rotate_halves(__m256d &a){ a = _mm256_permute_pd(a, 0b0101); }
inline void     swap_halves(__m256d &a){ a = _mm256_permute2f128_pd(a, a, 1); }

// non-simd cases
template<typename T> inline T add(const T a, const T b){ return a+b; }
template<typename T> inline T sub(const T a, const T b){ return a-b; }
template<typename T> inline T mul(const T a, const T b){ return a*b; }
template<typename T> inline T div(const T a, const T b){ return a/b; }
template<typename T> inline T sqrt(const T a){ std::sqrt(a); }
template<typename T> inline void setzero(T &a){ a = (T)0; }
template<typename T> inline void setconst(T &a, T c){ a = c; }
template<typename T> inline void rotate_halves(T &a){};
template<typename T> inline void swap_halves(T &a){};


template<typename T, typename V>
class NBodySimulation {

    T t;
    T tFinal;
    T tPlot;
    T tPlotDelta;

    T timeStepSize;
    V vtimeStepSize;

    T maxV;
    T minDx;

    int NumberOfBodies;
    int BlockSize = DEFAULT_BLOCK_SIZE;
    int VectorSize = 1;
    //T CollisionConstant;
    
    V *xx, *xy, *xz;
    V *vx, *vy, *vz;
    V *fx, *fy, *fz;
    V *mass;

    std::ofstream videoFile;
    int snapshotCounter;
    int timeStepCounter;


private:
    inline void clear_forces()
    {
        for (int i = 0; i < NumberOfBodies; i++)
        {
            setzero(fx[i]);
            setzero(fy[i]);
            setzero(fz[i]);
        }

    }

    inline void update_range(int i0, int i1)
    {
        //#pragma omp simd
        
        for (int i = i0; i < i1; i++)
        {
            xx[i] = add(xx[i], mul(vtimeStepSize, vx[i]));
            xy[i] = add(xy[i], mul(vtimeStepSize, vy[i]));
            xz[i] = add(xz[i], mul(vtimeStepSize, vz[i]));

            vx[i] = add(vx[i], mul(vtimeStepSize, div(fx[i], mass[i])));
            vy[i] = add(vy[i], mul(vtimeStepSize, div(fy[i], mass[i])));
            vz[i] = add(vz[i], mul(vtimeStepSize, div(fz[i], mass[i])));

            setzero(fx[i]);
            setzero(fy[i]);
            setzero(fz[i]);
        }
    }

    inline void process_data_distinct(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &fxi, V &fyi, V &fzi, V &fxj, V &fyj, V &fzj)
    {
        for (int k = 0; k < 2; k++)
        {
            // sizeof(V)/sizeof(T) = vector size is known at compile time so this shuld still get unrolled
            for (int l = 0; l < (sizeof(V)/sizeof(T))/2; l++)
            {
                V dx = sub(xxj, xxi);
                V dy = sub(xyj, xyi);
                V dz = sub(xzj, xzi);
                V distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
                V distance = sqrt(distance2);
                V c = div(mul(massi, massj), mul(distance2, distance));
                V dfx = mul(c, dx);
                V dfy = mul(c, dy);
                V dfz = mul(c, dz);
                
                fxi = add(fxi, dfx);
                fyi = add(fyi, dfy);
                fzi = add(fzi, dfz);
                
                fxj =  sub(fxj, dfx);
                fyj =  sub(fyj, dfy);
                fzj =  sub(fzj, dfz);

                //if (std::is_same<V, __m256>::value || std::is_same<V, __m256d>::value){
                    rotate_halves(xxj);
                    rotate_halves(xyj);
                    rotate_halves(xzj);
                    rotate_halves(massj);
                    rotate_halves(fxj);
                    rotate_halves(fyj);
                    rotate_halves(fzj);
                //}
            }
            //if (std::is_same<V, __m256>::value || std::is_same<V, __m256d>::value){
                swap_halves(xxj);
                swap_halves(xyj);
                swap_halves(xzj);
                swap_halves(massj);
                swap_halves(fxj);
                swap_halves(fyj);
                swap_halves(fzj);
            //}
        }
    }


    inline void process_blocks_pair(int blocki, int blockj) 
    {
        for (int i = blocki*BlockSize; i <= (blocki+1)*BlockSize - VectorSize; i++)
        {
            for (int j = blockj*BlockSize; j <= (blockj+1)*BlockSize - VectorSize ; j++)
            {
                //process_disjoint_data_pair(xx[i], xy[i], xz[i], mass[i], xx[j], xy[j], xz[j], mass[j], fx[i], fy[i], fz[i]);
                process_data_distinct(xx[i], xy[i], xz[i], mass[i], xx[j], xy[j], xz[j], mass[j], fx[i], fy[i], fz[i], fx[j], fy[j], fz[j]);
            }
        }
    }
    //#pragma omp declare simd

    inline void process_block_single(int blocki)
    {
        for (size_t i = blocki*BlockSize; i < blocki*BlockSize + BlockSize - 1; i++)
        {
            // process_data_self(xx[i], xy[i], xz[i], mass[i], fx[i], fy[i], fz[i]);
            for (size_t j = i+1; j < blocki*BlockSize + BlockSize; j++)
            {
                process_data_distinct(xx[i], xy[i], xz[i], mass[i], xx[j], xy[j], xz[j], mass[j], fx[i], fy[i], fz[i], fx[j], fy[j], fz[j]);
            }
        }
    }
    

public:
    NBodySimulation () :
        t(0), tFinal(0), tPlot(0), tPlotDelta(0), NumberOfBodies(0),
        mass(nullptr),
        timeStepSize(0), maxV(0), minDx(0), videoFile(nullptr),
        snapshotCounter(0), timeStepCounter(0) {};

    ~NBodySimulation () {
        free(xx); free(xy); free(xz);
        free(vx); free(vy); free(vz);
        free(fx); free(fy); free(fz);
        free(mass);
    }

    void loadData(int argc, char** argv, V* xx, V* xy, V* xz, V* vx, V* vy, V* vz, V* mass)
    {

    }

    void setUp(int argc, char** argv, int block_size) {
        NumberOfBodies = (argc-4) / 7;
        BlockSize = block_size;
        VectorSize = sizeof(V)/sizeof(T);
        //ectorSize = sizeof(V)/sizeof(T);
        
        size_t array_size = (NumberOfBodies + VectorSize - 1)  / VectorSize;
        std::cout << "array_size = " << array_size << std::endl;

        size_t alloc_size = array_size*sizeof(V);  // round up to the nearest multiple of 32
        std::cout << "alloc size = " << alloc_size << std::endl;
        xx = (V*)aligned_alloc(32, alloc_size);
        xy = (V*)aligned_alloc(32, alloc_size);
        xz = (V*)aligned_alloc(32, alloc_size);
        vx = (V*)aligned_alloc(32, alloc_size);
        vy = (V*)aligned_alloc(32, alloc_size);
        vz = (V*)aligned_alloc(32, alloc_size);
        fx = (V*)aligned_alloc(32, alloc_size);
        fy = (V*)aligned_alloc(32, alloc_size);
        fz = (V*)aligned_alloc(32, alloc_size);
        mass = (V*)aligned_alloc(32, alloc_size);
        
        int readArgument = 1;
        tPlotDelta   = std::stof(argv[readArgument]); readArgument++;
        tFinal       = std::stof(argv[readArgument]); readArgument++;
        timeStepSize = std::stof(argv[readArgument]); readArgument++;
        setconst(vtimeStepSize, timeStepSize);

        int i=0;
        for (; i < NumberOfBodies; i++)
        {
            xx[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            xy[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            xz[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            vx[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            vy[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            vz[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            mass[i/VectorSize][i%VectorSize] = std::stof(argv[readArgument]); readArgument++;
            if (mass[i/VectorSize][i%VectorSize]<=0.0 ) {
                std::cerr << "invalid mass for body " << i << std::endl;
                exit(-2);
            }
        }
        // pad last vector with zeros
        for (; i % VectorSize != 0; i++)
        {
            xx[i/VectorSize][i%VectorSize] = 0.0;
            xy[i/VectorSize][i%VectorSize] = 0.0;
            xz[i/VectorSize][i%VectorSize] = 0.0;
            vx[i/VectorSize][i%VectorSize] = 0.0;
            vy[i/VectorSize][i%VectorSize] = 0.0;
            vz[i/VectorSize][i%VectorSize] = 0.0;
            mass[i/VectorSize][i%VectorSize] = 0.0;
        }
        /*
        else
        {
            for (int i = 0; i < NumberOfBodies; i++)
            {
                xx[i] = std::stof(argv[readArgument]); readArgument++;
                xy[i] = std::stof(argv[readArgument]); readArgument++;
                xz[i] = std::stof(argv[readArgument]); readArgument++;

                vx[i] = std::stof(argv[readArgument]); readArgument++;
                vy[i] = std::stof(argv[readArgument]); readArgument++;
                vz[i] = std::stof(argv[readArgument]); readArgument++;

                mass[i] = std::stof(argv[readArgument]); readArgument++;
                if (mass[i]<=0.0 ) {
                    std::cerr << "invalid mass for body " << i << std::endl;
                    exit(-2);
                }
            }
        }
        */

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

    inline void updateBody() {
        timeStepCounter++;
        minDx  = std::numeric_limits<T>::max();

        int NumberOfBlocks = (NumberOfBodies + BlockSize - 1)/BlockSize;
        clear_forces();
        
        // Round-robin tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)
        for (int i = 0; i < NumberOfBlocks-1; i++)
        {
            // these threads will never access the same memory at once so can be run in parallel
            for (int j = 0; j < NumberOfBlocks/2; j++)
            {
                int a = (j == 0) ? 0 : ((i+j)%(NumberOfBlocks-1))+1;
                int b = ((i+NumberOfBlocks-j-1)%(NumberOfBlocks-1))+1;
                process_blocks_pair(a, b);
            }
        }

        // can run in parallel
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            int i0 = i*BlockSize;
            int i1 = std::min(i0+BlockSize, NumberOfBodies);
            process_block_single(i);
            update_range(i0, i1);
        }
        t += timeStepSize;
    }

    
    /**
     * ---------- DO NOT TOUCH ----------
     */
    bool hasReachedEnd() {
        return t > tFinal;
    }
    void openParaviewVideoFile() {
        videoFile.open("paraview-output/result.pvd");
        videoFile << "<?xml version=\"1.0\"?>" << std::endl
                << "<VTKFile type=\"Collection\""
                    " version=\"0.1\""
                    " byte_order=\"LittleEndian\""
                    " compressor=\"vtkZLibDataCompressor\">" << std::endl
                << "<Collection>";
    }
    void closeParaviewVideoFile() {
        videoFile << "</Collection>"
                << "</VTKFile>" << std::endl;
        videoFile.close();
    }
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
            out << xx[i/VectorSize][i % VectorSize]
                << " "
                << xy[i/VectorSize][i % VectorSize]
                << " "
                << xz[i/VectorSize][i % VectorSize]
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
    void printSnapshotSummary() {
        std::cout << "plot next snapshot"
                << ",\t time step=" << timeStepCounter
                << ",\t t="         << t
                << ",\t dt="        << timeStepSize
                << ",\t v_max="     << maxV
                << ",\t dx_min="    << minDx
                << std::endl;
    }
    void takeSnapshot() {
        if (t >= tPlot) {
        printParaviewSnapshot();
        printSnapshotSummary();
        tPlot += tPlotDelta;
        }
    }
    void printSummary() {
        std::cout << "Number of remaining objects: " << NumberOfBodies << std::endl;
        std::cout << "Position of first remaining object: "
                << xx[0][0] << ", " << xy[0][0] << ", " << xz[0][0] << std::endl;
    }
};


template <typename T1, typename T2>
void run(NBodySimulation<T1, T2> *nbs, int argc, char **argv, int block_size)
{
    // Code that initialises and runs the simulation.
    nbs->setUp(argc,argv,block_size);
    nbs->openParaviewVideoFile();
    nbs->takeSnapshot();
    
    //nbs->test();

    
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
                << "  --sp            Single precision computation (instead of double) " << std::endl
                << "  --avx2          Compute using AVX2 SIMD instructions" << std::endl
                << " ----------------------------------" << std::endl
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
        if (avx2){
            NBodySimulation<float, __m256> nbs;
            run(&nbs,argc,argv,block_size);
        }
        else{
            //NBodySimulation<float, float> nbs;
            //run(&nbs,argc,argv,block_size);
        }


    }
    else{
        if (avx2){
            //NBodySimulation<double, __m256d> nbs;
            //run(&nbs,argc,argv,block_size);
        } else{
            //NBodySimulation<double, double> nbs;
            //run(&nbs,argc,argv,block_size);
        }
    }
    return 0;
}
