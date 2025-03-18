#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <immintrin.h>

#define DEFAULT_BLOCK_SIZE 128

// avx2 intrinsics
//template<typename T> inline T load(const T *addr){return *addr; }
inline __m256 load(const float *addr){return _mm256_load_ps(addr); }
inline float load(const float *addr){return _mm256_load_ps(addr); }

inline __m256d load(const double *addr){return _mm256_load_pd(addr); }
//inline float load(const float *addr){return *addr; }

inline void store(float *addr, const __m256 val){ _mm256_store_ps(addr, val); }
inline void store(double *addr, const __m256d val){ _mm256_store_pd(addr, val); }
template<typename T> inline void store(T *addr, T val){ *addr = val; }

inline __m256 add(const __m256 a, const __m256 b){ return _mm256_add_ps(a, b); }
inline __m256d add(const __m256d a, const __m256d b){ return _mm256_add_pd(a, b); }
inline __m256 sub(const __m256 a, const __m256 b){ return _mm256_sub_ps(a, b); }
inline __m256d sub(const __m256d a, const __m256d b){ return _mm256_sub_pd(a, b); }
inline __m256 mul(const __m256 a, const __m256 b){ return _mm256_mul_ps(a, b); }
inline __m256d mul(const __m256d a, const __m256d b){ return _mm256_mul_pd(a, b); }
inline __m256 div(const __m256 a, const __m256 b){ return _mm256_div_ps(a, b); }
inline __m256d div(const __m256d a, const __m256d b){ return _mm256_div_pd(a, b); }
inline __m256 sqrt(const __m256 a){ return _mm256_sqrt_ps(a); }
inline __m256d sqrt(const __m256d a){ return _mm256_sqrt_pd(a); }
inline __m256 rotate_halves(const __m256 a){ return _mm256_permute_ps(a, 0b00111001); }
inline __m256d rotate_halves(const __m256d a){ return _mm256_permute_pd(a, 0b0101); }
inline __m256 swap_halves(const __m256 a){ return _mm256_permute2f128_ps(a, a, 1); }
inline __m256d swap_halves(const __m256d a){ return _mm256_permute2f128_pd(a, a, 1); }

// You can compile this file with
// g++ -O3 assignment-code.cpp -o assignment-code
// or with the Makefile  and run it with
// ./assignment-code

// Results will be added to the paraview-output directory. In it you will find
// a result.pvd file that you can open with ParaView. To see the points you will
// need to look a the properties of result.pvd and select the representation
// "Point Gaussian". Pressing play will play your time steps.

template<typename T, typename V>
class NBodySimulation {

    T t;
    T tFinal;
    T tPlot;
    T tPlotDelta;

    T timeStepSize;
    T maxV;
    T minDx;

    int NumberOfBodies;
    int BlockSize = DEFAULT_BLOCK_SIZE;
    int VectorSize = 1;
    //T CollisionConstant;
    
    T *xx, *xy, *xz;
    T *vx, *vy, *vz;
    T *fx, *fy, *fz;
    T *mass;

    std::ofstream videoFile;
    int snapshotCounter;
    int timeStepCounter;


private:
    inline void clear_forces()
    {
        for (int i = 0; i < NumberOfBodies; i++)
        {
            fx[i] = 0.0;
            fy[i] = 0.0;
            fz[i] = 0.0;
        }

    }

    inline void update_range(int i0, int i1)
    {
        //#pragma omp simd
        for (int i = i0; i < i1; i++)
        {
            xx[i] += timeStepSize*vx[i];
            xy[i] += timeStepSize*vy[i];
            xz[i] += timeStepSize*vz[i];
            //std::cout << "xx[" << i << "]=" << xx[i] << std::endl;

            vx[i] += timeStepSize*fx[i]/mass[i];
            vy[i] += timeStepSize*fy[i]/mass[i];
            vz[i] += timeStepSize*fz[i]/mass[i];

            fx[i] = 0.0;
            fy[i] = 0.0;
            fz[i] = 0.0;
        }
    }

    inline void process_data_pair_disjoint(__m256 &xxi, __m256 &xyi, __m256 &xzi, __m256 &massi, __m256 &xxj, __m256 &xyj, __m256 &xzj, __m256 &massj, __m256 &fxi, __m256 &fyi, __m256 &fzi, __m256 &fxj, __m256 &fyj, __m256 &fzj)
    {
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < 4; l++)
            {
                __m256 dx = sub(xxj, xxi);
                __m256 dy = sub(xyj, xyi);
                __m256 dz = sub(xzj, xzi);
                __m256 distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
                __m256 distance = sqrt(distance2);
                __m256 c = div(mul(massi, massj), mul(distance2, distance));
                __m256 dfx = mul(c, dx);
                __m256 dfy = mul(c, dy);
                __m256 dfz = mul(c, dz);
                
                fxi = add(fxi, dfx);
                fyi = add(fyi, dfy);
                fzi = add(fzi, dfz);
                
                fxj =  sub(fxj, dfx);
                fyj =  sub(fyj, dfy);
                fzj =  sub(fzj, dfz);

                xxj = rotate_halves(xxj);
                xyj = rotate_halves(xyj);
                xzj = rotate_halves(xzj);
                massj = rotate_halves(massj);
                fxj = rotate_halves(fxj);
                fyj = rotate_halves(fyj);
                fzj = rotate_halves(fzj);
            }
            xxj = swap_halves(xxj);
            xyj = swap_halves(xyj);
            xzj = swap_halves(xzj);
            massj = swap_halves(massj);
            fxj = swap_halves(fxj);
            fyj = swap_halves(fyj);
            fzj = swap_halves(fzj);
        }
    }

    inline void process_data_pair_disjoint(__m256d &xxi, __m256d &xyi, __m256d &xzi, __m256d &massi, __m256d &xxj, __m256d &xyj, __m256d &xzj, __m256d &massj, __m256d &fxi, __m256d &fyi, __m256d &fzi, __m256d &fxj, __m256d &fyj, __m256d &fzj)
    {
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < 2; l++)
            {
                __m256d dx = sub(xxj, xxi);
                __m256d dy = sub(xyj, xyi);
                __m256d dz = sub(xzj, xzi);
                __m256d distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
                __m256d distance = sqrt(distance2);
                __m256d c = div(mul(massi, massj), mul(distance2, distance));
                __m256d dfx = mul(c, dx);
                __m256d dfy = mul(c, dy);
                __m256d dfz = mul(c, dz);
                
                fxi = add(fxi, dfx);
                fyi = add(fyi, dfy);
                fzi = add(fzi, dfz);
                
                fxj =  sub(fxj, dfx);
                fyj =  sub(fyj, dfy);
                fzj =  sub(fzj, dfz);

                xxj = rotate_halves(xxj);
                xyj = rotate_halves(xyj);
                xzj = rotate_halves(xzj);
                massj = rotate_halves(massj);
                fxj = rotate_halves(fxj);
                fyj = rotate_halves(fyj);
                fzj = rotate_halves(fzj);
            }
            xxj = swap_halves(xxj);
            xyj = swap_halves(xyj);
            xzj = swap_halves(xzj);
            massj = swap_halves(massj);
            fxj = swap_halves(fxj);
            fyj = swap_halves(fyj);
            fzj = swap_halves(fzj);
        }
    }

    inline void process_data_pair_disjoint(T &xxi, T &xyi, T &xzi, T &massi, T &xxj, T &xyj, T &xzj, T &massj, T &fxi, T &fyi, T &fzi, T &fxj, T &fyj, T &fzj)
    {
        T dx = xxj - xxi;
        T dy = xyj - xyi;
        T dz = xzj - xzi;
        T distance2 = dx*dx + dy*dy + dz*dz;
        T distance = std::sqrt(distance2);
        T c = massi*massj/(distance2*distance);
        T dfx = c*dx;
        T dfy = c*dy;
        T dfz = c*dz;
        fxi += dfx;
        fyi += dfy;
        fzi += dfz;
        fxj -= dfx;
        fyj -= dfy;
        fzj -= dfz;
    }

    
    inline void process_blocks_pair(int blocki, int blockj) 
    {
        for (int i = blocki*BlockSize; i <= (blocki+1)*BlockSize - VectorSize; i += VectorSize)
        {
            V xxi = (V)load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V massi =load(mass + i);
            V fxi = load(fx + i);
            V fyi = load(fy + i);
            V fzi = load(fz + i);

            for (int j = blockj*BlockSize; j <= (blockj+1)*BlockSize - VectorSize ; j += VectorSize)
            {
                V xxj = load(xx + j);
                V xyj = load(xy + j);
                V xzj = load(xz + j);
                V massj = load(mass + j);
                V fxj = load(fx + j);
                V fyj = load(fy + j);
                V fzj = load(fz + j);
                process_data_pair_disjoint(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, fxi, fyi, fzi, fxj, fyj, fzj);
                store(fx+j, fxj);
                store(fy+j, fyj);
                store(fz+j, fzj);
            }
        store(fx+i, fxi);
        store(fy+i, fyi);
        store(fz+i, fzi);
        }
    }
    //#pragma omp declare simd

    inline void process_block_single(int blocki)
    {
        for (size_t i = blocki*BlockSize; i < blocki*BlockSize + BlockSize - 1; i++)
        {
            T xxi = xx[i];
            T xyi = xy[i];
            T xzi = xz[i];
            T massi = mass[i];

            T dfxi = 0.0;
            T dfyi = 0.0;
            T dfzi = 0.0;
            
            //#pragma omp simd
            for (size_t j = i+1; j < blocki*BlockSize + BlockSize; j++)
            {
                T xxj = xx[j];
                T xyj = xy[j];
                T xzj = xz[j];
                T massj = mass[j];

                T dx = xxj - xxi;            
                T dy = xyj - xyi;            
                T dz = xzj - xzi;

                T dst2 = dx*dx + dy*dy + dz*dz;
                T dst = std::sqrt(dst2);
                T c = massi*massj/(dst2*dst);

                T dfx = c*dx;
                T dfy = c*dy;
                T dfz = c*dz;

                dfxi += dfx;
                dfyi += dfy;
                dfzi += dfz;

                fx[j] -= dfx;
                fy[j] -= dfy;
                fz[j] -= dfz;
            }
            fx[i] += dfxi;
            fy[i] += dfyi;
            fz[i] += dfzi;
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

    void setUp(int argc, char** argv, int block_size) {
        NumberOfBodies = (argc-4) / 7;
        BlockSize = block_size;
        VectorSize = sizeof(V)/sizeof(T);

        size_t array_size = NumberOfBodies;
        size_t alloc_size = (array_size*sizeof(T) + 31) & -32;  // round up to the nearest multiple of 32
        xx = (T*)aligned_alloc(32, alloc_size);
        xy = (T*)aligned_alloc(32, alloc_size);
        xz = (T*)aligned_alloc(32, alloc_size);
        vx = (T*)aligned_alloc(32, alloc_size);
        vy = (T*)aligned_alloc(32, alloc_size);
        vz = (T*)aligned_alloc(32, alloc_size);
        fx = (T*)aligned_alloc(32, alloc_size);
        fy = (T*)aligned_alloc(32, alloc_size);
        fz = (T*)aligned_alloc(32, alloc_size);
        mass = (T*)aligned_alloc(32, alloc_size);

        int readArgument = 1;

        tPlotDelta   = std::stof(argv[readArgument]); readArgument++;
        tFinal       = std::stof(argv[readArgument]); readArgument++;
        timeStepSize = std::stof(argv[readArgument]); readArgument++;

        for (int i=0; i<NumberOfBodies; i++) {
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

    void test()
    {
        clear_forces();
        printf("process_pair ---------------\n");
        process_pair(0,1, xx, xy, xz, mass, fx, fy, fz);
        for (int i = 0; i < NumberOfBodies; i++){
            printf("fx[%i]=%f\n", i, fx[i]);
        }

        clear_forces();
        printf("process_pair_avx2 ---------------\n");
        process_pair_avx2(0,1, xx, xy, xz, mass, fx, fy, fz);
        for (int i = 0; i < NumberOfBodies; i++){
            printf("fx[%i]=%f\n", i, fx[i]);
        }

    }

    inline void updateBody() {
        timeStepCounter++;
        minDx  = std::numeric_limits<T>::max();

        int NumberOfBlocks = (NumberOfBodies + BlockSize - 1)/BlockSize;
//        std::cout << "numberofblocks = " << NumberOfBlocks << std::endl;
        //clear_forces();
        for (int i =0 ; i < NumberOfBodies; i++)
        {
            fx[i] = 0.0;
            fy[i] = 0.0;
            fz[i] = 0.0;
        }
        
        // Round-robin tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)
        for (int i = 0; i < NumberOfBlocks-1; i++)
        {
            // these threads will never access the same memory so can be run in parallel
           //#pragma omp parallel for
            for (int j = 0; j < NumberOfBlocks/2; j++)
            {
                int a = (j == 0) ? 0 : ((i+j)%(NumberOfBlocks-1))+1;
                int b = ((i+NumberOfBlocks-j-1)%(NumberOfBlocks-1))+1;
                process_blocks_pair(a, b);
            }
        }

        // can be done in parallel
        //#pragma omp parallel for
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            int i0 = i*BlockSize;
            int i1 = std::min(i0+BlockSize, NumberOfBodies);
            process_block_single(i);
            update_range(i0, i1);
        }

        // we have all forces now

        /*
        #ifdef __AVX2__
            mainLoop_AVX(xx, xy, xz, vx, vy, vz, fx, fy, fz);
        #else
            // regular update
        #endif
        */
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
        out << xx[i]
            << " "
            << xy[i]
            << " "
            << xz[i]
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
                << xx[0] << ", " << xy[0] << ", " << xz[0] << std::endl;
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
            NBodySimulation<float, float> nbs;
            run(&nbs,argc,argv,block_size);
        }


    }
    else{
        if (avx2){
            NBodySimulation<double, __m256d> nbs;
            run(&nbs,argc,argv,block_size);
        } else{
            NBodySimulation<double, double> nbs;
            run(&nbs,argc,argv,block_size);
        }
    }
    return 0;
}
