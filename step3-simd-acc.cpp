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

// SIMD intrinsics (here could add overloads for AVX-512, NEON etc...)

// AVX2 8x32-bit float
inline __m256   load(const float *addr){return _mm256_load_ps(addr); }
inline void     store(float *addr, const __m256 val){ _mm256_store_ps(addr, val); }
inline __m256   add(const __m256 a, const __m256 b){ return _mm256_add_ps(a, b); }
inline __m256   sub(const __m256 a, const __m256 b){ return _mm256_sub_ps(a, b); }
inline __m256   mul(const __m256 a, const __m256 b){ return _mm256_mul_ps(a, b); }
inline __m256   div(const __m256 a, const __m256 b){ return _mm256_div_ps(a, b); }
inline __m256   sqrt(const __m256 a){ return _mm256_sqrt_ps(a); }
inline void     setzero(__m256 &a){ a = _mm256_setzero_ps(); }
inline void     setconst(__m256 &a, const double c){ a = _mm256_set1_ps(c); }
inline void     rotate_halves(__m256 &a){ a = _mm256_permute_ps(a, 0b00111001); }
inline void     swap_halves(__m256 &a){ a = _mm256_permute2f128_ps(a, a, 1); }

// AVX2 4x64-bit double
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

// T - native data type
// V - SIMD vector with entries of type T 
template<typename T, typename V>
class NBodySimulationSIMD {

    T t;
    T tFinal;
    T tPlot;
    T tPlotDelta;
    T timeStepSize;
    T maxV;
    T minDx;

    int NumberOfBodies;
    int BlockSize = DEFAULT_BLOCK_SIZE;
    int NumberOfBlocks;
    const int VectorSize = sizeof(V)/sizeof(T);
    //T CollisionConstant;
    
    T *xx, *xy, *xz;
    T *vx, *vy, *vz;
    T *ax, *ay, *az;
    T *mass;
    V vtimeStepSize;

    std::ofstream videoFile;
    int snapshotCounter;
    int timeStepCounter;


private:
    inline void clear_forces()
    {
        for (int i = 0; i < NumberOfBodies; i+=VectorSize)
        {
            V zero;
            setzero(zero);
            store(ax+i, zero);
            store(ay+i, zero);
            store(az+i, zero);
            /*
            ax[i] = 0.0;
            ay[i] = 0.0;
            az[i] = 0.0;
            */
        }

    }

    inline void update_block(int block)
    {
        for (int i = block*BlockSize; i <= (block+1)*BlockSize - VectorSize; i+=VectorSize)
        {
            V xxi = load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V vxi = load(vx + i);
            V vyi = load(vy + i);
            V vzi = load(vz + i);
            V axi = load(ax + i);
            V ayi = load(ay + i);
            V azi = load(az + i);
            V massi = load(mass+i);

            store(xx+i, add(xxi, mul(vtimeStepSize, vxi)));
            store(xy+i, add(xyi, mul(vtimeStepSize, vyi)));
            store(xz+i, add(xzi, mul(vtimeStepSize, vzi)));
            store(vx+i, add(vxi, mul(vtimeStepSize, axi)));
            store(vy+i, add(vyi, mul(vtimeStepSize, ayi)));
            store(vz+i, add(vzi, mul(vtimeStepSize, azi)));

            /*
            V zero;
            setzero(zero);
            store(ax+i, zero);
            store(ay+i, zero);
            store(az+i, zero);
            */
        }
    }

    // here could check for collisions, eg return true of collision, false if not
    inline void vector_calc_accel(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &daxi, V &dayi, V &dazi, V &daxj, V &dayj, V &dazj)
    {
        V dx = sub(xxj, xxi);
        V dy = sub(xyj, xyi);
        V dz = sub(xzj, xzi);
        V distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
        V distance = sqrt(distance2);
        V distance3 = mul(distance2, distance);
        V ci = div(massj,distance3);
        V cj = div(massi,distance3);
        daxi = mul(ci, dx);
        dayi = mul(ci, dy);
        dazi = mul(ci, dz);
        daxj = mul(cj, dx);
        dayj = mul(cj, dy);
        dazj = mul(cj, dz);
    }

    /*
    inline void vector_calc_forces(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &dax, V &day, V &daz)
    {
        V dx = sub(xxj, xxi);
        V dy = sub(xyj, xyi);
        V dz = sub(xzj, xzi);
        V distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
        V distance = sqrt(distance2);
        V c = div(mul(massi, massj), mul(distance2, distance));

        dax = mul(c, dx);
        day = mul(c, dy);
        daz = mul(c, dz);
    }
    */
   
    inline void process_distinct_vectors(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &axi, V &ayi, V &azi, V &axj, V &ayj, V &azj)
    {
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < VectorSize/2; l++)
            {
                V daxi, dayi, dazi;
                V daxj, dayj, dazj;
                vector_calc_accel(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, daxi, dayi, dazi, daxj, dayj, dazj);

                axi = add(axi, daxi);
                ayi = add(ayi, dayi);
                azi = add(azi, dazi);
                axj =  sub(axj, daxj);
                ayj =  sub(ayj, dayj);
                azj =  sub(azj, dazj);

                rotate_halves(xxj);
                rotate_halves(xyj);
                rotate_halves(xzj);
                rotate_halves(massj);
                rotate_halves(axj);
                rotate_halves(ayj);
                rotate_halves(azj);
            }
            swap_halves(xxj);
            swap_halves(xyj);
            swap_halves(xzj);
            swap_halves(massj);
            swap_halves(axj);
            swap_halves(ayj);
            swap_halves(azj);
        }
    }

    inline void process_single_vector(V &xxi, V &xyi, V &xzi, V &massi, V &axi, V &ayi, V &azi)
    {
        // compare against all rotations but not the identity
        V xxj = xxi;
        V xyj = xyi;
        V xzj = xzi;
        V massj = massi;
        V dax, day, daz;
        // initial permutation
        for (int k = 1; k < VectorSize; k++)
        {
            rotate_halves(xxj);
            rotate_halves(xyj);
            rotate_halves(xzj);
            rotate_halves(massj);
            if (k == VectorSize / 2)
            {
                swap_halves(xxj);
                swap_halves(xyj);
                swap_halves(xzj);
                swap_halves(massj);
            }
            V daxi, dayi, dazi;
            V daxj, dayj, dazj;
            vector_calc_accel(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, daxi, dayi, dazi, daxj, dayj, dazj);
            axi = add(axi, daxi);
            ayi = add(ayi, dayi);
            azi = add(azi, dazi);
        }
    }

    inline void process_distinct_blocks(int blocki, int blockj) 
    {
        //std::cout << "process_distinct_blocks: " << blocki << ", " << blockj << std::endl;
        for (int i = blocki*BlockSize; i <= (blocki+1)*BlockSize - VectorSize; i+=VectorSize)
        {
            V xxi = load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V massi = load(mass + i);
            V axi = load(ax + i);
            V ayi = load(ay + i);
            V azi = load(az + i);
            for (int j = blockj*BlockSize; j <= (blockj+1)*BlockSize - VectorSize; j+=VectorSize)
            {
                V xxj = load(xx + j);
                V xyj = load(xy + j);
                V xzj = load(xz + j);
                V massj = load(mass + j);
                V axj = load(ax + j);
                V ayj = load(ay + j);
                V azj = load(az + j);
                process_distinct_vectors(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, axi, ayi, azi, axj, ayj, azj);
                store(ax+j, axj);
                store(ay+j, ayj);
                store(az+j, azj);
            }
            store(ax+i, axi);
            store(ay+i, ayi);
            store(az+i, azi);
        }
    }

    inline void process_single_block(int blocki)
    {
        //std::cout << "process_single_block: " << blocki <<std::endl;
        for (int i = blocki*BlockSize; i <= (blocki+1)*BlockSize - VectorSize; i+=VectorSize)
        {
            V xxi = load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V massi = load(mass + i);
            V axi = load(ax + i);
            V ayi = load(ay + i);
            V azi = load(az + i);
            process_single_vector(xxi, xyi, xzi, massi, axi, ayi, azi);
            for (int j = i+VectorSize; j <= (blocki+1)*BlockSize - VectorSize; j+=VectorSize)
            {
                //std::cout << "i=" << i << "; j=" << j << std::endl;
                V xxj = load(xx + j);
                V xyj = load(xy + j);
                V xzj = load(xz + j);
                V massj = load(mass + j);
                V axj = load(ax + j);
                V ayj = load(ay + j);
                V azj = load(az + j);
                process_distinct_vectors(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, axi, ayi, azi, axj, ayj, azj);
                store(ax+j, axj);
                store(ay+j, ayj);
                store(az+j, azj);
            }
            store(ax+i, axi);
            store(ay+i, ayi);
            store(az+i, azi);
        }
    }
    

public:
    NBodySimulationSIMD () :
        t(0), tFinal(0), tPlot(0), tPlotDelta(0), NumberOfBodies(0),
        mass(nullptr), timeStepSize(0), maxV(0), minDx(0), videoFile(nullptr),
        snapshotCounter(0), timeStepCounter(0) {};

    ~NBodySimulationSIMD () {
        free(xx); free(xy); free(xz);
        free(vx); free(vy); free(vz);
        free(ax); free(ay); free(az);
        free(mass);
    }

    void setUp(int argc, char** argv, int block_size) {
        NumberOfBodies = (argc-4) / 7;
        BlockSize = block_size;
        NumberOfBlocks = (NumberOfBodies + BlockSize - 1)/BlockSize;
        std::cout << "numberofblocks = " << NumberOfBlocks << std::endl;
        //exit(0);

        size_t array_size = (NumberOfBodies + VectorSize - 1) & -VectorSize;    // round up to nearest multiple of VectorSize
        size_t alloc_size = array_size*sizeof(T);
        xx = (T*)aligned_alloc(32, alloc_size);
        xy = (T*)aligned_alloc(32, alloc_size);
        xz = (T*)aligned_alloc(32, alloc_size);
        vx = (T*)aligned_alloc(32, alloc_size);
        vy = (T*)aligned_alloc(32, alloc_size);
        vz = (T*)aligned_alloc(32, alloc_size);
        ax = (T*)aligned_alloc(32, alloc_size);
        ay = (T*)aligned_alloc(32, alloc_size);
        az = (T*)aligned_alloc(32, alloc_size);
        mass = (T*)aligned_alloc(32, alloc_size);

        int readArgument = 1;

        tPlotDelta   = std::stof(argv[readArgument]); readArgument++;
        tFinal       = std::stof(argv[readArgument]); readArgument++;
        timeStepSize = std::stof(argv[readArgument]); readArgument++;
        std::cout << "tPlotDelta=" << tPlotDelta << std::endl;
        std::cout << "tFinal=" << tFinal << std::endl;
        std::cout << "timeStepSize=" << timeStepSize << std::endl;
        

        setconst(vtimeStepSize, timeStepSize);
        int i = 0;
        for (; i<NumberOfBodies; i++) {
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
        // pad with zeros to multiple of VectorSize
        for (; i % VectorSize != 0; i++)
        {
            xx[i] = 0.0;
            xy[i] = 0.0;
            xz[i] = 0.0;
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
            mass[i] = 0.0;
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
        int i = 0;
        std::cout << "vectorized process single vector" << std::endl;
        clear_forces();
        V xxi = load(xx + i);
        V xyi = load(xy + i);
        V xzi = load(xz + i);
        V massi = load(mass + i);
        V axi = load(ax + i);
        V ayi = load(ay + i);
        V azi = load(az + i);
        process_single_vector(xxi, xyi, xzi, massi, axi, ayi, azi);
        store(ax + i, axi);
        store(ay + i, ayi);
        store(az + i, azi);

        for (int i = 0; i < 8; i++)
        {
            std::cout << "ax[" << i << "]=" << ax[i] << std::endl;
        }        

        std::cout << "nonvectorized" << std::endl;
        clear_forces();
        for (int i = 0; i < 8; i++)
        {
            for (int j = i+1; j < 8; j++)
            {
                T dx = xx[j] - xx[i];
                T dy = xy[j] - xy[i];
                T dz = xz[j] - xz[i];
                T distance2 = dx*dx + dy*dy + dz*dz;
                T distance = std::sqrt(distance2);
                T c = mass[i]*mass[j]/(distance2*distance);

                T dax = c*dx;
                T day = c*dy;
                T daz = c*dz;

                ax[i] += dax;
                ay[i] += dax;
                az[i] += dax;
                ax[j] -= dax;
                ay[j] -= dax;
                az[j] -= dax;
            }
        }
        for (int i = 0; i < 8; i++)
        {
            std::cout << "ax[" << i << "]=" << ax[i] << std::endl;
        }        

    }

    inline void updateBody() {
        timeStepCounter++;
        minDx  = std::numeric_limits<T>::max();
        clear_forces();
        
        // Round-robin tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)
        for (int i = 0; i < NumberOfBlocks-1; i++)
        {
            // these threads will never access the same memory at once so can be run in parallel
            for (int j = 0; j < NumberOfBlocks/2; j++)
            {
                int a = (j == 0) ? 0 : ((i+j)%(NumberOfBlocks-1))+1;
                int b = ((i+NumberOfBlocks-j-1)%(NumberOfBlocks-1))+1;
                process_distinct_blocks(a, b);
            }
        }
        
        // can run in parallel
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            process_single_block(i);
            update_block(i);
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


template <typename T>
void run(T *nbs, int argc, char **argv, int block_size)
{
    // Code that initialises and runs the simulation.
    nbs->setUp(argc,argv,block_size);

    //nbs->test();
    //exit(0);

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
                << "  --avx2          Compute using explicit AVX2 implementation" << std::endl
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
            NBodySimulationSIMD<float, __m256> nbs;
            run(&nbs,argc,argv,block_size);
        }
        else{
            //NBodySimulation<float, float> nbs;
            //run(&nbs,argc,argv,block_size);
        }

    }
    else{
        if (avx2){
            NBodySimulationSIMD<double, __m256d> nbs;
            run(&nbs,argc,argv,block_size);
        } else{
            //NBodySimulation<double, double> nbs;
            //run(&nbs,argc,argv,block_size);
        }
    }
    
    return 0;
}
