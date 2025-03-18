/**------------------------------------------------------------
 *                      OpenMP version
 *------------------------------------------------------------*/
#ifndef DEFAULT_BLOCK_SIZE
#define DEFAULT_BLOCK_SIZE 128
#endif 


// T - native data type
template<typename T>
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
    int NumberOfBlocks;

    T minCollisionCondition;
    T collisionConstant;

    
    T * __restrict xx;
    T * __restrict xy;
    T * __restrict xz;
    T * __restrict vx;
    T * __restrict vy;
    T * __restrict vz;
    T * __restrict fx;
    T * __restrict fy;
    T * __restrict fz;
    T * __restrict mass;
    
    std::ofstream videoFile;
    int snapshotCounter;
    int timeStepCounter;


private:
    inline void clear_forces()
    {
        #pragma omp simd
        for (int i = 0; i < NumberOfBodies; i+=1)
        {
            fx[i] = 0;
            fy[i] = 0;
            fz[i] = 0;
        }

    }

    inline T update_block(int block)
    {
        T blockMaxV2 = 0; 
        for (int i = block*BlockSize; i <= std::min((block+1)*BlockSize, NumberOfBodies) - 1; i+=1)
        {
            xx[i] += timeStepSize*vx[i];
            xy[i] += timeStepSize*vy[i];
            xz[i] += timeStepSize*vz[i];

            vx[i] += timeStepSize*fx[i]/mass[i];
            vy[i] += timeStepSize*fy[i]/mass[i];
            vz[i] += timeStepSize*fz[i]/mass[i];

            T v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
            blockMaxV2 = std::max(blockMaxV2, v2);
        }
        return blockMaxV2;
    }

    // here could check for collisions, eg return true of collision, false if not
    #pragma omp declare simd
    inline void calc_forces(T &xxi, T &xyi, T &xzi, T &massi, T &xxj, T &xyj, T &xzj, T &massj, T &dfx, T &dfy, T &dfz, T &distance)
    {
        T dx = xxj - xxi;
        T dy = xyj - xyi;
        T dz = xzj - xzi;
        T distance2 = dx*dx + dy*dy + dz*dz;
        distance = std::sqrt(distance2);
        T c = massi*massj/(distance2*distance);
        dfx = c*dx;
        dfy = c*dy;
        dfz = c*dz;
    }


    // doing this way is inefficient bo it won't be done often
    inline void process_collision()
    {
        for (int i = 0; i < NumberOfBodies; i++)
        {
            for (int j = i+1; j < NumberOfBodies; j++)
            {
                T dx = xx[j]-xx[i];
                T dy = xy[j]-xy[i];
                T dz = xz[j]-xz[i];
                T dst = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dst/(mass[i]+mass[j]) < collisionConstant)
                {
                    xx[i] = (mass[i]*xx[i] + mass[j]*xx[j])/(mass[i]+mass[j]);
                    xy[i] = (mass[i]*xy[i] + mass[j]*xy[j])/(mass[i]+mass[j]);
                    xz[i] = (mass[i]*xz[i] + mass[j]*xz[j])/(mass[i]+mass[j]);

                    vx[i] = (mass[i]*vx[i] + mass[j]*vx[j])/(mass[i]*mass[j]);
                    vy[i] = (mass[i]*vy[i] + mass[j]*vy[j])/(mass[i]*mass[j]);
                    vz[i] = (mass[i]*vz[i] + mass[j]*vz[j])/(mass[i]*mass[j]);

                    mass[i] += mass[j];

                    // remove body l by replacing it with the last one 
                    xx[j] = xx[NumberOfBodies-1];
                    xy[j] = xy[NumberOfBodies-1];
                    xz[j] = xz[NumberOfBodies-1];
                    vx[j] = vx[NumberOfBodies-1];
                    vy[j] = vy[NumberOfBodies-1];
                    vz[j] = vz[NumberOfBodies-1];
                    mass[i] = mass[NumberOfBodies-1];
                    NumberOfBodies--;
                    return;
                }
            }
        }
    }

    inline bool process_distinct_blocks(int blocki, int blockj) 
    {
        T m_minDx = std::numeric_limits<T>::max();
        T m_minCollisionCondition = std::numeric_limits<T>::max();

        #pragma omp simd
        for (int i = blocki*BlockSize; i <= std::min((blocki+1)*BlockSize, NumberOfBodies) - 1; i++)
        {
            T xxi = xx[i];
            T xyi = xy[i];
            T xzi = xz[i];
            T massi = mass[i];
            T fxi = fx[i];
            T fyi = fy[i];
            T fzi = fz[i];
            
            #pragma omp simd reduction(+:fxi, fyi, fzi) reduction(min:m_minDx, m_minCollisionCondition)
            for (int j = blockj*BlockSize; j <= std::min((blockj+1)*BlockSize, NumberOfBodies) - 1; j++)
            {
                T xxj = xx[j];
                T xyj = xy[j];
                T xzj = xz[j];
                T massj = mass[j];             
                T dfx, dfy, dfz, distance;
                calc_forces(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, dfx, dfy, dfz, distance);
                m_minDx = std::min(m_minDx, distance);
                m_minCollisionCondition = std::min(m_minCollisionCondition, distance/(massi+massj));
                fx[j] -= dfx;
                fy[j] -= dfy;
                fz[j] -= dfz;
                fxi += dfx;
                fyi += dfy;
                fzi += dfz;
            }
            fx[i] = fxi;
            fy[i] = fyi;
            fz[i] = fzi;         
        }
        minDx = std::min(minDx, m_minDx);
        return m_minCollisionCondition < collisionConstant;       
    }




    inline bool process_single_block(int blocki)
    {
        T m_minDx = std::numeric_limits<T>::max();
        T m_minCollisionCondition = std::numeric_limits<T>::max();

        //#pragma omp simd
        for (int i = blocki*BlockSize; i <= std::min((blocki+1)*BlockSize, NumberOfBodies) - 1; i++)
        {
            T fxi = fx[i];
            T fyi = fy[i];
            T fzi = fz[i];
            T xxi = xx[i];
            T xyi = xy[i];
            T xzi = xz[i];
            T massi = mass[i];

            #pragma omp simd reduction(min: m_minDx, m_minCollisionCondition)
            for (int j = blocki*BlockSize; j <= i - 1; j++)
            {       
                T xxj = xx[j];
                T xyj = xy[j];
                T xzj = xz[j];
                T massj = mass[j];             
                T dfx, dfy, dfz, distance;
                calc_forces(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, dfx, dfy, dfz, distance);
                m_minDx = std::min(m_minDx, distance);
                m_minCollisionCondition = std::min(m_minCollisionCondition, distance/(massi+massj));
                fxi += dfx;
                fyi += dfy;
                fzi += dfz;
            }

            #pragma omp simd reduction(min: m_minDx, m_minCollisionCondition)
            for (int j = i+1; j <= std::min((blocki+1)*BlockSize, NumberOfBodies) - 1; j++)
            {       
                T xxj = xx[j];
                T xyj = xy[j];
                T xzj = xz[j];
                T massj = mass[j];             
                T dfx, dfy, dfz, distance;
                calc_forces(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, dfx, dfy, dfz, distance);
                m_minDx = std::min(m_minDx, distance);
                m_minCollisionCondition = std::min(m_minCollisionCondition, distance/(massi+massj));
                fxi += dfx;
                fyi += dfy;
                fzi += dfz;
            }

            fx[i] = fxi;
            fy[i] = fyi;
            fz[i] = fzi;
        }
        minDx = std::min(minDx, m_minDx);
        return m_minCollisionCondition < collisionConstant;
    }

public:
    NBodySimulation () :
        t(0), tFinal(0), tPlot(0), tPlotDelta(0), NumberOfBodies(0),
        mass(nullptr), timeStepSize(0), maxV(0), minDx(0), videoFile(nullptr),
        snapshotCounter(0), timeStepCounter(0) {};

    ~NBodySimulation () {
        free(xx); free(xy); free(xz);
        free(vx); free(vy); free(vz);
        free(fx); free(fy); free(fz);
        free(mass);
    }

    void setUp(int argc, char** argv, int block_size) {
        NumberOfBodies = (argc-4) / 7;
        collisionConstant = 1.0/(100*NumberOfBodies);

        BlockSize = block_size;
        NumberOfBlocks = (NumberOfBodies + BlockSize - 1)/BlockSize;
        //std::cout << "numberofblocks = " << NumberOfBlocks << std::endl;
        
        size_t array_size = NumberOfBodies;    // round up to nearest multiple of VectorSize
        
        //size_t array_size = (NumberOfBodies + VectorSize - 1) & -VectorSize;    // round up to nearest multiple of VectorSize
        size_t alloc_size = array_size*sizeof(T);
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
        std::cout << "tPlotDelta=" << tPlotDelta << std::endl;
        std::cout << "tFinal=" << tFinal << std::endl;
        std::cout << "timeStepSize=" << timeStepSize << std::endl;
        

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
        minCollisionCondition = std::numeric_limits<T>::max();

    /**
     * Processing forces
     **/
        clear_forces();
        // Round-robin tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)
        // (so jobs for each round can be parallel)
        bool collision = false;
        for (int i = 0; i < NumberOfBlocks-1; i++)
        {
            #pragma omp parallel for
            for (int j = 0; j < NumberOfBlocks/2; j++)
            {
                int a = (j == 0) ? 0 : ((i+j)%(NumberOfBlocks-1))+1;
                int b = ((i+NumberOfBlocks-j-1)%(NumberOfBlocks-1))+1;
                if (process_distinct_blocks(a, b)) collision = true;
            }
        }
        if (collision)
        {
            process_collision();
            return;
        }

        // can run in parallel
        #pragma omp parallel for
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            if (process_single_block(i)) collision = true;
        }
        if (collision)
        {
            process_collision();
            return;
        }

        T maxV2 = 0;
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            T blockMaxV2 = update_block(i);
            maxV2 = std::max(maxV2, blockMaxV2);
        }
        maxV = std::sqrt(maxV2);

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