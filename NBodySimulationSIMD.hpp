/**------------------------------------------------------------
 *                      SIMD intrinsics version
 *------------------------------------------------------------*/
#ifndef DEFAULT_BLOCK_SIZE
#define DEFAULT_BLOCK_SIZE 128
#endif 


// SIMD intrinsics (here could also add overloads for AVX-512, NEON etc...)
#ifdef __AVX2__
#include <immintrin.h>
// AVX2 8x32-bit float
inline __m256   load(const float *addr){return _mm256_load_ps(addr); }
inline void     store(float *addr, const __m256 val){ _mm256_store_ps(addr, val); }
inline __m256   add(const __m256 a, const __m256 b){ return _mm256_add_ps(a, b); }
inline __m256   sub(const __m256 a, const __m256 b){ return _mm256_sub_ps(a, b); }
inline __m256   mul(const __m256 a, const __m256 b){ return _mm256_mul_ps(a, b); }
inline __m256   div(const __m256 a, const __m256 b){ return _mm256_div_ps(a, b); }
//inline __m256   conddiv(const __m256 a, const __m256 b, const __m256 cond){ return _mm256_and_ps(cond, _mm256_div_ps(a, b)); }
/*
inline __m256   conddiv(const __m256 a, const __m256 b, const __m256 c, const __m256 cond){
        __m256 r1 = _mm256_and_ps(cond, div(a, b));
        __m256 r2 = _mm256_andnot_ps(cond, c);
        return _mm256_add_ps(r1, r2);
}
*/
inline __m256   condset(const __m256 a, const __m256 b, const __m256 cond){ return _mm256_add_ps(_mm256_and_ps(cond, a), _mm256_andnot_ps(cond, b)); }

inline __m256   sqrt(const __m256 a){ return _mm256_sqrt_ps(a); }
inline __m256   cond_nonzero(const __m256 a){ return _mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_NEQ_OQ); }
inline __m256   cond_and(const __m256 a, const __m256 b){ return _mm256_and_ps(a, b); }
inline float    hmin(const __m256 a){
                    __m128 a0 = _mm256_extractf128_ps(a, 0);
                    __m128 a1 = _mm256_extractf128_ps(a, 1);
                    __m128 m1 = _mm_min_ps(a0, a1);
                    __m128 m2 = _mm_permute_ps(m1, 0b10110001);
                    __m128 m3 = _mm_min_ps(m1, m2);
                    __m128 m4 = _mm_permute_ps(m1, 0b00001010);
                    return _mm_min_ps(m3, m4)[0];
                }
inline float    hmax(const __m256 a){
                    __m128 a0 = _mm256_extractf128_ps(a, 0);
                    __m128 a1 = _mm256_extractf128_ps(a, 1);
                    __m128 m1 = _mm_max_ps(a0, a1);
                    __m128 m2 = _mm_permute_ps(m1, 0b10110001);
                    __m128 m3 = _mm_max_ps(m1, m2);
                    __m128 m4 = _mm_permute_ps(m1, 0b00001010);
                    return _mm_max_ps(m3, m4)[0];
                }
inline __m256   vmin(const __m256 a, const __m256 b){ return _mm256_min_ps(a, b); }
inline __m256   vmax(const __m256 a, const __m256 b){ return _mm256_max_ps(a, b); }
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
/*
inline __m256d  conddiv(const __m256d a, const __m256d b, const __m256d c, const __m256d cond){
        __m256d r1 = _mm256_and_pd(cond, div(a, b));
        __m256d r2 = _mm256_andnot_pd(cond, c);
        return _mm256_add_pd(r1, r2);
}
*/
inline __m256d   condset(const __m256d a, const __m256d b, const __m256d cond){ return _mm256_add_pd(_mm256_and_pd(cond, a), _mm256_andnot_pd(cond, b)); }

inline __m256d   cond_nonzero(const __m256d a){ return _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_NEQ_OQ); }
inline __m256d   cond_and(const __m256d a, const __m256d b){ return _mm256_and_pd(a, b); }

inline __m256d  sqrt(const __m256d a){ return _mm256_sqrt_pd(a); }
inline double   hmin(const __m256d a){
                    __m128d a0 = _mm256_extractf128_pd(a, 0);
                    __m128d a1 = _mm256_extractf128_pd(a, 1); 
                    __m128d m1 = _mm_min_pd(a0, a1); 
                    __m128d m2 = _mm_permute_pd(m1, 1); 
                    return _mm_min_pd(m1, m2)[0]; 
                }
inline double   hmax(const __m256d a){
                    __m128d a0 = _mm256_extractf128_pd(a, 0);
                    __m128d a1 = _mm256_extractf128_pd(a, 1); 
                    __m128d m1 = _mm_max_pd(a0, a1); 
                    __m128d m2 = _mm_permute_pd(m1, 1); 
                    return _mm_max_pd(m1, m2)[0]; 
                }
inline __m256d   vmin(const __m256d a, const __m256d b){ return _mm256_min_pd(a, b); }
inline __m256d   vmax(const __m256d a, const __m256d b){ return _mm256_max_pd(a, b); }
inline void     setzero(__m256d &a){ a = _mm256_setzero_pd(); }
inline void     setconst(__m256d &a, const double c){ a = _mm256_set1_pd(c); }
inline void     rotate_halves(__m256d &a){ a = _mm256_permute_pd(a, 0b0101); }
inline void     swap_halves(__m256d &a){ a = _mm256_permute2f128_pd(a, a, 1); }
#endif

// T - native data type
// V - SIMD vector with entries of type T 
template<typename T, typename V>
class NBodySimulationSIMD {

    T t;
    T tFinal;
    T tPlot;
    T tPlotDelta;
    
    T timeStepSize;
    V vtimeStepSize;

    T maxV;
    V vmaxV;

    T minDx;
    V vminDx;

    V vnumericLimitMax;
    V vzero;

    int NumberOfBodies;
    int BlockSize = DEFAULT_BLOCK_SIZE;
    int NumberOfBlocks;
    int ArraySize;
    const int VectorSize = sizeof(V)/sizeof(T);
    T collisionConstant;
    
    T *xx, *xy, *xz;
    T *vx, *vy, *vz;
    T *fx, *fy, *fz;
    T *mass;

    std::ofstream videoFile;
    int snapshotCounter;
    int timeStepCounter;


private:
    inline void printvec(V v)
    {
        for (int i = 0; i < VectorSize; i++)
        {
            std::cout << v[i] << ", ";
        }
        std::cout << std::endl;
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

                    mass[NumberOfBodies-1] = 0.0;
                    NumberOfBodies--;
                    return;
                }
            }
        }
    }


    inline void clear_forces()
    {
        for (int i = 0; i < NumberOfBodies; i+=VectorSize)
        {
            V zero;
            setzero(zero);
            store(fx+i, zero);
            store(fy+i, zero);
            store(fz+i, zero);
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
            V fxi = load(fx + i);
            V fyi = load(fy + i);
            V fzi = load(fz + i);
            V massi = load(mass+i);
            store(xx+i, add(xxi, mul(vtimeStepSize, vxi)));
            store(xy+i, add(xyi, mul(vtimeStepSize, vyi)));
            store(xz+i, add(xzi, mul(vtimeStepSize, vzi)));
            V cond_nz_massi = cond_nonzero(massi);
            store(vx+i, add(vxi, mul(vtimeStepSize, condset(div(fxi, massi), vzero, cond_nz_massi))));
            store(vy+i, add(vyi, mul(vtimeStepSize, condset(div(fyi, massi), vzero, cond_nz_massi))));
            store(vz+i, add(vzi, mul(vtimeStepSize, condset(div(fzi, massi), vzero, cond_nz_massi))));
        }
    }

    // here could check for collisions, eg return true of collision, false if not
    inline void vector_calc_forces(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &dfx, V &dfy, V &dfz, V &distance, V &collisionCondition)
    {
        V dx = sub(xxj, xxi);
        V dy = sub(xyj, xyi);
        V dz = sub(xzj, xzi);
        V distance2 = add(mul(dx, dx), add(mul(dy, dy), mul(dz, dz)));
        distance = sqrt(distance2);
        V c = div(mul(massi, massj), mul(distance2, distance));
        c = condset(c, vzero, cond_nonzero(distance));
        dfx = mul(c, dx);
        dfy = mul(c, dy);
        dfz = mul(c, dz);

        distance = condset(distance, vnumericLimitMax, cond_nonzero(c));
        collisionCondition = condset(div(distance, add(massi, massj)), vnumericLimitMax, cond_nonzero(c));
        
        //V massimassj = mul(massi, massj);
        //distance = condset(distance, vnumericLimitMax, cond_nonzero(massi));
        //distance = condset(distance, vnumericLimitMax, cond_nonzero(massj));
        
        //distance = condset(distance, vnumericLimitMax, cond_nonzero(massimassj));
        //collisionCondition = condset(div(distance, add(massi, massj)), vnumericLimitMax, cond_nonzero(massi));
        //collisionCondition = condset(div(distance, add(massi, massj)), vnumericLimitMax, cond_nonzero(massj));


        /*
        std::cout << "massi: "; printvec(massi);
        std::cout << "massj: "; printvec(massj);
        std::cout << "massimassj: "; printvec(massimassj);
        std::cout << "distance: "; printvec(distance);

        printvec(massimassj);
        printvec(distance);
        */
    }

    inline void process_distinct_vectors(V &xxi, V &xyi, V &xzi, V &massi, V &xxj, V &xyj, V &xzj, V &massj, V &fxi, V &fyi, V &fzi, V &fxj, V &fyj, V &fzj, V &minDistance, V &minCollisionCondition)
    {
//        V m_minDistance = vnumericLimitMax;
//        V m_minCollisionCondition = vnumericLimitMax;
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < VectorSize/2; l++)
            {
                V dfx, dfy, dfz, distance, collisionCondition;
                vector_calc_forces(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, dfx, dfy, dfz, distance, collisionCondition);
        
                //distance = condset(distance, vnumericLimitMax, cond_nonzero(massj));
                //distance = condset(distance, vnumericLimitMax, cond_nonzero(massj));
                minDistance = vmin(minDistance, distance);
                //V collisionCondition = div(distance, add(massi, massj));
                //collisionCondition = condset(collisionCondition, vnumericLimitMax, cond_nonzero(massj));
                minCollisionCondition = vmin(minCollisionCondition, collisionCondition);
                
                fxi = add(fxi, dfx);
                fyi = add(fyi, dfy);
                fzi = add(fzi, dfz);
                fxj =  sub(fxj, dfx);
                fyj =  sub(fyj, dfy);
                fzj =  sub(fzj, dfz);
                rotate_halves(xxj);
                rotate_halves(xyj);
                rotate_halves(xzj);
                rotate_halves(massj);
                rotate_halves(fxj);
                rotate_halves(fyj);
                rotate_halves(fzj);
            }
            swap_halves(xxj);
            swap_halves(xyj);
            swap_halves(xzj);
            swap_halves(massj);
            swap_halves(fxj);
            swap_halves(fyj);
            swap_halves(fzj);
        }
        //minDistance = m_minDistance;
        //minCollisionCondition = m_minCollisionCondition;
        //std::cout << "m_minDistance: "; printvec(m_minDistance);
        //std::cout << "m_minCollisionCondition: "; printvec(m_minCollisionCondition);

    }

    inline void process_single_vector(V &xxi, V &xyi, V &xzi, V &massi, V &fxi, V &fyi, V &fzi, V &minDistance, V &minCollisionCondition)
    {
        //V m_minDistance = vnumericLimitMax;
        //V m_minCollisionCondition = vnumericLimitMax;

        // compare against all rotations but not the identity
        V xxj = xxi;
        V xyj = xyi;
        V xzj = xzi;
        V massj = massi;
        V dfx, dfy, dfz;
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
            V dfx, dfy, dfz, distance, collisionCondition;
            vector_calc_forces(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, dfx, dfy, dfz, distance, collisionCondition);
            minDistance = vmin(minDistance, distance);
            minCollisionCondition = vmin(minCollisionCondition, collisionCondition);
            
            //V collisionCondition = div(distance, add(massi, massj));
            //collisionCondition = condset(collisionCondition, vnumericLimitMax, cond_nonzero(massj));
            
//            printvec(distance);
            //exit(0);
            //distance = condset(distance, vnumericLimitMax, cond_nonzero(massj));
            //distance = condset(distance, vnumericLimitMax, cond_nonzero(massj));
            //m_minDistance = vmin(m_minDistance, distance);
    
            //V collisionCondition = div(distance, add(massi, massj));
            //collisionCondition = condset(collisionCondition, vnumericLimitMax, cond_nonzero(massj));
            //m_minCollisionCondition = vmin(m_minCollisionCondition, collisionCondition);

            fxi = add(fxi, dfx);
            fyi = add(fyi, dfy);
            fzi = add(fzi, dfz);
        }
        //std::cout << "m_minDistance: "; printvec(m_minDistance);
        //std::cout << "m_minCollisionCondition: "; printvec(m_minCollisionCondition);

        //minDistance = m_minDistance;
        //minCollisionCondition = m_minCollisionCondition;

    }

    inline void process_distinct_blocks(int blocki, int blockj, V &minDistance, V &minCollisionCondition) 
    {
        //V m_minDistance = vnumericLimitMax;
        //V m_minCollisionCondition = vnumericLimitMax;
        for (int i = blocki*BlockSize; i <= std::min((blocki+1)*BlockSize, ArraySize) - VectorSize; i+=VectorSize)
        {
            V xxi = load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V massi = load(mass + i);
            V fxi = load(fx + i);
            V fyi = load(fy + i);
            V fzi = load(fz + i);
            for (int j = blockj*BlockSize; j <= std::min((blockj+1)*BlockSize, ArraySize) - VectorSize; j+=VectorSize)
            {
                V xxj = load(xx + j);
                V xyj = load(xy + j);
                V xzj = load(xz + j);
                V massj = load(mass + j);
                V fxj = load(fx + j);
                V fyj = load(fy + j);
                V fzj = load(fz + j);
                //V singlevector_minDistance, singlevector_minCollisionCondition;
                process_distinct_vectors(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, fxi, fyi, fzi, fxj, fyj, fzj, minDistance, minCollisionCondition);
                
                //minDistance = vmin(minDistance, singlevector_minDistance);
                //minCollisionCondition = vmin(minDistance, singlevector_minCollisionCondition);

                store(fx+j, fxj);
                store(fy+j, fyj);
                store(fz+j, fzj);
            }
            store(fx+i, fxi);
            store(fy+i, fyi);
            store(fz+i, fzi);
        }
        //minDistance = m_minDistance;
        //minCollisionCondition = m_minCollisionCondition;
        //std::cout << "m_minDistance: "; printvec(m_minDistance);
        //std::cout << "m_minCollisionCondition: "; printvec(m_minCollisionCondition);

    }

    inline void process_single_block(int blocki, V &minDistance, V &minCollisionCondition)
    {
        //std::cout << "start block " << blocki << std::endl;
        //std::cout << "process_single_block: " << blocki <<std::endl;
        //V m_minDistance = vnumericLimitMax;
        //V m_minCollisionCondition = vnumericLimitMax;
        //std::cout << std::min((blocki+1)*BlockSize, ArraySize) - VectorSize << std::endl;
        for (int i = blocki*BlockSize; i <= std::min((blocki+1)*BlockSize, ArraySize) - VectorSize; i+=VectorSize)
        {
            V xxi = load(xx + i);
            V xyi = load(xy + i);
            V xzi = load(xz + i);
            V massi = load(mass + i);
            V fxi = load(fx + i);
            V fyi = load(fy + i);
            V fzi = load(fz + i);

            //V singlevector_minDistance, singlevector_minCollisionCondition;
            process_single_vector(xxi, xyi, xzi, massi, fxi, fyi, fzi, minDistance, minCollisionCondition);
            //m_minDistance = vmin(m_minDistance, singlevector_minDistance);
            //m_minCollisionCondition = vmin(m_minDistance, singlevector_minCollisionCondition);


            //printvec(minDistance);
            for (int j = i+VectorSize; j <= std::min((blocki+1)*BlockSize, ArraySize) - VectorSize; j+=VectorSize)
            {
                //std::cout << " j=" << j <<std::endl;
                V xxj = load(xx + j);
                V xyj = load(xy + j);
                V xzj = load(xz + j);
                V massj = load(mass + j);
                V fxj = load(fx + j);
                V fyj = load(fy + j);
                V fzj = load(fz + j);
                //V singlevector_minDistance, singlevector_minCollisionCondition;
                process_distinct_vectors(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, fxi, fyi, fzi, fxj, fyj, fzj, minDistance, minCollisionCondition);
                //m_minDistance = vmin(m_minDistance, singlevector_minDistance);
                //m_minCollisionCondition = vmin(m_minDistance, singlevector_minCollisionCondition);

                //process_distinct_vectors(xxi, xyi, xzi, massi, xxj, xyj, xzj, massj, fxi, fyi, fzi, fxj, fyj, fzj, m_minDistance, m_minCollisionCondition);
                // where mass[i] = 0 set m_minCollisionCondition = max val


                store(fx+j, fxj);
                store(fy+j, fyj);
                store(fz+j, fzj);
            }
            store(fx+i, fxi);
            store(fy+i, fyi);
            store(fz+i, fzi);
        }
        //std::cout << "done block" << std::endl;
        //minDistance = m_minDistance;
        //minCollisionCondition = m_minCollisionCondition;
    }
    

public:
    NBodySimulationSIMD () :
        t(0), tFinal(0), tPlot(0), tPlotDelta(0), NumberOfBodies(0),
        mass(nullptr), timeStepSize(0), maxV(0), minDx(0), videoFile(nullptr),
        snapshotCounter(0), timeStepCounter(0) {};

    ~NBodySimulationSIMD () {
        free(xx); free(xy); free(xz);
        free(vx); free(vy); free(vz);
        free(fx); free(fy); free(fz);
        free(mass);
    }


    inline void updateBody() {
        timeStepCounter++;
        minDx  = std::numeric_limits<T>::max();
        V vminDistance = vnumericLimitMax;
        V vminCollisionCondition = vnumericLimitMax;

        clear_forces();

        // Round-robin tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)
        // (so jobs for each round can run parallel)
        for (int i = 0; i < NumberOfBlocks-1; i++)
        {
            for (int j = 0; j < NumberOfBlocks/2; j++)
            {
                int a = (j == 0) ? 0 : ((i+j)%(NumberOfBlocks-1))+1;
                int b = ((i+NumberOfBlocks-j-1)%(NumberOfBlocks-1))+1;
                process_distinct_blocks(a, b, vminDistance, vminCollisionCondition);
            }
        }
      
        // can run in parallel
        for (int i = 0; i < NumberOfBlocks; i++)
        {
            process_single_block(i, vminDistance, vminCollisionCondition);
        }
            
        minDx = hmin(vminDistance);
        T minCollisionCondition = hmin(vminCollisionCondition);

        if (minCollisionCondition < collisionConstant)
        {
            //std::cout << "there has been collision" << std::endl;
            process_collision();
            return;
        }
        

        for (int i = 0; i < NumberOfBlocks; i++)
        {
            update_block(i);
        }

        //std::cout << "vminDx[0]=" << vminDx[0] << std::endl;
        t += timeStepSize;
    }






































    void setUp(int argc, char** argv, int block_size) {
        setconst(vnumericLimitMax, std::numeric_limits<T>::max());
        setzero(vzero);

        NumberOfBodies = (argc-4) / 7;
        ArraySize = (NumberOfBodies + VectorSize - 1) & -VectorSize;    // round up to nearest multiple of VectorSize
        BlockSize = (std::min(block_size, NumberOfBodies) + VectorSize - 1) & -VectorSize;
        NumberOfBlocks = (NumberOfBodies + BlockSize - 1)/BlockSize;

        collisionConstant = 1.0/(100*NumberOfBodies);


        std::cout << "NumberOfBodies = " << NumberOfBodies << std::endl;
        std::cout << "VectorSize = " << VectorSize << std::endl;
        std::cout << "BlockSize = " << BlockSize << std::endl;
        std::cout << "numberofblocks = " << NumberOfBlocks << std::endl;
        std::cout << "collisionConstant = " << collisionConstant << std::endl;
        //exit(0);
        
        std::cout << "array_size = " << ArraySize << std::endl;

        size_t alloc_size = ArraySize*sizeof(T);
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

        for (int i = 0; i < ArraySize; i+=VectorSize)
        {
            V massi = load(mass + i);
            //std::cout << "mass: "; printvec(massi);
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