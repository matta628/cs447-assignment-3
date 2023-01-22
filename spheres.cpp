#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <assert.h>

#include <iostream>
#include <random>
#include <functional>
#include <chrono>

using namespace std;

double time(const std::function<void ()> &f) {
    f(); // Run once to warmup.
    // Now time it for real.
    auto start = std::chrono::system_clock::now();
    f();
    auto stop = std::chrono::system_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

void calculateLineLengths(){
	const int N = 1'000'000;
	alignas(32) static float lines[N][8];

	//generate points
	default_random_engine generator;
	uniform_real_distribution<float> distribution(-1.0,1.0);
	//each row represents two 4D points, A and B,
	//organized as follows: [A_x, B_x, A_y, B_y, A_z, B_z, A_t, B_t] 
	for (int i = 0; i < N; i++){
		for (int j = 0; j < 8; j++){
			lines[i][j] = distribution(generator);
		}
	}

	//sequential test
	alignas(32) static float l_s[N];
	auto seq = [&]() {
				   for(int i = 0; i < N; i++){
					   float dist_sqrd = 0.0;
					   for (int j = 0; j < 8; j+=2){
						   float diff = (lines[i][j+1]-lines[i][j]);
						   dist_sqrd += diff*diff;
					   }
					   l_s[i] = sqrt(dist_sqrd);
				   }
			   };
	double seq_time = (N/time(seq))/1000000;
	cout << "Sequential: " << seq_time << " Mops/s" << endl;
	
	//reorganize points to aid SIMD
	//take lines[][] in 8x8 matrix chunks and transpose each matrix
	for (int m = 0; m < N; m+=8){
		for (int i = 0; i < 8; i++){
			for (int j = i+1; j < 8; j++){
				float temp = lines[m+i][j];
				lines[m+i][j] = lines[m+j][i];
				lines[m+j][i] = temp;
			}
		}
	}

	//vectorized test
	alignas(32) static float l_v[N];
	auto vec =
		[&]{
			for (int m = 0; m < N; m+=8){
				__m256 ymm_start_x = _mm256_load_ps(lines[m]);
				__m256 ymm_end_x = _mm256_load_ps(lines[m+1]);
 				__m256 ymm_start_y = _mm256_load_ps(lines[m+2]);
				__m256 ymm_end_y = _mm256_load_ps(lines[m+3]);
				__m256 ymm_start_z = _mm256_load_ps(lines[m+4]);
				__m256 ymm_end_z = _mm256_load_ps(lines[m+5]);
 				__m256 ymm_start_t = _mm256_load_ps(lines[m+6]);
				__m256 ymm_end_t = _mm256_load_ps(lines[m+7]);

				__m256 ymm_diff_x = _mm256_sub_ps(ymm_end_x, ymm_start_x);
				__m256 ymm_diff_y = _mm256_sub_ps(ymm_end_y, ymm_start_y);
				__m256 ymm_diff_z = _mm256_sub_ps(ymm_end_z, ymm_start_z);
				__m256 ymm_diff_t = _mm256_sub_ps(ymm_end_t, ymm_start_t);

				__m256 ymm_diff_sqrd_x = _mm256_mul_ps(ymm_diff_x, ymm_diff_x);
				__m256 ymm_diff_sqrd_y = _mm256_mul_ps(ymm_diff_y, ymm_diff_y);
				__m256 ymm_diff_sqrd_z = _mm256_mul_ps(ymm_diff_z, ymm_diff_z);
				__m256 ymm_diff_sqrd_t = _mm256_mul_ps(ymm_diff_t, ymm_diff_t);
				
				__m256 ymm_dist = _mm256_sqrt_ps(ymm_diff_sqrd_x + ymm_diff_sqrd_y + ymm_diff_sqrd_z + ymm_diff_sqrd_t);
				
				_mm256_store_ps(l_v + m, ymm_dist);
			}
		};

	double vec_time = (N/time(vec))/1000000;
	cout << "Vector: " << vec_time <<" Mops/s" << endl;

	cout << "speedup: " << vec_time/seq_time << "x faster" << endl;

	for (int i = 0; i < N; i++){
		if (l_s[i] != l_v[i])
			assert(false);
	}
}

void sampleSphereSequential(int dim, int runs){
	//initialize histogram
	int histo[100] = {0};

	mt19937 generator(1234);
	uniform_real_distribution<double> distribution(-1.0,1.0);
	
	int within = 0;
	for (int i = 0; i < runs; i++){
		//generate dim-d point and calc distance squared
		double dist_2 = 0;
		
		for (int j = 0; j < dim; j++){
			double pt_j = distribution(generator);
			dist_2 += pt_j * pt_j;
		}

		//filter out points past the radius of the unit hypersphere
		//add valid points to histogram
		if (dist_2 <= 1){
			double dist = sqrt(dist_2);
			int interval = (int)(dist*100);
			histo[interval]++;
			within++;
		}
	}

	//print out histogram results
	for (int i = 0; i < 100; i++){
		printf("[%.2f - %.2f) = %2.2f\n", i/100.0, (i+1)/100.0, 100.0*histo[i]/within);
	}

	//print success rate
	//printf("%.4f%% of points were within the hypersphere\n", 100.0*within/runs);
}

void sampleSphereParallel(int dim, int runs){
	//initialize histogram
	int histo[100] = {0};

    #pragma omp parallel 
	{
		mt19937 generator(omp_get_thread_num());
		uniform_real_distribution<double> distribution(-1.0,1.0);

		int n_threads = omp_get_num_threads();
		int runs_per_thread = runs/n_threads;

		int thread_histo[100] = {0};

		for (int i = 0; i < runs_per_thread; i++){
			//generate dim-d point and calc distance squared
			double dist_2 = 0;
		
			for (int j = 0; j < dim; j++){
				double pt_j = distribution(generator);
				dist_2 += pt_j * pt_j;
			}
			
			//filter out points past the radius of the unit hypersphere
			//add valid points to histogram
			if (dist_2 <= 1){
				double dist = sqrt(dist_2);
				int interval = (int)(dist*100);
				thread_histo[interval]++;
			}
		}

		for (int i = 0; i < 100; i++){
			#pragma omp critical
			{
				histo[i] += thread_histo[i];
			}
		}
	}

	int within = 0;
	for (int i = 0; i < 100; i++){
		within += histo[i];
	}
	
	//print out histogram results
	for (int i = 0; i < 100; i++){
		printf("[%.2f - %.2f) = %2.2f%%\n", i/100.0, (i+1)/100.0, 100.0*histo[i]/within);
	}

	//print success rate
	//printf("%.4f%% of points were within the hypersphere\n", 100.0*within/runs);
}

int main(int argc, char **argv){
	//default number of runs
	int runs = 1'000'000;

	if (argc < 2 || argc > 3){
		printf("usage: ./spheres [dimension] [optional: runs]\n");
		return -1;
	}
	else if(argc == 3){
		//set runs if provided
		runs = atoi(argv[2]);
	}

	//Parse dimension from command line
	int dim = atoi(argv[1]);
	if (dim < 2 || dim > 16){
		printf("dimensionality from 2 to 16 only\n");
		return -1;
	}
	else if (dim == 2){
		printf("Plots within a circle\n");
	}
	else if (dim == 3){
		printf("Plots within a sphere\n");
	}
	else{
		printf("Plots within a %d-Ball\n", dim);
	}

	//sampleSphereSequential(dim, runs);
	sampleSphereParallel(dim, runs);

	printf("\nCalculate line lengths\n");
	calculateLineLengths();
	
	return 0;
}


