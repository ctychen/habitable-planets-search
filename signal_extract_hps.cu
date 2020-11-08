#include <iostream>
#include <fstream>
#include <ctime>// include this header
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iomanip>
#include <stdio.h>
using namespace std;


//nvcc -o test_hps signal_extract_hps.cu
//CUDA_VISIBLE_DEVICES=1 ./test1
//CUDA_VISIBLE_DEVICES=1  ./test_hps 64768 testid
__global__ void fmod_gpu(double *ptime,double *period){

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  ptime[index] = fmod(ptime[index] + (period[0]/2.0), period[0]) - period[0]/2.0;
  __syncthreads();

}

__global__ void fill_gpu(double *gpu_cube, double *slides, int iter_p, int no_block){

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int index_fill = blockIdx.x * blockDim.x + threadIdx.x+iter_p*no_block;
  gpu_cube[index_fill] = slides[index];
  __syncthreads();

}



__global__ void shared_mean(double *p_flux, double *p_out, int no_thread){

  const int index_shared = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double shared_mem[];
  shared_mem[index_shared] = p_flux[index];
  __syncthreads();


  double sum =0.0;
  for(int i = 0; i<no_thread; i++){
    sum += shared_mem[i];
  }
  p_out[blockIdx.x] =sum/(double)no_thread;
}



int main(int argc, char *argv[])
{


  int start_s=clock();
  int p_num = 44000; //38400;//32640; //38400;//100000;
  //int vector_size = 64768;
  int vector_size = atoi(argv[1]);

  thrust::host_vector<double> time_array(vector_size);
  thrust::host_vector<double> flux_array(vector_size);
  thrust::host_vector<double> period_array(p_num);

  double t_sum_fmod=0.0;

  ifstream input("search_array_hps.txt");
  //read the searched period here:
  for(int ii = 0; ii < p_num; ii++)
  {
    input >> period_array[ii];
  }

  ifstream input1("kepler_buffer_hps.txt");
  //read the data here:
  for(int i = 0; i < vector_size; i++)
  {
    input1 >> time_array[i] >> flux_array[i];
  }

  thrust::device_vector<double> time_per = time_array;
  thrust::device_vector<double> flux_per = flux_array;

  const int numblocks = 8192;
  const int blocksize = vector_size/numblocks;

  thrust::device_vector<double> flux_cube(numblocks*p_num,1);
  double* p_cube = thrust::raw_pointer_cast(&flux_cube[0]);
  //starter of the for loop
  for(int p_iter = 0; p_iter < p_num; p_iter++){


    thrust::device_vector<double> period(1,0);
    thrust::device_vector<double> flux_final(8192,1);


    thrust::device_vector<double> time = time_per;
    thrust::device_vector<double> flux = flux_per;


    period[0] = period_array[p_iter];

    double* d_time = thrust::raw_pointer_cast(&time[0]);
    double* d_flux = thrust::raw_pointer_cast(&flux[0]);
    double* d_period = thrust::raw_pointer_cast(&period[0]);

    double* p_out = thrust::raw_pointer_cast(&flux_final[0]);


    //num_blocks, blocksize
    fmod_gpu<<<numblocks,blocksize>>>(d_time, d_period);

    int start_s=clock();
    thrust::stable_sort_by_key(thrust::device,time.begin(),time.end(), flux.begin(), thrust::less<float>());

    int start_s3=clock();
    t_sum_fmod += double (start_s3 - start_s);
    //num_blocks, blocksize
    shared_mean<<<numblocks,blocksize, blocksize*sizeof(double)>>>(d_flux,  p_out, blocksize);



    //thrust::host_vector<double> flux_out = flux_final;
    std::cout<<"start to fast fill"<<endl;
    fill_gpu<<<128,64>>>(p_cube, p_out, p_iter, numblocks);
    std::cout<<p_iter+1<<endl;


    //std::cout<<&p_iter<<endl;
    //std::cout<<period[0]<<endl;


    //std::string data_out_name = "/media/etdisk2/Yinan_Zhao/gpu_codes_new/folded_data/data_gpu"+std::to_string(p_iter)+".txt";
    //std::string data_out_name = "./folded_data/data_gpu"+std::to_string(p_iter)+".txt";
    //std::ofstream out(data_out_name);
    //for(int kk = 0; kk<256; kk++){
    //  out<<std::setprecision(16)<<flux_out[kk]<<std::endl;
    //}

  }

  std::cout<<"start to save"<<endl;
  thrust::host_vector<double> flux_out = flux_cube;
  //std::cout<<argv[2]<<endl;

  //string data_out_name = "test_data"+".bin";
  //string dataname= "/media/rd3/cchen/cchen/hsp_gpu_fold/all_kepler_folded/" + std::string("data_") +std::string(argv[2])+ std::string( "_hps.bin");
  string dataname= std::string("/media/rd3/cchen/cchen/hsp_gpu_fold/hsp_100to200/data_") +std::string(argv[2])+ std::string( "_hps.bin");
  FILE *file = fopen(dataname.c_str(), "wb");
      fwrite(flux_out.data(), sizeof(double), numblocks*p_num, file);
      fclose(file);


  //std::string data_out_name = "/media/etdisk2/Yinan_Zhao/gpu_codes_new/data_gpu.txt";
  //std::ofstream out(data_out_name);
  //for(int kk = 0; kk<256*100000; kk++){
  //  out<<std::setprecision(16)<<flux_out[kk]<<std::endl;
  //}


//end of the for loop

  int stop_s=clock();
  double t = double (stop_s - start_s);
  cout << "time: " << (t / double(CLOCKS_PER_SEC)) << endl;
  cout << "file time: " << (t_sum_fmod / double(CLOCKS_PER_SEC)) << endl;
  return 0;
}
