#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<device_functions.h>
#include<device_atomic_functions.h>

#include<malloc.h>
#include<stdlib.h>
#include<cuda.h>

#include<iostream>
#include <stdio.h>
#include<math.h>
#include"device_launch_parameters.h"
#include<algorithm>
#include <cuda_runtime_api.h> //for synchthreads

#include<ctime>

#define N (1024 * 1024)						//total number of elements
#define SUB_ARR_SIZE 1024 			//size of the smaller sub-arrays, shared array size 
#define GRID_SIZE N/SUB_ARR_SIZE	//number of sub-arrays, number of blocks

// this function finds the smallest value in specific positions, i.e., indices in the array respecting that the returned value is from a non-copied sub-array  
int SmallestElementIndex(int * arr, int * indices, int grid_size, int sub_arr_size)  //was int * indices
{
	int index = 0;
	
	//find the first non-copied subarray 
	for(int k=0; k<grid_size; k++)
	{
		if(indices[k]/(k+1) < sub_arr_size)
		{
			index=k;
			break;
		}
	}

	//compare the value at the found index with othere non-copied sub-arrays and return the index of the minimum
	for(int i=index+1; i<grid_size; i++)
		if(arr[indices[i]] < arr[indices[index]] && indices[i]/(i+1) < sub_arr_size)
			index = i;
	return index;
}

// this function finds the index of the min/equal element
int min_index (int * arr, int size)
{
	int index = 0;
	 for(int i=1; i<size; i++)
		 if(arr[i] < arr[index])
			 index = i;
	 return index;
}


void merge(int * arr, int grid_size, int sub_arr_size, int * result_arr)
{
	int  * indices	=	new int [grid_size];			//moving indices, each index points at the beginning of a sub-array at the beginning
	int  * counters	=	new int [grid_size];			//counts the number of elements, in the corresponding sub-array, whch have been copied to the result
	
	int argmin, result_index=0;							//argmin is used for finding the last sub-array which has not copied completely yet
	indices[0] = 0;
	counters[0]=0;

	//initialization
	for(int k=1; k<grid_size; k++)
	{
		indices[k] = indices[k-1] + sub_arr_size;
		counters[k]=0;
	}
	
	int merged_arrays =0;		//number of completely copied sub-arrays
	
	//while there are more than one sub-array not copied
	while(merged_arrays <grid_size-1)
		{
			//find the smallest value among the ones pointed by indices
			argmin = SmallestElementIndex(arr, indices, grid_size, sub_arr_size);
			
			//copy the smallest value to the result and increase the corresponding index
			result_arr[result_index] = arr[indices[argmin]];
			indices[argmin]++;
			counters[argmin]++;

			if((counters[argmin]%sub_arr_size)==0)
				merged_arrays++;
			result_index++;
		}
	
	//for the last sub-array
	int min_idx = indices[min_index(counters, grid_size)];
	
	for(int i = result_index; i< (grid_size * sub_arr_size); i++, min_idx++)
		result_arr[i] = arr[min_idx];
		
	delete [] counters;
	delete [] indices;
	//delete counters;
	//delete indices;
}

// this function can be used for debugging
__device__ void print_arr(int * arr, int size, int thread_idx)
{
	for(int i =0; i<size;i++)
		printf("array[%d]=\t %d \n",size * thread_idx + i, arr[i]);
}

//This function takes a thread-part array along with the number of elements, thread indices are for debugging 
__device__ void swap(int * sub_arr, bool up, int thread_size, int thread_idx, int p, int q)
{
	
	int j=0 , i = thread_size/2;			//two indices, one pointing at the beggining of the array, the pther at the half
	int temp;								// used for swapping two elements

	//print_arr(sub_arr, thread_size, thread_idx); // was used for debugging
	
	while(i<= thread_size-1)
	{
		//this comparison cares about the direction as well as the values, values should be sorted in the defined direction, or swap
		if((sub_arr[j] > sub_arr[i]) == !up)
		{
			//printf("swapping \t%d \t%d, \tthread \t%d \t p=\t%d \t q=\t%d \n", sub_arr[j] , sub_arr[i], thread_idx, p , q);

			temp = sub_arr[j];
			sub_arr[j] = sub_arr[i];
			sub_arr[i] = temp;			
		}
		i++;
		j++;
	}
}

//bitonic sort kernel
__global__ void bitonic(int * arr)
{
	__shared__ int  shared_arr [SUB_ARR_SIZE] ;						//the block shared array
	__shared__ int threads_per_block;								//dynamic block size, phase (p) and sub-phase (q) dependant
	__shared__ int thread_size;										//dynamic thread size, phase and subphase dependant
	int direction_assisstant;										//to be used in deciding the direction of comparing elements for each thread
	int idx = threadIdx.x + blockDim.x * blockIdx.x;				//global thread index
	
	//thread 0 in each block is the responsible for controlling block and thread sizes for each sub-phase
	if(threadIdx.x == 0)
	{
		
		threads_per_block = SUB_ARR_SIZE;
		thread_size= 2;
	}
	
	//in parallel, each thread copies two elements from the original array to the corresponding position in the shared array
	for(int i=0; i< 2; i++)
	{
		shared_arr[threadIdx.x*2 +i] = arr[idx*2 +i];	
	}
	
	__syncthreads(); 
	//here,  we are sure that data has been copied from global to shared mem completely

	//loop for each phase, p
	for(int p=0; p<log2f(SUB_ARR_SIZE); p++)
	{
		//printf("---------------------------------------\n");
		
		//thread 0 in each block changes the sizes of blocks and threads using the value of p 
		if(threadIdx.x == 0)	
		{
			threads_per_block /=powf(2, p+1);
			thread_size= SUB_ARR_SIZE/threads_per_block;
			//printf("thread size=\t%d,\t thread \t%d,\t block \t%d\n", thread_size, threadIdx.x, blockIdx.x);
			
		}	
		__syncthreads();  
		//Here all the threads in this block know the propper size of themselves and blocks, as well
		
		for(int q=0; q<=p; q++)
		{
			//activate the required number of threads only
			if(threadIdx.x <threads_per_block )
			{
				// calculate the up-to-date global index for each block using threads_per_block instead of blockDim.x
				idx = threadIdx.x + threads_per_block * blockIdx.x;

				//printf("p =\t%d, \t q =\t %d, \t I am thread\t %d, \t%d  \t in block\t%d \n", p, q, idx, threadIdx.x, blockIdx.x);
				//printf("__________ THREAD SIZE =\t%d, idx= \t%d \n", thread_size, idx);

				
				direction_assisstant = powf(2,q);
				//call the swap function which does all the needed swappings inside the corresponding part of the shared array
				swap(shared_arr + threadIdx.x * thread_size, (threadIdx.x/direction_assisstant)%2, thread_size, idx, p, q);  //idx, p, andq are for debugging only
			}			
			__syncthreads();

			//here all threads in this block have finished the current subphase
			//update thread and block sizes for next subphase and suspend til all the threads in the current block have known the new values
			
			if(threadIdx.x == 0)
			{
				threads_per_block *= 2;
				thread_size = SUB_ARR_SIZE/threads_per_block;
			}
			
			__syncthreads();

		}
		
	}

	__syncthreads();
	//here, all block-thread have finished all the sub-phases
	//copy the sorted shared array to the propper position in the result array, on parallel
	for(int i=0; i< 2; i++)
	{
		arr[idx * 2 +i] = shared_arr[threadIdx.x * 2 + i];		
	}

}

int main()
{

    int * arr;						//to save the unsorted data 
	int * resulted_arr;				//to save the partially-sorted data
	int * merged_sorted_arr;		//to save the completely-sorted data
	int * dev_arr;					//to use in device
	int size = N * sizeof(int);

	cudaEvent_t start, stop;
	clock_t begin, end;
	float elapsed_time;				//to measure the parallel processing time in sorting the array partially
	
	//random intialization, memory allocation

	cudaMalloc((void **) &dev_arr, size);
	arr = (int *)malloc(size);
	merged_sorted_arr = (int *)malloc(size);
	resulted_arr = (int *)malloc(size); 
	
	for(int i=0; i<N; i++)
	{
		arr[i] = rand();
		
	//	printf("element %d =  %d \n", i, arr[i]);
	}

	//copy data from host to device
	cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
	
	std::cout<<"number of phases  "<<log10(SUB_ARR_SIZE)/log10(2)<<std::endl;
	
	printf("NUMBER OF ELEMENTS:\t%d\t SUB_ARR_SIZE \t%d\t GRID_SIZE \t%d\n", N, SUB_ARR_SIZE, GRID_SIZE);
	
	//record timing
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	////////////////////////////////////////////// kernel invokation
	bitonic<<<GRID_SIZE, SUB_ARR_SIZE/2>>>(dev_arr);

	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	printf("parallel part, Elapsed_time:\t %3.10f \t ms \n", elapsed_time);
	
	//copying the result from device to host
	cudaMemcpy(resulted_arr,dev_arr, size, cudaMemcpyDeviceToHost);

	//for(int i=0; i<N;)
	//{		
	//	//printf("resulted element %d =  %d \n", i, resulted_arr[i]);
	//	i++;
	//	//to split sub-arrays in display, for the sake of easyness of monitoring results
	//	/*if(i % SUB_ARR_SIZE ==0)
	//		printf("---------------------------------------\n");*/
	//	//This was for debugging, when errors show this line of hashes#
	//	if(i<N)
	//		if(resulted_arr[i] < resulted_arr[i-1] && ((i % SUB_ARR_SIZE) !=0) )
	//			printf("#############################################\n");
	//}
	
	//To merge sorted sub-arrays, measre sequential complexity
	begin = clock();
	merge(resulted_arr, GRID_SIZE, SUB_ARR_SIZE, merged_sorted_arr);
	end = clock();
	elapsed_time = (float)(end-begin)/CLOCKS_PER_SEC;
	printf("sequential part, Elapsed_time:\t %3.3f \t ms \n", elapsed_time);

	//to print completely-sorted array
	//for(int i=0; i<N;)
	//{		
	//	//printf("sorted resulted element \t%d\t =  %d \n", i, merged_sorted_arr[i]);
	//	i++;
	//	//again for debugging, when errors, show a line of hashes#
	//	if(i<N)
	//		if(merged_sorted_arr[i] < merged_sorted_arr[i-1])
	//			printf("#############################################\n");
	//}
	std::cout<<"median element is "<<merged_sorted_arr[N/2]<<"\n";

	//liberating the allocated memory in both host and device
	cudaFree(dev_arr);
	free(arr);
	free(resulted_arr);
	free(merged_sorted_arr);

}
