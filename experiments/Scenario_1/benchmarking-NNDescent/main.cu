#include <assert.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <vector>
#include <thread> 
#include <unistd.h>


#include "gpuknn/gen_large_knngraph.cuh"
#include "gpuknn/knncuda_tools.cuh"
#include "gpuknn/knnmerge.cuh"
#include "gpuknn/nndescent.cuh"
#include "tools/distfunc.hpp"
#include "tools/evaluate.hpp"
#include "tools/filetool.hpp"
#include "tools/knndata_manager.hpp"
#include "tools/timer.hpp"
#include "xmuknn.h"

#define NUM_GPU 3
#define N_SAMPLE 20000000

using namespace std;
using namespace xmuknn;



void ToTxtResult(const string &kgraph_path, const string &out_path) {
  NNDElement *result_graph;
  int num, dim;
  FileTool::ReadBinaryVecs(kgraph_path, &result_graph, &num, &dim);

  int *result_index_graph = new int[num * dim];
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      result_index_graph[i * dim + j] = result_graph[i * dim + j].label();
    }
  }
  FileTool::WriteTxtVecs(out_path, result_index_graph, num, dim);
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      result_index_graph[i * dim + j] = result_graph[i * dim + j].distance();
    }
  }
  FileTool::WriteTxtVecs(out_path + ".txt", result_index_graph, num, dim);
  delete[] result_graph;
  delete[] result_index_graph;
}


__global__ void InitiateVecs(float *vec_dev,float *vectors,int vecs_dim,int offset=0){

    int list_id = blockIdx.x;
    int tx = threadIdx.x;
    

    vec_dev[list_id * vecs_dim + tx] = vectors[(list_id+offset) * vecs_dim + tx];

}



void  NNSPlit(NNDElement **knngraph_dev,float **vectors_dev,int vectors_size,int vecs_dim,int gpu_id){



    cudaSetDevice (gpu_id);


    printf("Iniciando a construcao do grafo para o Grafo %d processado pela GPU %d -> (%d,%d)\n",gpu_id,gpu_id,vectors_size,vecs_dim);

    gpuknn::NNDescent(knngraph_dev, *vectors_dev, vectors_size,
                vecs_dim,6,true);

    printf("\nConstrucao do grafo para o Grafo %d finalizado pela GPU %d -> (%d,%d) \n",gpu_id,gpu_id ,vectors_size,vecs_dim);

    return;
    }


void TestCUDAMerge() {

    //Define os arquvios de entrada e saída
    string out_path = "/nndescent/GPU_KNNG/results/sift1m_knng_k30.kgraph";

    string base_path = "/nndescent/GPU_KNNG/data/artificial/SK-20M_data.txt";

    //Variáveis que irão armazenar os dados de entrada
    float *vectors;
    int vecs_size, vecs_dim;

    // Unified memory para os vetores de entrada
    cudaMallocManaged(&vectors,(size_t)N_SAMPLE  * 12 * sizeof(float) );

    //Leitura dos vetores de entrada
    FileTool::ReadTxtVecs(base_path,&vectors,&vecs_size,&vecs_dim);


    //Define o número de shards, nesta implementação o número de shards deve ser no máximo o nro de GPUS
    int shards_num = 3;
    int vectors_size[shards_num];

    //Define o tamanho de cada shard
    for (int i=0;i<shards_num;i++){

        if (i != shards_num-1){ vectors_size[i] = vecs_size / shards_num;}

        else{
            vectors_size[i] = vecs_size;
            for (int j=0;j<i;j++){
            vectors_size[i] -= vectors_size[j];
            }
        }
    }

    // Cria o vetor que irá armazenar os dados de cada shard
    float *vectors_dev[shards_num];
    NNDElement *knngraph_dev[shards_num];

    printf("Dividindo os dados em shards...\n");


    float *merged_vecs;

    cudaMallocManaged(&merged_vecs, (size_t)(vectors_size[0] + vectors_size[1])* vecs_dim * sizeof(float));


    for (int i=0;i<shards_num;i++){
        cudaMallocManaged(&vectors_dev[i], (size_t)vectors_size[i] * vecs_dim * sizeof(float));
        cudaMallocManaged(&knngraph_dev[i],(size_t)vectors_size[i] * NEIGHB_NUM_PER_LIST * sizeof(NNDElement));
    }


    InitiateVecs<<< vectors_size[0],vecs_dim >>>(vectors_dev[0],vectors,vecs_dim,0);

    InitiateVecs<<< vectors_size[1],vecs_dim >>>(vectors_dev[1],vectors,vecs_dim,vectors_size[0]);

    InitiateVecs<<< vectors_size[2],vecs_dim >>>(vectors_dev[2],vectors,vecs_dim,vectors_size[0] + vectors_size[1]);

    InitiateVecs<<<(vectors_size[0] + vectors_size[1]),vecs_dim>>>(merged_vecs,vectors,vecs_dim,0);


    cudaDeviceSynchronize();

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }


    int soma = vectors_size[0];

    NNDElement *knngraph_merged_dev[shards_num-1];


    /*for (int i=0;i<shards_num-1;i++){
        soma += vectors_size[i+1];
        cudaMallocManaged(&knngraph_merged_dev[i], (size_t)soma *NEIGHB_NUM_PER_LIST *
                                sizeof(NNDElement));
    }*/
    
    printf("NNDescent iniciando\n");
    sleep(5);

    thread myThreads[shards_num];

    Timer merge_timer;
    merge_timer.start();



    for (int s=0;s<shards_num;s++)
        myThreads[s] = std::thread(NNSPlit,&knngraph_dev[s],&vectors_dev[s],vectors_size[s],vecs_dim,s);

    for (int s=0; s<shards_num; s++){
    myThreads[s].join();
    }


    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }
    printf("Que otimo, todos os sub-grafos foram gerados com exito\n");


    


  
    printf("Vamos para o merge\n");


    cudaSetDevice(1);

    
    gpuknn::KNNMerge(&knngraph_merged_dev[0], vectors_dev[0], vectors_size[0],
                   knngraph_dev[0], vectors_dev[1], vectors_size[1],
                   knngraph_dev[1], true);
    
    printf("OLAAAAAAAAAAAAAAAAAAAAA");

    gpuknn::KNNMerge(&knngraph_merged_dev[1], merged_vecs, vectors_size[0]*2,
                   knngraph_merged_dev[0], vectors_dev[2], vectors_size[2],
                   knngraph_dev[2], true);
    

    
    cerr << "Total cost: " << merge_timer.end() << endl;


    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }

    printf("Finalizado namoralzinha\n");







  std::ofstream out(out_path);


  if (!out.is_open()) {
    cerr << "Output file is not opened!" << endl;
    return;
  }

  else{printf("Arquivo de saida aberto corretamente\n");}

  NNDElement *result_graph_host;


  ToHostKNNGraph(&result_graph_host, knngraph_merged_dev[1],
                vectors_size[0] + vectors_size[1] + vectors_size[2] , NEIGHB_NUM_PER_LIST);

 FileTool::WriteBinaryVecs(out_path, result_graph_host,
                            vectors_size[0] + vectors_size[1] + vectors_size[2] ,
                            NEIGHB_NUM_PER_LIST);



  return;

}


void CheckKNNGraph() {
  NNDElement *knn_graph;
  int num, k;
  FileTool::ReadBinaryVecs("/nndescent/GPU_KNNG/results/sift1m_knng_k30.kgraph",
                           &knn_graph, &num, &k);
  cout << num << " " << k << endl;
  int id = 0;
  
  while (cin >> id) {
    printf("%d - ",id);
    for (int i = 0; i < k; i++) {
      printf("(%d, %f) ", knn_graph[id * k + i].label(),
             knn_graph[id * k + i].distance());
    }
    puts("");
  }
  delete[] knn_graph;
}



int main() {




  TestCUDAMerge();

  string base_path = "/nndescent/GPU_KNNG/results/sift1m_knng_k30.kgraph";


  printf("EScrevendo a saida em um arquivo .txt\n");
  ToTxtResult(base_path,base_path+ ".txt");
  printf("Saida escrita com sucesso, terminando o programa...\n");


}
