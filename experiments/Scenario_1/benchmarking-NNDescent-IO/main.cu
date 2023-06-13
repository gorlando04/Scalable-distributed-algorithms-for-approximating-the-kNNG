#include <assert.h>
#include <unistd.h>

#include <string.h>


#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <vector>

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

using namespace std;
using namespace xmuknn;





void ToTxtResult(const string &kgraph_path, const string &out_path, long int n) {
  NNDElement *result_graph;
  int num, dim;
  FileTool::ReadBinaryVecs(kgraph_path, &result_graph, &num, &dim);

  num = n;

  int *result_index_graph = new int[n * dim];


  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      result_index_graph[i * dim + j] = result_graph[(i) * dim + j].label();
    }
  }
  FileTool::WriteTxtVecs(out_path, result_index_graph, num, dim);

  delete[] result_graph;
  delete[] result_index_graph;
}


void TestConstructLargeKNNGraph(int shards,int n) {


  string ref_path = "/nndescent/GPU_KNNG/data/vectors";

  string result_path = "/nndescent/GPU_KNNG/results/NNDescent-KNNG.kgraph";


  Timer timer;
  timer.start();

  int K = 32;
  
  GenLargeKNNGraph(ref_path, result_path, K,shards);


  printf("Time cost = %lf \n",timer.end());

  ToTxtResult(result_path,result_path + ".txt",n);





}




int main ( int argc, char *argv[]) {

/*

In the main function we have two steps that must be done. The first one is that the .txt file that contains the data must be turn into a .fvecs file,
to be possible to process it. To do this the variable PREPARE should be set to true; if we already have the .fvecs file, it must be set to false, to construct 
the kNNG
*/



  bool PREPARE = true;
  int shards = 30;
  long int n = 1000000;
  // ./main true|false
  if (argc == 2){
    printf("PREPARE SET TO %s. Running will be initiated.\n",argv[1]);
   
    if (strcmp(argv[1],"false") == 0)
        PREPARE = false;

  
  }
  
  // ./main true|false SHARDS
  else if (argc == 4){
    printf("PREPARE SET TO %s.\n",argv[1]);

    if (strcmp(argv[1],"false") == 0)
        PREPARE = false;

    shards = atoi(argv[2]);
    printf("SHARDS SET TO %d.\n",shards);

   n = atoi(argv[3]);
   printf("N SET TO %d.\n",n);

  
  }
  
  else
        printf("Standart settings\n");

 

  if (PREPARE){
    string base_path = "/nndescent/GPU_KNNG/data/artificial/SK_data.txt";
    float *vectors;
    long int vecs_size, vecs_dim;

    FileTool::ReadTxtVecs(base_path,&vectors,&vecs_size,&vecs_dim);

    printf("DIM = %d\n",vecs_dim);

    // Arquivo em que será criado o .fvecs que será utilizado
    string out_path = "/nndescent/GPU_KNNG/data/vectors.fvecs";

    // Escrita em binário
    FileTool::WriteBinaryVecs(out_path, vectors,
                              vecs_size,
                              vecs_dim);

  }


  else
    TestConstructLargeKNNGraph(shards,n);

 sleep(2);

 return 0;

}

