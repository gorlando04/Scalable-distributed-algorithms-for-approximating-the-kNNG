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


// Here we set N_SAMPLE 10.000 because we just want to check the result for the 10.000 first samples
#define N_SAMPLE 10000


void ToTxtResult(const string &kgraph_path, const string &out_path,bool end=false) {
  NNDElement *result_graph;
  int num, dim;
  FileTool::ReadBinaryVecs(kgraph_path, &result_graph, &num, &dim);
  long int offset = 0;

  if (end){
    offset = FileTool::GetFVecsNum("/nndescent/GPU_KNNG/data/vectors.fvecs") - N_SAMPLE;
  }
  printf("OFFSET = %d\n",offset);
  num = N_SAMPLE;

  int *result_index_graph = new int[num * dim];


  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      result_index_graph[i * dim + j] = result_graph[(i + offset) * dim + j].label();
    }
  }
  FileTool::WriteTxtVecs(out_path, result_index_graph, num, dim);

  delete[] result_graph;
  delete[] result_index_graph;
}


void TestConstructLargeKNNGraph(int shards) {


  string ref_path = "/nndescent/GPU_KNNG/data/vectors";

  string result_path = "/nndescent/GPU_KNNG/results/NNDescent-KNNG.kgraph";


  Timer timer;
  timer.start();

  int K = 32;
  
  GenLargeKNNGraph(ref_path, result_path, K,shards);


  printf("Time cost = %lf \n"timer.end());
  //Put the result into a txt file. BEGIN
  ToTxtResult(result_path,result_path + "-BEGIN.txt");

  //Put the result into a txt file. END
  ToTxtResult(result_path,result_path + "-FINAL.txt",true);




}




int main ( int argc, char *argv[]) {

/*

In the main function we have two steps that must be done. The first one is that the .txt file that contains the data must be turn into a .fvecs file,
to be possible to process it. To do this the variable PREPARE should be set to true; if we already have the .fvecs file, it must be set to false, to construct 
the kNNG
*/



  bool PREPARE = true;
  int shards = 30;

  // ./main true|false
  if (argc == 2){
    printf("PREPARE SET TO %s. Running will be initiated.\n",argv[1]);
   
    if (strcmp(argv[1],"false") == 0)
        PREPARE = false;

  
  }
  
  // ./main true|false SHARDS
  else if (argc == 3){
    printf("PREPARE SET TO %s.\n",argv[1]);

    if (strcmp(argv[1],"false") == 0)
        PREPARE = false;

    shards = atoi(argv[2]);
    printf("SHARDS SET TO %d.\n",shards);

  
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
    TestConstructLargeKNNGraph(shards);

 sleep(2);

 return 0;

}

