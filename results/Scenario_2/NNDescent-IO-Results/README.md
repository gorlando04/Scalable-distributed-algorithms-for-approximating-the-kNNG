# Results

We do not have a .csv here, because the NNDescentI/O experiments were different. So we stored the results in each Folder, and in the Table below we have a descriptions of the folders and the datasets inside it.

| Folder Name | Dataset | N_sample | Dimensions | Size (GB) |
| --- | --- | --- | --- | --- |
| Test0 | SK-100M-12d | 100.000.000 | 12 | 4.8 |
| Test1 | SK-150M-12d | 150.000.000 | 12 | 7.2 |
| Test2 | SK-200M-12d | 200.000.000 | 12 | 9.6 |
| Test3 | SK-250M-12d | 250.000.000 | 12 | 12 |
| Test4 | SK-300M-12d | 300.000.000 | 12 | 14.4 |
| Test5 | SK-350M-12d | 350.000.000 | 12 | 16.8 |
| Test6 | SK-400M-12d | 400.000.000 | 12 | 19.2 |


In each folder we have 3 files: Result.txt, Recall-Begin.txt  and Recall-Final.txt. Result.txt file stores the time spent for building the kNNG, and both Recall files, store the first 10.000 recall and then, the last 10.000 recall
