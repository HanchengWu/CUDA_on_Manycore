/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "phi_template.h"

#define min( a, b )     a > b ? b : a
#define ceilDiv( a, b )   ( a + b - 1 ) / b
#define print( x )      printf( #x ": %lu\n", (unsigned long) x )

#define DEBUG  0
#define VERIFY 0

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28 // character position of the latitude value in each record
#define OPEN 10000  // initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int *a);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
#include "euclid.h"
/**
* This program finds the k-nearest neighbors
**/
int main(int argc, char* argv[])
{
  int    i=0;
  float lat, lng;
  int quiet=0,timing=0,platform=0,device=0;

  std::vector<Record> records;
  std::vector<LatLong> locations;
  char filename[100];
  int resultsCount=10;
  int affinity;

    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device, &affinity)) {
      printUsage();
      return 0;
    }

    int numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;

    //Pointers to host memory
  float *distances;
  //Pointers to device memory
  LatLong *d_locations;
  float *d_distances;

  int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
  /**
  * Allocate memory on host and device
  */
  distances = (float *)malloc(sizeof(float) * numRecords);

  phiSetDevice(1);
  
  //cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
  phiMalloc(&d_locations,numRecords);
  //cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);
  phiMalloc(&d_distances,numRecords);

   /**
    * Transfer data from host to device
    */
    //cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);
  phiMemcpy(d_locations,(LatLong *)&locations[0],numRecords,CpuToPhi);

    /**
    * Execute kernel
    */

    FILE *fresult = fopen("result_copro.txt","a+");

  double s_time=gettime_ms();

    dim3 gridDim(112);
    euclid(affinity, 1.0, gridDim, threadsPerBlock, d_locations, d_distances, numRecords, lat, lng);

double e_time=gettime_ms();
    fprintf(fresult,"%-10d,%-18s,%-20lf\n",affinity,filename, e_time-s_time);
printf("\nExecution time:%lf.\n", e_time-s_time);
    fclose(fresult);

    //Copy data from device memory to host memory
    //cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );
    phiMemcpy(distances,d_distances,numRecords,PhiToCpu);
  // find the resultsCount least distances
    findLowest(records,distances,numRecords,resultsCount);

    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
    free(distances);
    //Free memory
  //cudaFree(d_locations);
    phiFree(d_locations);
  //cudaFree(d_distances);
    phiFree(d_distances);
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
  int    i=0;
  char dbname[64];
  int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
  while(!feof(flist)) {
    /**
    * Read in all records of length REC_LENGTH
    * If this is the last file in the filelist, then done
    * else open next file to be read next iteration
    */
    if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int *affinity){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
	    case 'a': //affinity
	      i++;
	      *affinity =atoi(argv[i]);
	      break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
