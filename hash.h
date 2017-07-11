#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
// Signed variables are for wimps 
#define uchar unsigned char 
#define uint unsigned int 

// DBL_INT_ADD treats two unsigned ints a and b as one 64-bit integer and adds c to it
#define DBL_INT_ADD(a,b,c) if (a > 0xffffffff - c) ++b; a += c; 

typedef struct { 
   uchar data[64]; 
   uint datalen; 
   uint bitlen[2]; 
   uint state[5]; 
   uint k[4]; 
} SHA1_CTX;

struct device_stats {
  unsigned char word[64];     // found word passed from GPU
  int hash_found;     // boolean if word is found
};

struct cuda_device {
  int device_id;
  struct cudaDeviceProp prop;
  unsigned char salt[51];
  int max_threads;
  int max_blocks;
  int shared_memory;

  void *device_global_memory;
  int device_global_memory_len;

  void *host_memory;

  void *device_stats_memory;
  struct device_stats stats;

  unsigned int target_hash[4];
};