#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "hash.h"

//Shared memory where hash will be stored
extern __shared__ unsigned int words[];

__constant__ unsigned int target_hash[5]; 
__constant__ uint32_t r_cuda[64];
__constant__ uint32_t k_cuda[64];
__constant__ unsigned char target_salt[51];


// Note: All variables are unsigned 32 bit and wrap modulo 2^32 when calculating
// r specifies the per-round shift amounts
uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
					5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
					4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
					6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

// Use binary integer part of the sines of integers (in radians) as constants// Initialize variables:
//variaveis que precisarão ser declaradas no device
uint32_t k[] = {
		0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
		0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
		0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
		0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
		0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
		0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
		0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
		0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
		0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
		0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
		0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
		0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
		0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
		0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
		0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
		0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};


__device__ unsigned int *format_shared_memory(unsigned int thread_id, unsigned int *memory) {
  unsigned int *shared_memory;
  unsigned int *global_memory;
  int x;

	// we need to get a pointer to our shared memory portion
	shared_memory = &words[threadIdx.x * 16];
	global_memory = &memory[thread_id * 16];
	for(x=0; x < 16; x++) {
		shared_memory[x] = global_memory[x];
	}
	return shared_memory;

}

__device__ void sha1_transform(SHA1_CTX *ctx,  uchar data[]) {  
   uint a,b,c,d,e,i,j,t,m[80]; 
      
   for (i=0,j=0; i < 16; ++i, j += 4) 
      m[i] = (data[j] << 24) + (data[j+1] << 16) + (data[j+2] << 8) + (data[j+3]); 
   for ( ; i < 80; ++i) { 
      m[i] = (m[i-3] ^ m[i-8] ^ m[i-14] ^ m[i-16]); 
      m[i] = (m[i] << 1) | (m[i] >> 31); 
   }  
   
   a = ctx->state[0]; 
   b = ctx->state[1]; 
   c = ctx->state[2]; 
   d = ctx->state[3]; 
   e = ctx->state[4]; 
   
   for (i=0; i < 20; ++i) { 
      t = LEFTROTATE(a,5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i]; 
      e = d; 
      d = c; 
      c = LEFTROTATE(b,30); 
      b = a; 
      a = t; 
   }  
   for ( ; i < 40; ++i) { 
      t = LEFTROTATE(a,5) + (b ^ c ^ d) + e + ctx->k[1] + m[i]; 
      e = d; 
      d = c; 
      c = LEFTROTATE(b,30); 
      b = a; 
      a = t; 
   }  
   for ( ; i < 60; ++i) { 
      t = LEFTROTATE(a,5) + ((b & c) ^ (b & d) ^ (c & d))  + e + ctx->k[2] + m[i]; 
      e = d; 
      d = c; 
      c = LEFTROTATE(b,30); 
      b = a; 
      a = t; 
   }  
   for ( ; i < 80; ++i) { 
      t = LEFTROTATE(a,5) + (b ^ c ^ d) + e + ctx->k[3] + m[i]; 
      e = d; 
      d = c; 
      c = LEFTROTATE(b,30); 
      b = a; 
      a = t; 
   }  
   
   ctx->state[0] += a; 
   ctx->state[1] += b; 
   ctx->state[2] += c; 
   ctx->state[3] += d; 
   ctx->state[4] += e; 
}  

__device__ void sha1_init(SHA1_CTX *ctx) 
{  
   ctx->datalen = 0; 
   ctx->bitlen[0] = 0; 
   ctx->bitlen[1] = 0; 
   ctx->state[0] = 0x67452301; 
   ctx->state[1] = 0xEFCDAB89; 
   ctx->state[2] = 0x98BADCFE; 
   ctx->state[3] = 0x10325476; 
   ctx->state[4] = 0xc3d2e1f0; 
   ctx->k[0] = 0x5a827999; 
   ctx->k[1] = 0x6ed9eba1; 
   ctx->k[2] = 0x8f1bbcdc; 
   ctx->k[3] = 0xca62c1d6; 
}  

__device__ void sha1_update(SHA1_CTX *ctx, uchar data[], uint len) 
{  
   uint i;
   
   for (i=0; i < len; ++i) { 
      ctx->data[ctx->datalen] = data[i]; 
      ctx->datalen++; 
      if (ctx->datalen == 64) { 
         sha1_transform(ctx,ctx->data); 
         DBL_INT_ADD(ctx->bitlen[0],ctx->bitlen[1],512); 
         ctx->datalen = 0; 
      }  
   }  
}  

__device__ void sha1_final(SHA1_CTX *ctx, uchar hash[]) 
{  
   uint i; 
   
   i = ctx->datalen; 
   
   // Pad whatever data is left in the buffer. 
   if (ctx->datalen < 56) { 
      ctx->data[i++] = 0x80; 
      while (i < 56) 
         ctx->data[i++] = 0x00; 
   }  
   else { 
      ctx->data[i++] = 0x80; 
      while (i < 64) 
         ctx->data[i++] = 0x00; 
      sha1_transform(ctx, ctx->data); 
      memset(ctx->data,0,56); 
   }  
   
   // Append to the padding the total message's length in bits and transform. 
   DBL_INT_ADD(ctx->bitlen[0],ctx->bitlen[1],8 * ctx->datalen); 
   ctx->data[63] = ctx->bitlen[0]; 
   ctx->data[62] = ctx->bitlen[0] >> 8; 
   ctx->data[61] = ctx->bitlen[0] >> 16; 
   ctx->data[60] = ctx->bitlen[0] >> 24; 
   ctx->data[59] = ctx->bitlen[1]; 
   ctx->data[58] = ctx->bitlen[1] >> 8; 
   ctx->data[57] = ctx->bitlen[1] >> 16;  
   ctx->data[56] = ctx->bitlen[1] >> 24; 
   sha1_transform(ctx, ctx->data); 
   
}  


__device__ void hash_cuda(uint32_t * msg, uint32_t *hash, SHA1_CTX *ctx) {
    unsigned char buf[83];
    int i,j, k;
    uint32_t mask;

	uint32_t h0 = 0x67452301;
	uint32_t h1 = 0xefcdab89;
	uint32_t h2 = 0x98badcfe;
	uint32_t h3 = 0x10325476;


	// Pre-processing: adding a single 1 bit
	//append "1" bit to message    
	/* Notice: the input bytes are considered as bits strings,
	   where the first bit is the most significant bit of the byte.[37] */
 
	// Pre-processing: padding with zeros
	//append "0" bit until message length in bit ≡ 448 (mod 512)
	//append length mod (2 pow 64) to message
	// Process the message in successive 512-bit chunks:
	//for each 512-bit chunk of message:

	//MD5 execution
	int offset;
	for(offset=0; offset<56; offset += (512/8)) {
 
		// break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
		uint32_t *w = (uint32_t *) (msg);
 
		// Initialize hash value for this chunk:
		uint32_t a = h0;
		uint32_t b = h1;
		uint32_t c = h2;
		uint32_t d = h3;
 
		// Main loop:
		uint32_t i;
		for(i = 0; i<64; i++) {
  
			uint32_t f, g;
 
			 if (i < 16) {
				f = (b & c) | ((~b) & d);
				g = i;
			} else if (i < 32) {
				f = (d & b) | ((~d) & c);
				g = (5*i + 1) % 16;
			} else if (i < 48) {
				f = b ^ c ^ d;
				g = (3*i + 5) % 16;          
			} else {
				f = c ^ (b | (~d));
				g = (7*i) % 16;
			}

			uint32_t temp = d;
			d = c;
			c = b;
			b = b + LEFTROTATE((a + f + k_cuda[i] + w[g]), r_cuda[i]);
			a = temp;
		}
		// Add this chunk's hash to result so far:
		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;
	}
	hash[0] = h0;
	hash[1] = h1;
	hash[2] = h2;
	hash[3] = h3;
	//manipulating the hash hex to string
    k = 31;
    for(i = 3;i >= 0 ; i-- ){
 		mask = 0x0f000000;
    	for (j = 28; j >= 4; j-=8){
            buf[k] = ((hash[i] & mask ) >> (j-4) )+0x30 > 0x39? 
            	((hash[i] & mask ) >> (j-4) ) + 0x57 : ((hash[i] & mask ) >> (j-4) ) + 0x30 ;
  			mask <<= 4;
  			
  			buf[k-1] = ((hash[i] & mask ) >> j )+0x30 > 0x39?
  				((hash[i] & mask ) >> j ) + 0x57 : ((hash[i] & mask ) >> j ) + 0x30 ;
			mask >>= 12;
  			k-=2;
      	}
    }
    //applying the salt to the MD5 hash
    for (i = 32,j=0; i < 82; i++, j++) {
    	buf[i] = target_salt[j];
    }
	buf[82] = 0;
   	sha1_init(*&ctx); 
   	sha1_update(*&ctx,buf,82); 
   	sha1_final(*&ctx,buf);
}
__global__ void cuda_calculate(void *memory, struct device_stats *stats) {
	unsigned int id;
	unsigned int *shared_memory;
	uint hash[4];
	SHA1_CTX ctx;
	int x;
	
	id = (blockIdx.x * blockDim.x) + threadIdx.x;		// get our thread unique ID in this run

	shared_memory = format_shared_memory(id, (unsigned int *)memory);

	hash_cuda(shared_memory,hash,&ctx);
	
	if (ctx.state[0] == target_hash[0] && ctx.state[1] == target_hash[1] && ctx.state[2] == target_hash[2] && ctx.state[3] == target_hash[3] && ctx.state[4] == target_hash[4]) {
		// !! WE HAVE A MATCH !!
		stats->hash_found = 1;
		for(x=0; x<64; x++) {
			// copy the matched word accross
			stats->word[x] = *(char *)((char *)shared_memory + x);
		}
	}
}

extern "C" void init_constants(struct cuda_device *device){
	 // put our target hash into the GPU constant memory as this will not change (and we can't spare shared memory for speed)
	
	if (cudaMemcpyToSymbol(target_salt ,device->salt, 50, 0, cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
          	printf("Error initalizing constants\n");
          	return;
    }
	if (cudaMemcpyToSymbol(target_hash, device->target_hash, 24, 0, cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
			printf("Error target hash constant\n");
			return;
	}

	if (cudaMemcpyToSymbol(r_cuda, r, sizeof(r), 0, cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
			printf("Error to copy R's constants\n");
			return;
	}
	
	if (cudaMemcpyToSymbol(k_cuda, k, sizeof(k), 0, cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
		printf("Error to copy K's constants\n");
		return;
	}
}

extern "C" void calculate_hash(struct cuda_device *device) {
	
	//now we need to transfer the MD5 hashes to the graphics card for preperation
	if (cudaMemcpy(device->device_global_memory, device->host_memory, device->device_global_memory_len, cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
		printf("Error Copying Words to GPU\n");
		return;
	}

	cuda_calculate <<< device->max_blocks, device->max_threads, device->shared_memory >>> (device->device_global_memory,
	 (struct device_stats *)device->device_stats_memory);

}
