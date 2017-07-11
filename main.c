#include "utils.c"

#define INF 9999999999
#define MAX_DB_HASH_CHECK INF //change to INF if it's needed to check the whole database

extern void calculate_hash(struct cuda_device *);
extern void init_constants(struct cuda_device *);

// This is the main function of the program.
// This function should receive 3 arguments: 1- Wordlist file name, 2- Database file name, 3- Output file name
// These 2 first argument files should be in the same directory folder of this program.
int main(int argc, char **argv) {

   uint32_t x, y, available_words_in_wordlist = 1, available_words_in_db = 1, current_words_in_wordlist;
   uint32_t current_words_in_db, flag_hash_found, current_words = 0, available_words = 1;
   char text_md5_with_salt[83], input_hash[4][9], db_word[92], db_hash[51], db_salt[51], **db_line;
   struct wordlist_file wordlist_file;
   unsigned char hash[20];
   const char *sha1_text, *md5_from_wordlist;
   FILE *db_file, *output_file;
   struct cuda_device device;

   uint32_t carregar = 0;	
   //check the number of arguments passed to main function
   if (argc != 4) {
      printf("Not a valid number of parameters!\n");
      return -1;
   }

   //open Database file that contains a SHA1 hash and a salt string as "SHA1:salt" format
   db_file = fopen(argv[2],"r");

   //open or create the output file that will receive the "CrackedPassword:salt" found of each Database hash
   output_file = fopen(argv[3],"w");

   //
   printf("\nThe input files are db_10.txt, db_50.txt and db_100.txt\n");

   //enable the benchmark data, as number of hash generated and the total time of program execution
   #ifdef BENCHMARK
   uint32_t counter = 0;
   uint32_t db_hash_check = 0;
   struct timeval start;
   gettimeofday(&start, NULL);
   #endif
   // we now need to calculate the optimal amount of threads to use for this card
   calculate_cuda_params(&device);

   //This loop is responsible to:
   // . Get each line of Database file (that corresponds to "SHA1:salt" format),
   // . Split the SHA1 hash and salt string
   // . Process and read the wordlist file
   // . Check if there is a word available in wordlist file
   // . Finish when there is no more line in Database file to process

   // first things first, we need to select our CUDA device
   
   if (get_cuda_device(&device) == -1) {
      printf("No Cuda Device Listed\n");
      return -1;
   }

   // allocate global memory for use on device
   if (cudaMalloc(&device.device_global_memory, device.device_global_memory_len) != CUDA_SUCCESS) {
      printf("Error allocating Global Memory\n");
      return -1;
   }
   
   // Allocate the 'stats' that will indicate if we are successful in cracking
   if (cudaMalloc(&device.device_stats_memory, sizeof(struct device_stats)) != CUDA_SUCCESS) {
      printf("Error allocating Device Status memory\n");
      return -1;
   }

   // Host memory that needed to be copied to the graphics card
   if ((device.host_memory = malloc(device.device_global_memory_len)) == NULL) {
         printf("Error allocating the Host Memory.\n");
         return -1;
    }

   while(fgets(db_word,sizeof(db_word), db_file) != NULL){

      //Disregard line if it has only '\n'
      if(strcmp(db_word,"\n")==0) 
         continue;
      
      //Split line of Database file in SHA1 string and salt string
      db_line = split(db_word, ":");
      strcpy(db_hash,db_line[0]);
      strcpy(db_salt,db_line[1]);
		
      //Split the input hash into 4 blocks
      memset(input_hash, 0, sizeof(input_hash));
      for(x=0; x < 5; x++) {
         strncpy(input_hash[x], db_hash + (x * 8), 8);
         device.target_hash[x] = _httoi(input_hash[x]);
      }

      // make sure the stats are clear on the device
      if (cudaMemset(device.device_stats_memory, 0, sizeof(struct device_stats)) != CUDA_SUCCESS) {
         printf("Error Clearing Stats on device\n");
         return -1;
      }
      strcpy(device.salt,db_salt);

      init_constants(&device);

      #ifdef BENCHMARK
      if(db_hash_check++ == MAX_DB_HASH_CHECK)
      break;
      #endif
      
      //Process and read the wordlist file
      process_wordlist(argv[ARG_WORDLIST], &wordlist_file);
      read_wordlist(&wordlist_file);

      available_words_in_wordlist = 1;
      current_words_in_wordlist = 0;
      flag_hash_found = 0;
	   //This loop is responsible to:
      // . Load a set of words to device global memory (that corresponds to a set of password strings)
      // . Run unique SHA1(MD5($password)+salt) function in each GPU's threads.
      // . Finish when there is no more line in Wordlist file to process OR if the SHA1 hashes are equals 
      while(available_words_in_wordlist) {  
         
         memset(device.host_memory, 0, device.device_global_memory_len);

         for( x = 0; x < (device.device_global_memory_len / 64) && wordlist_file.words[current_words] != (char *)0; x++, current_words++) {
         
         #ifdef BENCHMARK
            //counter of words
            counter++;
         #endif

            char *padded = md5_pad(wordlist_file.words[current_words]);
            memcpy(device.host_memory + (x * 64), padded , 64);
         }
         //Used to verifiy the needed to read more words to the buffer.
         if (wordlist_file.words[current_words] == (char *)0) {
            current_words = 0;
	        if (!read_wordlist(&wordlist_file)) {
               // No more words available
               available_words_in_wordlist = 0;
               // We continue as we want to flush the cache !
             }
	      }
         calculate_hash(&device);

         if (cudaMemcpy(&device.stats, device.device_stats_memory, sizeof(struct device_stats), cudaMemcpyDeviceToHost) != CUDA_SUCCESS) {
            printf("Error to copy Device Status from the GPU\n");
            return -1;
         }

         if (device.stats.hash_found == 1) {
            fprintf(output_file,"%s:%s\n",md5_unpad(device.stats.word) ,db_salt);
            flag_hash_found = 1;
            break;
         }
      }
      //If any word of wordlist generated the same hash of the current database line
      if(flag_hash_found == 0){
      	fprintf(output_file,"????????:%s\n", db_salt);
      }
   }
   

   #ifdef BENCHMARK 
   struct timeval  end;
   gettimeofday(&end, NULL);
   long long time = (end.tv_sec * (unsigned int)1e6 + end.tv_usec) - (start.tv_sec * (unsigned int)1e6 + start.tv_usec);
   printf("Hashes calculate %d hashes Total time %f seconds\n", counter, (float)((float)time / 1000.0) / 1000.0);
   printf("Words per second: %d\n", counter / (time / 1000) * 1000);
   #endif
	
   //Close input files
   fclose(output_file);
   fclose(db_file);
   //Free memories allocated to the GPU
   cudaFree(device.device_global_memory);
   cudaFree(device.device_stats_memory);

   return 0;
}
