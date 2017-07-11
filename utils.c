/*
 * Utils.c
 * Utils file include useful functions for the program execution
 */
#include "utils.h"

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

char *md5_unpad(char *input) {
  static char md5_unpadded[FILE_BUFFER];
  unsigned int orig_length;
  int x;

  if (input == NULL) {
    return NULL;
  }

  memset(md5_unpadded, 0, sizeof(md5_unpadded));

  orig_length = (*((unsigned int *)input + 14) / 8);

  strncpy(md5_unpadded, input, orig_length);

  return md5_unpadded;
}


char *md5_pad(char *input){
    uint8_t *msg;

    int initial_len = strlen(input);
    int new_len = 56;

    msg = (uint8_t *)calloc(new_len+64,1); // also appends "0" bits 
                                   // (we alloc also 64 extra bytes...)
    memcpy(msg, input, initial_len);
    msg[initial_len] = 128; // write the "1" bit
 
    uint32_t bits_len = 8*initial_len; // note, we append the len
    memcpy(msg + new_len, &bits_len, 4);           // in bits at the end of the buffer
   return (char *)msg; 
}

int get_cuda_device(struct cuda_device *device) {
  int device_count;

  if (cudaGetDeviceCount(&device_count) != CUDA_SUCCESS) {
    // cuda not supported
    return -1;
  }

  while(device_count >= 0) {
    if (cudaGetDeviceProperties(&device->prop, device_count) == CUDA_SUCCESS) {
      // we have found our device
      device->device_id = device_count;
      return device_count;
    }

    device_count--;
  }

  return -1;
}

int calculate_cuda_params(struct cuda_device *device) {
  uint32_t  max_threads;
  uint32_t  max_blocks;
  uint32_t shared_memory;

  max_threads = 1024; 
  shared_memory = 2048;
  // calculate the most threads that we can support optimally
  while (max_threads > 0 && (shared_memory / max_threads) < 64) { max_threads--; }

  // now we spread our threads across blocks
  max_blocks = 32;

  device->max_threads = max_threads;    // most threads we support
  device->max_blocks = max_blocks;    // most blocks we support
  device->shared_memory = shared_memory;    // shared memory required

  // now we need to have (device.max_threads * device.max_blocks) number of words in memory for the graphics card
  device->device_global_memory_len = (device->max_threads * device->max_blocks) * 64;
  return 1;
}

struct CHexMap {
  char chr;
  int value;
};

#define true 1
#define false 0

#define HexMapL 22

int _httoi(const char *value) {
  struct CHexMap HexMap[HexMapL] = {
    {'0', 0}, {'1', 1},
    {'2', 2}, {'3', 3},
    {'4', 4}, {'5', 5},
    {'6', 6}, {'7', 7},
    {'8', 8}, {'9', 9},
    {'A', 10}, {'B', 11},
    {'C', 12}, {'D', 13},
    {'E', 14}, {'F', 15},
    {'a', 10}, {'b', 11},
    {'c', 12}, {'d', 13},
    {'e', 14}, {'f', 15},
  };
  int i;

  char *mstr = strdup(value);
  char *s = mstr;
  int result = 0;
  int found = false;

  if (*s == '0' && *(s + 1) == 'X') {
    s += 2;
  }

  int firsttime = true;

  while (*s != '\0') {
    for (i = 0; i < HexMapL; i++) {

      if (*s == HexMap[i].chr) {

        if (!firsttime) {
          result <<= 4;
        }
        
        result |= HexMap[i].value;
        found = true;
        break;
      }
    }

    if (!found) {
      break;
    }

    s++;
    firsttime = false;
  }

  free(mstr);
  return result;
 }


int read_wordlist(struct wordlist_file *file) {
   unsigned int x;
   char delim;
   unsigned int start, end;
   int wordcount;

   //Free any previous words before allocating new ones
   for(x=0; x < WORDS_TO_CACHE; x++) {
      if (file->words[x] != (void *)0) {
         free(file->words[x]);
      }
   }

   //Clear all previous memory allocs which are now invalid
   memset(file->words, 0, (WORDS_TO_CACHE + 1) * sizeof(char *));
   wordcount = 0;

   //Need to read from the file and find words
   switch(file->delim) {
      case CRLF:
         delim = '\r';
      break;
      case LF:
         delim = '\n';
      break;
   }

   for(start=x=file->current_offset; x < file->len && wordcount < WORDS_TO_CACHE; x++) {
      if (file->map[x] == delim) {
         //Mark the end of the word
         end = x;
         file->words[wordcount] = (char *)malloc((end - start) + 1);
         memset(file->words[wordcount], 0, (end-start) + 1);
         memcpy(file->words[wordcount], file->map + start, end - start);
         //Increment wordcount
         wordcount++;

         start = end + file->delim;
      }
   }

   file->current_offset = start;

   return wordcount;
}

// responsible for inputting words from the word list
int process_wordlist(char *wordlist, struct wordlist_file *file) {
   int file_offset, word_offset;
   static char *words[WORDS_TO_CACHE + 1];      // the extra '1' is for the NULL char* signaling the end of the list
   struct stat stat;
   int x;

   memset(file, 0, sizeof(struct wordlist_file));

   file->fd = (int) open(wordlist, O_RDONLY);

   if (file->fd == -1) {
      // error opening wordlist
      return -1;
   }

   if (fstat(file->fd, &stat) == -1) {
      // error statting the wordlist file
      return -1;
   }

   file->len = stat.st_size;
   file->map = (char *)mmap(NULL, file->len, PROT_READ, MAP_SHARED, file->fd, 0);

   // now we must detect the deliminator of the line (\r\n or just \n)
   
   for(x=0; x < stat.st_size; x++) {
      if (file->map[x] == '\n') {
         if (x > 1 && file->map[x-1] == '\r') {
            // the line ends with '\r\n'
            file->delim = 2;
         } else {
            // the line ends with just '\n'
            file->delim = 1;
         }

         break;
      }
   }

   if (file->delim == 0) {
      // no deliminator
      printf("Words do not end with \'\\r\\n\' or \'\\n\'\n");
      return -1;
   }

   // set our memory to 0x000000000
   memset(words, 0, sizeof(words));
   file->words = words;

   return 1;
}

//split db line in 2 words: sha1 hash (32 Hex chars) and salt (50 Hex chars)
char** split(char* word, char* separator){

  int i = 0;
  char* token;
  char* string;
  char** words = (char**)malloc(2*sizeof(char));
  string = strdup(word);

  if (string != NULL) {

    while ((token = strsep(&string, separator)) != NULL)
    {
      *(words + i) = (char*)malloc(60*sizeof(char));
      words[i] = token;
      i++;
    }
  }

  return words;
}


