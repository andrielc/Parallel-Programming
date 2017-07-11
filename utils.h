#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <driver_types.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <bits/time.h>
#include <sys/time.h>
#include "hash.h"

#define ARG_WORDLIST 1

#define WORDS_TO_CACHE 10000
#define FILE_BUFFER 512

#define LF 1
#define CRLF 2

struct wordlist_file {
   int fd;
   int len;
   char *map;
   char **words;
   int current_offset;
   int delim;
};

double rtclock();
int read_wordlist(struct wordlist_file *file);
int process_wordlist(char *wordlist, struct wordlist_file *file);
char** split(char* word, char* separator);
char *md5_unpad(char *input);
char *md5_pad(char *input);
int get_cuda_device(struct cuda_device *device);
int calculate_cuda_params(struct cuda_device *device);
int _httoi(const char *value);
