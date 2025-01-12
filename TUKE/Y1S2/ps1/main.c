#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "bmp.h"
#include "playfair.h"


int main(){
 char *encrypted, *decrypted;

 // even length of string
encrypted = playfair_encrypt("please", "Taxxxiii");
printf("%s\n", encrypted);
// "Taxxxiii" --> "TAXXXIIXIX"
// RS EE VJ JV JV
decrypted = playfair_decrypt("please", encrypted);
printf("%s", decrypted);
// TAXXXIIXIX
free(encrypted);
free(decrypted);
 printf("\n");
 // HELXLOVORLDX
 free(encrypted);
 free(decrypted);
 unsigned char *bmpE;
 char *bmpD;
 bmpE = bmp_encrypt("TUKE","student djafldjslfajds");
 for(int i = 0; bmpE[i];i++){
  printf("%x ",bmpE[i]);
 }
 printf("\n");
 bmpD = bmp_decrypt("TUKE",bmpE);
 printf("%s\n",bmpD);
 free(bmpE);
 free(bmpD);


    return 0;
}
