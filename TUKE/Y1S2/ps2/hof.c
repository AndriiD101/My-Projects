#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "hof.h"

#define MAX_NAME_LENGTH 50


void swap(struct player *a, struct player *b) {
    struct player temp = *a;
    *a = *b;
    *b = temp;
}

int load(struct player list[]){
    int position=0;
    char content_of_file[1000];
    FILE *file = fopen(HOF_FILE, "r");
    if (file == NULL) return -1;

    while(position < 10 && (fgets(content_of_file, sizeof(content_of_file), file))){

        struct player player;
        char * name = strtok(content_of_file, " ");
        char * score = strtok(NULL, " ");

        if(name != NULL && score != NULL) {
            strncpy(player.name, name, sizeof(player.name)-1);
            player.name[sizeof(player.name)-1] = '\0';
            player.score = atoi(score);

            list[position] = player;
            position++;
        }
    }

    bool swapped = false;
    for (int i = 0; i < position-1; i++){
        swapped = false;
        for(int j = 0; j<position-i-1; j++){
            if(list[j].score < list[j+1].score){
                swap(&list[j], &list[j+1]);
                swapped = true;
            }
        }
        if(!swapped){
            break;
        }
    }

    fclose(file);
    return position;
}

bool save(const struct player list[], const int size){
    struct player *temp_list = calloc((size_t)size, sizeof(struct player));
    for(int i = 0; i < size; i++){
        temp_list[i] = list[i];
    }
    FILE *file;
    file = fopen(HOF_FILE, "w");
    if (file == NULL){
        return false;
    }
    bool swapped = false;
    for (int i = 0; i < size-1; i++){
        swapped = false;
        for(int j = 0; j<size-i-1; j++){
            if(temp_list[j].score < temp_list[j+1].score){
                swap(&temp_list[j], &temp_list[j+1]);
                swapped = true;
            }
        }
        if(!swapped){
            break;
        }
    }
    fclose(file);
    return true;
}

bool add_player(struct player list[], int* size, const struct player player){
    if(list && size && player.name && *size>0 && *size<=10) {
        for(int i = 0; i < *size; i++){
            if(list[i].score <= player.score){
                list[i] = player;
                if(*size<10) *size+=1;
                return true;
            }
        }
        if(*size<10){
            list[*size] = player;
            *size+=1;
            return true;
        }
    }
    return false;
}


// int main() {
//     // Sample player list
//     struct player array[10];
//     printf("%d\n", load(array));
//     return 0;
// }