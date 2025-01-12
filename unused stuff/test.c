#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//#define COUNT 10


typedef struct treenode{
    int root;
    struct treenode *left;
    struct treenode *right;
} treenode;

treenode *createnode(int value){
    treenode *node = malloc(sizeof(treenode));
    if(node!=NULL){
        node->left = NULL;
        node->right = NULL;
        node->root = value;
    }
    return node;
}

treenode *put_values_in_tree(treenode *node, int value){
    if(node==NULL) return createnode(value);

    if(value<node->root){
        node->left = put_values_in_tree(node->left, value);
    }
    else{
        node->right = put_values_in_tree(node->right, value);
    }
    return node;
}

int check_is_same(treenode* p, treenode* q) {
    if (p == NULL && q == NULL) return 1;
    if (p == NULL || q == NULL) return 0;
    return check_is_same(p->left, q->left) && check_is_same(p->right, q->right);
}

int count_unique_trees(treenode** array_of_trees, int n) {
    int count = 0;
    int* visited = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (visited[i] == 0) {
            count++;
            for (int j = i + 1; j < n; j++) {
                if (check_is_same(array_of_trees[i], array_of_trees[j])) {
                    visited[j] = 1;
                }
            }
        }
    }
    free(visited);
    return count;
}

void deallocate(treenode *root){
    if(root != NULL){
    deallocate(root->left);
    deallocate(root->right);
    free(root);
    }
}

int main(){
    int n, k;
    // printf("n k: ");
    scanf("%d %d", &n, &k);
    if(1>n || n>50) return -1;
    if(1>k || k>20) return -1;
    // treenode **array_of_trees=NULL;
    // array_of_trees = (treenode**) malloc(n * sizeof(struct treenode*));
    int input_tree_values[n][k];
    for (int i = 0; i < n; i++) {
        treenode *root = NULL;
        for(int j = 0; j < k; j++) {
            scanf("%d", &input_tree_values[i][j]);
            if(input_tree_values[i][j]<1 || input_tree_values[i][j] > pow(10,6)) return -1;
        }
        root = createnode(input_tree_values[i][0]);
        for(int cols = 1; cols < k; cols++) { 
                root = put_values_in_tree(root, input_tree_values[i][cols]);
        }
        // array_of_trees[i] = root;
    }

    // printf("%d\n", is_unique_trees(array_of_trees, n));

    // for(int i = 0; i<n; i++){
        // deallocate(array_of_trees[i]);
        // array_of_trees[i] = NULL;
    // }
    // free(array_of_trees);
    return 0;
}
