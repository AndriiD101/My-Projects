#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define COUNT 10

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
    else if(value>node->root){
        node->right = put_values_in_tree(node->right, value);
    }
    return node;
}

// void print2DUtil(treenode* root, int space)
// {
//     if (root == NULL)
//         return;
 
//     space += COUNT;
 
//     print2DUtil(root->right, space);
 
//     printf("\n");
//     for (int i = COUNT; i < space; i++)
//         printf(" ");
//     printf("%d\n", root->root);
 
//     print2DUtil(root->left, space);
// }
 
// void print2D(treenode* root)
// {
//     print2DUtil(root, 0);
// }

int isSameStructure(treenode* p, treenode* q) {
    if (p == NULL && q == NULL) return 1;
    if (p == NULL || q == NULL) return 0;
    return isSameStructure(p->left, q->left) && isSameStructure(p->right, q->right);
}

int countUniqueTrees(treenode** array_of_trees, int n) {
    int count = 0;
    int* visited = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (visited[i] == 0) {
            count++;
            for (int j = i + 1; j < n; j++) {
                if (isSameStructure(array_of_trees[i], array_of_trees[j])) {
                    visited[j] = 1;
                }
            }
        }
    }
    free(visited);
    return count;
}

int main(){
    int n, k, value;
    // printf("n k: ");
    scanf("%d %d", &n, &k);
    if(1>n || n>50) return -1;
    if(1>k || k>20) return -1;
    treenode **array_of_trees = malloc(n * sizeof(struct treenode*));
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
        array_of_trees[i] = root;
    }

    // printf("Binary trees (inorder traversal):\n");
    // for (int i = 0; i < n; ++i) {
    //     printf("Tree %d: ", i + 1);
    //     printf("\n");
    //     print2D(array_of_trees[i]);
    //     printf("\n");
    // }

    printf("%d\n", countUniqueTrees(array_of_trees, n));

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(array_of_trees[i]);
    }
    free(array_of_trees);

    return 0;
}
