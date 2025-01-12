#include <stdio.h>
#include <math.h>

#define MAX_HOLES 10000
#define MAX_SLICES 100
#define CHEESE_SIZE 100000. * 100000. * 100000.
#define M_PI 3.14159265358979323846

typedef struct {
    double R; 
    double X, Y, Z; 
} sphere_data;

double volum_of_cap_sphere(double R, double h) {
    // return (M_PI * h * h * (3 * R - h)) / 3.0;
    return M_PI * (1. / 3.) * h * h * (3. * R - h);
}

// calculate the volume of a segment a sphere intersecting a slice
double calculate_segment_volume(sphere_data sphere, double slice_bottom, double slice_top) {
    double volume = 0.0;
    double sphere_bottom = sphere.Z - sphere.R; // Z is the center of the sphere
    double sphere_top = sphere.Z + sphere.R;

    //if sphere in slice
    if (slice_top > sphere_bottom && slice_bottom < sphere_top) {
        double lowest_position = fmax(0, slice_bottom - sphere_bottom);
        double highest_position = fmin(sphere.R * 2, slice_top - sphere_bottom);
        volume = volum_of_cap_sphere(sphere.R, highest_position) - volum_of_cap_sphere(sphere.R, lowest_position);
    }
    return volume;
}

int main() {
    int M, S;
    scanf("%d %d", &M, &S); // M amount of holes, S amount of pices

    sphere_data holes[MAX_HOLES];
    for (int i = 0; i < M; i++) {
        scanf("%lf %lf %lf %lf", &holes[i].R, &holes[i].X, &holes[i].Y, &holes[i].Z);
        if(holes[i].R > 0 && holes[i].X < 0 && holes[i].Y < 0 && holes[i].Z < 0 && holes[i].X>MAX_HOLES && holes[i].Y>MAX_HOLES && holes[i].Z>MAX_HOLES) return -1;
    }

    double total_volume = CHEESE_SIZE; // Total volume without holes

    // Adjust the volume of cheese by subtracting the volume of the holes
    for (int i = 0; i < M; i++) {
        total_volume -= (4.0 / 3.0) * M_PI * pow(holes[i].R, 3);
    }

    // Calculate the target volume for each slice
    double targeted_volume = total_volume / S;
    // Calculate the thickness of each slice
    for (int i = 0; i < S; i++) {
        double volume_of_slice = 0.0;
        double slice_bottom = i * (100000 / S);
        double slice_top = (i + 1) * (100000 / S);

        volume_of_slice += CHEESE_SIZE / S; // Volume of the slice without holes

        for (int j = 0; j < M; j++) {
            volume_of_slice -= calculate_segment_volume(holes[j], slice_bottom, slice_top);
        }
        double result = (targeted_volume / volume_of_slice * (slice_top - slice_bottom)) / 1000;
        printf("%.9lf\n", result);
    }

    return 0;
}
