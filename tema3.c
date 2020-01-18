// Deaconu Andreea-Carina, 334CC

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "filters.h"

float clampPixel(float pxl) {
    if (pxl < 0)
        return 0;
    if (pxl > 255)
        return 255;
    return pxl;
}

unsigned char *apply_filter(char *filter, unsigned char *image, int line_size, int numchannels, int q) {
    int i, j, x, y, index;
    float tmp;
    unsigned char *partial_image;

    partial_image = calloc(q * line_size, sizeof(unsigned char));
    index = 0;

    if (!strcmp(filter, "smooth")) 
        for (i = 1; i <= q; i++) {
            index += numchannels;
            for (j = numchannels; j < line_size - numchannels; j++) {
                tmp = 0;
                for (x = 0; x < 3; x++)
                    for (y = 0; y < 3; y++)
                        tmp += K_smooth[2 - x][2 - y] / 9 * (float)image[(i - 1 + x) * line_size + j + (y - 1) * numchannels];
                partial_image[index++] = (unsigned char) clampPixel(tmp);
            }
            index += numchannels;
        }

    if (!strcmp(filter, "blur")) 
        for (i = 1; i <= q; i++) {
            index += numchannels;
            for (j = numchannels; j < line_size - numchannels; j++) {
                tmp = 0;
                for (x = 0; x < 3; x++)
                    for (y = 0; y < 3; y++)
                        tmp += K_blur[2 - x][2 - y] / 16 * (float)image[(i - 1 + x) * line_size + j + (y - 1) * numchannels];
                partial_image[index++] = (unsigned char) clampPixel(tmp);
            }
            index += numchannels;
        }

    if (!strcmp(filter, "sharpen")) 
        for (i = 1; i <= q; i++) {
            index += numchannels;
            for (j = numchannels; j < line_size - numchannels; j++) {
                tmp = 0;
                for (x = 0; x < 3; x++)
                    for (y = 0; y < 3; y++)
                        tmp += K_sharpen[2 - x][2 - y] / 3 * (float)image[(i - 1 + x) * line_size + j + (y - 1) * numchannels];
                partial_image[index++] = (unsigned char) clampPixel(tmp);
            }
            index += numchannels;
        }

    if (!strcmp(filter, "mean")) 
        for (i = 1; i <= q; i++) {
            index += numchannels;
            for (j = numchannels; j < line_size - numchannels; j++) {
                tmp = 0;
                for (x = 0; x < 3; x++)
                    for (y = 0; y < 3; y++)
                        tmp += K_mean[2 - x][2 - y] * (float)image[(i - 1 + x) * line_size + j + (y - 1) * numchannels];
                partial_image[index++] = (unsigned char) clampPixel(tmp);
            }
            index += numchannels;
        }

    if (!strcmp(filter, "emboss")) 
        for (i = 1; i <= q; i++) {
            index += numchannels;
            for (j = numchannels; j < line_size - numchannels; j++) {
                tmp = 0;
                for (x = 0; x < 3; x++)
                    for (y = 0; y < 3; y++)
                        tmp += K_emboss[2 - x][2 - y] * (float)image[(i - 1 + x) * line_size + j + (y - 1) * numchannels];
                partial_image[index++] = (unsigned char) clampPixel(tmp);
            }
            index += numchannels;
        }

    return partial_image;
}

int main (int argc, char **argv) {
    int rank, nr_proc, i, j, P, width, height, maxval, numchannels, total_size, line_size, q, start;
    unsigned char *image, *partial_image;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &nr_proc);

    // read image
    if (rank == 0) {
        FILE *fin;
        fin = fopen(argv[1], "r");
        fscanf(fin, "P%d\n# Created by GIMP version 2.10.14 PNM plug-in\n%d %d\n%d\n", &P, &width, &height, &maxval);
        numchannels = P == 5 ? 1 : 3; // black and white image or rgb one
        line_size = (width + 2) * numchannels;
        total_size = (height + 2) * line_size;
        image = calloc(total_size, sizeof(unsigned char));
        for (i = 1; i <= height; i++)
            fread(image + i * line_size + numchannels, sizeof(unsigned char), width * numchannels, fin); // "i * line_size" represents the line number, while "+ numchannels" skips the first pixel which was added to border the matrix with zeros
        fclose(fin);
    }

    // if filters were specified
    if (argc > 3) 
    {
        // sends picture info to the other processes
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numchannels, 1, MPI_INT, 0, MPI_COMM_WORLD);

        q = height / nr_proc; // calculate nr of lines to be processed by one process
        if (rank == nr_proc - 1 && height % nr_proc != 0) // if division is not exact
                q += height % nr_proc; // the last process gets the rest
        line_size = (width + 2) * numchannels;
        partial_image = calloc((q + 2) * line_size, sizeof(unsigned char));

        for (j = 3; j < argc; j++) 
        {
            if (rank == 0) 
            {
                start = 1;
                for (i = 1; i < nr_proc; i++) 
                {
                    start += q;
                    if (i == nr_proc - 1 && height % nr_proc != 0) // sends to each process its part of the image
                        MPI_Send(image + (start - 1) * line_size, (q + height % nr_proc + 2) * line_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
                    else
                        MPI_Send(image + (start - 1) * line_size, (q + 2) * line_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
                } 
                memcpy(image + line_size, apply_filter(argv[j], image, line_size, numchannels, q), q * line_size); // applies the filter to its part of the image
                start = 1;
                for (i = 1; i < nr_proc; i++) 
                {
                    start += q;
                    if (i == nr_proc - 1 && height % nr_proc != 0) // receives from each process the modified image
                        MPI_Recv(image + start * line_size, (q + height % nr_proc) * line_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    else
                        MPI_Recv(image + start * line_size, q * line_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            else 
            {   
                MPI_Recv(partial_image, (q + 2) * line_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // each process receives its partial image
                MPI_Send(apply_filter(argv[j], partial_image, line_size, numchannels, q), q * line_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD); // sends back the modified partial image
            }
        }
    }

    // write image
    if (rank == 0) {
        FILE *fout;
        char s[100];
        fout = fopen(argv[2], "w");
        sprintf(s, "P%d\n%d %d\n%d\n", P, width, height, maxval);
        fwrite(s, sizeof(unsigned char), strlen(s), fout);
        for (i = 1; i <= height; i++)
            fwrite(image + i * line_size + numchannels, sizeof(unsigned char), width * numchannels, fout);
        fclose(fout);
    }

    MPI_Finalize();
    return 0;
}
