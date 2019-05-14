/*
 * Author: Lluis Jofre Cruanyes, 2019
 * 
 * Generate an input matrix for qr_mpi
 *
 * Run example: ./generate_matrix 10 4 2
 * 
 * output:
 * 
 *   1 1 1 1
 *   1 2 1 1
 *   1 1 3 1
 *   1 1 1 4
 *   1 1 1 1
 *   1 1 1 1
 *   1 2 1 1
 *   1 1 3 1
 *   1 1 1 4
 *   1 1 1 1
 *   
*/

#include <iostream>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>     
#include <math.h>

                    
int main( int argc, char* argv[] ) {

  // Read input arguments
  int rows    = atoi( argv[1] ); // Number of rows
  printf( "%d,\n", rows ); 
  int columns = atoi( argv[2] ); // Number of columns
  printf( "%d,\n", columns ); 
  int blocks  = atoi( argv[3] ); // Number of blocks
  printf( "%d,\n", blocks ); 

  // Generate matrix file
  for( int i = 0; i < rows; i++ ) {

    int i_blocked = i%( rows/blocks );

    for( int j = 0; j < columns; j++ ) {

      if( i_blocked == j ) {
        
        printf( "%f,", ( i_blocked + 1.0 ) );

      } else {
      
        printf( "%f,", 1.0 );

      }

    }

    printf( "\n" );

  }

}
