/*
 * Author: Lluis Jofre Cruanyes, 2019
 *
 * Perform QR decomposition of input matrix   
 *   
 * Run example: mpiexec -n 2 ./program -file file_name [-silent]
 *
*/

#include "mpi.h"
#include <iostream>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "cTimer.h"      
#include "mkl.h"
                                                                    
using namespace std;


int main( int argc, char* argv[] ) { 

  FILE *file;
  arg::cTimer timer;
  bool verbose = true, definition;   
  int i, j, k, l, nr;
  int world_rank, world_size;
  int num_rows, num_columns, num_blocks;
  double *A;
  
  // Initialize MPI  
  MPI_Init( &argc, &argv );
 
  // Obtain world_rank and world_size 
  MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
  //cout << world_rank << endl;
  MPI_Comm_size( MPI_COMM_WORLD, &world_size );
  //cout << world_size << endl;

  // Read matrix input file and arguments
  for( k = 0; k < world_size; k++ ) {

    // One processor at a time
    if( world_rank == k ) {
    
      for( i = 0; i < argc; i++ ) {
      
        if( strcmp( argv[i], "-file" ) == 0 ) { 

          if( ( file = fopen( argv[++i], "r" ) ) == NULL ) {

            if( verbose ) printf( "Cannot open file.\n" );
            MPI_Finalize();

            return(0);

          } else {

            definition = false;

          }

        }
      
        if( strcmp( argv[i], "-silent" ) == 0 ) verbose = false;

      }

      // Input arguments incomplete
      if( definition ) {

        printf( "Insufficient input arguments.\nUse: mpiexec -n 1 ./program -file file_name [-silent]\n" );
        MPI_Finalize();

        return(0);

      }

      // Scan file for number of rows, colums and blocks
      fscanf( file, "%d,", &num_rows );
      //cout << num_rows << endl;
      fscanf( file, "%d,", &num_columns );
      //cout << num_columns << endl;
      fscanf( file, "%d,", &num_blocks );
      //cout << num_blocks << endl;

      // Allocate matrix A and load from file
      A = ( double* ) malloc( ( num_rows*num_columns )*sizeof( double ) );
      for( i = 0; i < num_rows; i++ ) {

        for( j = 0; j < num_columns; j++ ) {

          // Read term i,j from file
          double value;
          fscanf( file, "%lf,", &value );
          A[j+i*num_columns] = value;

        }

      }

      if( world_rank == 0 ) {

        if( verbose ) printf( "Matrix A:\n" );
        for( i = 0; i < num_rows; i++ ) {

          for( j = 0; j < num_columns; j++ ) {

            if( verbose ) printf( "%lf ", A[j+i*num_columns] );

          }

          if( verbose ) printf( "\n" );

        }

      }

    }

    // Synchronize processors
    MPI_Barrier( MPI_COMM_WORLD );

  }

  // Check compatibility between matrix rows and number of processors
  if( ( num_rows%world_size ) != 0 ) {

    if( world_rank == 0 ) cout << "\nNumber of rows has to be a multiple of number of processors!!!\n" << endl;

    return(0);

  }


  // START: parallel QR-decomposition using LAPACK library

  // Allocate and initialize processor k A sub-matrix
  nr = num_rows/world_size;
  double *A_k = ( double* ) malloc( ( nr*num_columns )*sizeof( double ) );
  for( i = 0; i < ( num_rows/world_size ); i++ ) {

    for( j = 0; j < num_columns; j++ ) {

      A_k[j+i*num_columns] = A[j+(i+world_rank*( num_rows/world_size ) )*num_columns];

    }

  }

  // Start timer 
  MPI_Barrier( MPI_COMM_WORLD );
  timer.CpuStart();  

  // Loop through levels
  for( l = 0; l <= ( log( world_size )/log( 2 ) ); l++ ) {

    if( verbose ) cout << "Level " << l << endl;

    // Step 1: perform QR factorization of the sub-matrix
    //int M = ( int( pow( 2, l ) )*num_rows )/world_size; 
    int M = nr; 
    int N = num_columns;  
    int K = min( M, N );
    int LDA = max( 1, N );  
    double *TAU_k = ( double* ) malloc( max( 1, ( min( M, N ) ) )*sizeof( double ) );
    double *Q_k = ( double* ) malloc( ( M*N )*sizeof( double ) );
    double *R_k = ( double* ) malloc( ( N*N )*sizeof( double ) );

    // Run DGEQRF from LAPACK
    LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, N, A_k, LDA, TAU_k );

    // Save matrix R_k
    if( verbose ) printf( "\nMatrix R_k:\n" );
    for( i = 0; i < N; i++ ) {
    
      for( j = 0; j < N; j++ ) {

        if( i > j ) {

          R_k[j+i*N] = 0.0;

        } else {

          R_k[j+i*N] = A_k[j+i*N];

        }
        if( verbose ) printf( "%lf ", R_k[j+i*N] );

      }

      if( verbose ) printf( "\n" );

    }

    // Run DORGQR from LAPACK
    LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, N, K, A_k, LDA, TAU_k );
 
    // Save matrix Q_k
    if( verbose ) printf( "\nMatrix Q_k:\n" );
    for( i = 0; i < M; i++ ) {

      for( j = 0; j < N; j++ ) {

        Q_k[j+i*N] = A_k[j+i*N];
        if( verbose ) printf( "%lf ", Q_k[j+i*N] );

      }

      if( verbose ) printf( "\n" );

    }

    // Free pointers
    if( A_k != NULL )   free ( A_k );
    if( Q_k != NULL )   free ( Q_k );
    if( TAU_k != NULL ) free ( TAU_k );

    // If for loop completed
    if( l == ( log( world_size )/log( 2 ) ) ) {
    
      // Free pointers
      if( R_k != NULL ) free ( R_k );

      break;

    }
  
    // Step 2: communicate R_k matrices between processors

    // Set processors communication scheme
    int world_comm[world_size];
    int neighbor = pow( 2, l );
    int step     = pow( 2, l + 1 );
    for( i = 0; i < world_size; i += step ) {

      for( j = i; j < ( i + neighbor ); j++ ) {

        world_comm[j] = j + neighbor;
        world_comm[j + neighbor] = j;

      }

    }

    // Allocate and initialize new k sub-matrices
    nr = 2*N;
    A_k = ( double* ) malloc( ( nr*N )*sizeof( double ) );
    if( world_rank < world_comm[world_rank] ) {

      for( i = 0; i < N; i++ ) {

        for( j = 0; j < N; j++ ) {
  
          A_k[j+i*N] = R_k[j+i*N];

        }

      }

    } else {

      for( i = N; i < 2*N; i++ ) {
  
        for( j = 0; j < N; j++ ) {

          A_k[j+i*N] = R_k[j+(i-N)*N];
  
        }

      }

    }

    // Send & Receive matrices ( use R_k as a buffer )
    MPI_Send( R_k, N*N, MPI_DOUBLE, world_comm[world_rank], 0, MPI_COMM_WORLD );
    MPI_Recv( R_k, N*N, MPI_DOUBLE, world_comm[world_rank], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

    // Complete A_k matrix with matrix received
    if( world_rank < world_comm[world_rank] ) {

      for( i = N; i < 2*N; i++ ) {

        for( j = 0; j < N; j++ ) {

          A_k[j+i*N] = R_k[j+(i-N)*N];

        }

      }

    } else {

      for( i = 0; i < N; i++ ) {

        for( j = 0; j < N; j++ ) {

          A_k[j+i*N] = R_k[j+i*N];

        }

      }

    }

    // Free pointers
    if( R_k != NULL ) free ( R_k );

    // Activate only for debbuging
    //MPI_Barrier( MPI_COMM_WORLD );

  }

  // End timer 
  MPI_Barrier( MPI_COMM_WORLD );
  double time = timer.CpuStop().CpuSeconds();

  // END: parallel QR-decomposition using LAPACK library


  // Free pointers
  if( A != NULL ) free ( A );
 
  // Print time     
  if( world_rank == 0 ) {

    printf( "\nTime: %f\n", time );

  }                   

  // Finalize MPI  
  MPI_Finalize();

  return(0);                                                                                                

}
