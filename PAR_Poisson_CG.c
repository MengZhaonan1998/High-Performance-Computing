/*
 * PAR_Poisson_CG.c
 * 2D Poison equation solver
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define DEBUG 0

#define max(a,b) ((a)>(b)?a:b)

enum
{
  X_DIR, Y_DIR
};

/* global variables */
int gridsize[2];
double precision_goal;		/* precision_goal of solution */
int max_iter;			/* maximum number of iterations alowed */

/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */

/* local grid related variables */
double **phi;			/* grid */
int **source;			/* TRUE if subgrid element is a source */
int dim[2];			/* grid dimensions */

/* Conjugate Gradient */
double **pCG, **rCG, **vCG;
double global_residue;


/* MPI */
int proc_rank;                  /* processor identifier */
int proc_coord[2];              /* coordinates of current process in processgrid */
int proc_top, proc_right, proc_bottom, proc_left;  /* ranks of neighboring procs */

int P;                          /* total number of processes */
int P_grid[2];                  /* process grid dimensions */
MPI_Comm grid_comm;             /* grid COMMUNICATOR */ 
MPI_Status status;
MPI_Datatype border_type[2];

int offset[2];

double wtime;

void Setup_Proc_Grid(int argc, char **argv);
void Setup_Grid();
void Do_Step();
void InitCG();
void Solve();
void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();

void start_timer()
{
  if (!timer_on)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    ticks = clock();
    wtime = MPI_Wtime();
    timer_on = 1;
  }
}

void resume_timer()
{
  if (!timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 1;
  }
}

void stop_timer()
{
  if (timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 0;
  }
}

void print_timer()
{

  if (timer_on)
  {
    stop_timer();
    printf("(%i) Elapsed processortime: %14.6f s (%5.1f%% CPU)\n",
		   proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
    
    resume_timer();
  }
  else
    printf("(%i) Elapsed processortime: %14.6f s (%5.1f%% CPU)\n", 
		   proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}    


void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("%s\n", mesg);
  if (terminate)
    exit(1);
}



void Setup_Proc_Grid(int argc, char **argv)
{
  int wrap_around[2];
  int reorder;

  Debug("My_MPI_Init",0);

  /* Retrieve the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  
  /* Calculate the number of processes per column and per row for the grid */
  if (argc > 2)
  {
     P_grid[X_DIR] = atoi(argv[1]);   // ?? argc argv??
     P_grid[Y_DIR] = atoi(argv[2]);
     if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
	  Debug("ERROR Process grid dimensions do not match with P", 1);
  }
  else
     Debug("ERROR Wrong paramter input", 1);

  /* Create process topology (2D grid) */
  wrap_around[X_DIR] = 0;
  wrap_around[Y_DIR] = 0;    /* do not connect first and last process */
  reorder = 1;  /* reorder process ranks */

  MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm);  /* Creates a new communicator grid_comm */
  
  /* Retrieve new rank and cartesian coordinates of this process */
  MPI_Comm_rank(grid_comm, &proc_rank);

  MPI_Cart_coords(grid_comm, proc_rank, 2, proc_coord);

  printf("(%i)(x,y)=(%i,%i)\n",proc_rank,proc_coord[X_DIR],proc_coord[Y_DIR]);

  /* calculate ranks of neighboring processes */
 
  // Note the output of MPI_Cart_shift() function !!! 
  MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_bottom,  &proc_top);
  MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);

  //if (DEBUG)
  printf("(%i) top %i, right %i, bottom %i, left %i\n",proc_rank, proc_top, proc_right, proc_bottom, proc_left);
 
  /*
  FILE *f;
  f = fopen("ex1_2_4.txt","a");
  fprintf(f,"The grid topology is %i * %i\n",atoi(argv[1]),atoi(argv[2]));
  fclose(f); 
  */

}

void Setup_Grid()
{
  int x, y, s;
  double source_x, source_y, source_val;
  FILE *f;
  int upper_offset[2];

  Debug("Setup_Subgrid", 0);   
  
  if (proc_rank == 0)
  {
   f = fopen("input.dat", "r");
   if (f == NULL)
        Debug("Error opening input.dat", 1);
   fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
   fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
   fscanf(f, "precision goal: %lf\n", &precision_goal);
   fscanf(f, "max iterations: %i\n", &max_iter);
  }
  MPI_Bcast(&gridsize, 2, MPI_INT, 0, MPI_COMM_WORLD);           /* broadcast the array gridsize in one call */
  MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* broadcast precision_goal */
  MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);           /* broadcast max_iter */

  /* Calculate top left corner of local grid  */
  offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
  offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];
  upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR]+1) / P_grid[X_DIR];
  upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR]+1) / P_grid[Y_DIR];
 
  /* Calculate dimensions of local grid  */
  dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];
  dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];

  /* Add space for rows/columns of neighboring grid  */
  dim[Y_DIR] += 2;
  dim[X_DIR] += 2;

  /* allocate memory */
  if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
    Debug("Setup_Subgrid : malloc(phi) failed", 1);
  if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
    Debug("Setup_Subgrid : malloc(source) failed", 1);
  if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
    Debug("Setup_Subgrid : malloc(*phi) failed", 1);
  if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) == NULL)
    Debug("Setup_Subgrid : malloc(*source) failed", 1);
  for (x = 1; x < dim[X_DIR]; x++)
  {
    phi[x] = phi[0] + x * dim[Y_DIR];
    source[x] = source[0] + x * dim[Y_DIR];
  }

  /* set all values to '0' */
  for (x = 0; x < dim[X_DIR]; x++)
    for (y = 0; y < dim[Y_DIR]; y++)
    {
      phi[x][y] = 0.0;
      source[x][y] = 0;
    }

  /* put sources in field */
  do
  {
    if (proc_rank == 0)
    	s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);
    MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (s==3)
    {
      MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);     /* broadcast source_x */ 
      MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);     /* broadcast source_y */
      MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);   /* broadcast source_val */

      x = source_x * gridsize[X_DIR];
      y = source_y * gridsize[Y_DIR];
      x += 1;
      y += 1;
      
      x = x - offset[X_DIR];
      y = y - offset[Y_DIR];
      if (x>0 && x < dim[X_DIR]-1 && y > 0 && y < dim[Y_DIR]-1)
      {                  /*indices in domain of this process*/
         phi[x][y] = source_val;
	 source[x][y] = 1;
      }


    }
  }
  while (s==3);
  
  if (proc_rank == 0) fclose(f);
}

void Setup_MPI_Datatypes()
{
  Debug("Setup_MPI_Datatypes",0);

  /* Datatype for vertical data exchange (Y_DIR) */
  MPI_Type_vector(dim[X_DIR]-2,1,dim[Y_DIR],MPI_DOUBLE,&border_type[Y_DIR]);
  MPI_Type_commit(&border_type[Y_DIR]);

  /* Datatype for horizontal data exchange (X_DIR) */
  MPI_Type_vector(dim[Y_DIR]-2,1,1,MPI_DOUBLE,&border_type[X_DIR]);
  MPI_Type_commit(&border_type[X_DIR]);
}

void Do_Step()
{
  int x, y;
  double a, g, global_pdotv, pdotv, global_new_rdotr, new_rdotr;

  /* Calculate "v" in interior of my grid (matrix-vector multiply) */
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
    {
      vCG[x][y] = pCG[x][y];
      if (source[x][y] != 1)    /* only if point is not fixed */
        vCG[x][y] -= 0.25 * (pCG[x+1][y] + pCG[x-1][y] + pCG[x][y+1] + pCG[x][y-1]);
    }

  pdotv = 0;
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      pdotv += pCG[x][y] * vCG[x][y];

  MPI_Allreduce(&pdotv, &global_pdotv, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
  
  a = global_residue / global_pdotv;

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      phi[x][y] += a* pCG[x][y];
  
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      rCG[x][y] -= a * vCG[x][y];

  new_rdotr = 0;
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      new_rdotr += rCG[x][y] * rCG[x][y];

  MPI_Allreduce(&new_rdotr, &global_new_rdotr, 1, MPI_DOUBLE, MPI_SUM, grid_comm);

  g = global_new_rdotr / global_residue;
  global_residue = global_new_rdotr;

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      pCG[x][y] = rCG[x][y] + g * pCG[x][y];
}

void InitCG()
{
  int x,y;
  double rdotr = 0;

  /* allocate memory for CG arrays */
  pCG = malloc(dim[X_DIR]*sizeof(*pCG));
  pCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**pCG));
  for (x = 1; x < dim[X_DIR]; x++) pCG[x] = pCG[0] + x * dim[Y_DIR];

  rCG = malloc(dim[X_DIR] * sizeof(*rCG));
  rCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**rCG));
  for (x = 1; x < dim[X_DIR]; x++) rCG[x] = rCG[0] + x * dim[Y_DIR];

  vCG = malloc(dim[X_DIR] * sizeof(*vCG));
  vCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**vCG));
  for (x = 1; x < dim[X_DIR]; x++) vCG[x] = vCG[0] + x * dim[Y_DIR];

  /* initiate rCG and pCG */
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
    {
      rCG[x][y] = 0;
      if (source[x][y] != 1)
	rCG[x][y] = 0.25 * (phi[x+1][y] + phi[x-1][y] + phi[x][y+1] + phi[x][y-1]) - phi[x][y];
      pCG[x][y] = rCG[x][y];
      rdotr += rCG[x][y] * rCG[x][y];
    }

  /* obtain the global_residue also for the initial phi */
  MPI_Allreduce(&rdotr, &global_residue, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
}


void Exchange_Borders()
{
  Debug("Exchange_Borders",0);
  
  MPI_Sendrecv(&pCG[1][dim[Y_DIR]-2], 1, border_type[Y_DIR], proc_top, 0,
	       &pCG[1][0], 1, border_type[Y_DIR], proc_bottom, 0,
	       grid_comm, &status);    
  
  MPI_Sendrecv(&pCG[1][1], 1, border_type[Y_DIR], proc_bottom, 0,
	       &pCG[1][dim[Y_DIR]-1], 1, border_type[Y_DIR], proc_top, 0,
	       grid_comm, &status);

  MPI_Sendrecv(&pCG[1][1], 1, border_type[X_DIR], proc_left, 0,
	       &pCG[dim[X_DIR]-1][1], 1, border_type[X_DIR], proc_right, 0,
	       grid_comm, &status);

  MPI_Sendrecv(&pCG[dim[X_DIR]-2][1], 1, border_type[X_DIR], proc_right, 0,
	       &pCG[0][1], 1, border_type[X_DIR], proc_left, 0,
	       grid_comm, &status);
  
}



void Solve()
{
  int count = 0;
 
  Debug("Solve", 0);

  InitCG();
 
  while (global_residue > precision_goal && count < max_iter)
  {
    Exchange_Borders();
    Do_Step();
    count++; 
  }
  
  printf("(Processor rank: %i) Number of iterations : %i\n", proc_rank, count);

}

void Write_Grid()
{
  int x, y;
  FILE *f;
  
  char filename[40];
  sprintf(filename,"output%i.dat",proc_rank);

  if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid : fopen failed", 1);
  
  Debug("Write_Grid", 0);

  for (x = 1; x < dim[X_DIR] - 1; x++)
   for (y = 1; y < dim[Y_DIR] - 1; y++)
    fprintf(f, "%i %i %f\n", x + offset[X_DIR], y + offset[Y_DIR], phi[x][y]);

  fclose(f);
}

void Clean_Up()
{
  Debug("Clean_Up", 0);

  free(phi[0]);
  free(phi);
  free(source[0]);
  free(source);
}

int main(int argc, char **argv)
{ 
  MPI_Init(&argc, &argv);

  Setup_Proc_Grid(argc, argv);

  start_timer();

  Setup_Grid();

  Setup_MPI_Datatypes();  
  
  Solve();

  Write_Grid();

  print_timer();

  Clean_Up();  

  MPI_Finalize();

  return 0;
}
