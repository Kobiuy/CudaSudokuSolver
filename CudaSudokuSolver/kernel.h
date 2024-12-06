#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <atomic>
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include "sudokuCPU.h"
#include <iomanip>
#include <chrono>
#define BOARD_SIZE 81
#define BOARD_DIM 9
#define BLOCK_SIZE 1024
#define BLOCK_COUNT 1
#define STACK_CAPACITY 256
#define RETURN_STACK_CAPACITY 81
const int N = 3;
const int N2 = N * N;
using namespace std;
using namespace std::chrono;

struct Stack {
	struct Element {
		char elem1;
		char elem2;
	};

	Element array[STACK_CAPACITY];
	int top;
	uint16_t capacity;

	__device__ Stack();

	__device__ bool push(char cValue, char iValue);

	__device__ bool pop(char* cValue, char* iValue);
};
struct Boards {
	uint16_t* columnBitmask;
	uint16_t* rowBitmask;
	uint16_t* squareBitmask;
	char* boardValues;
	uint16_t* gameID;
	bool* done;
	bool* valid;

	Boards();
	Boards(const char* data, uint16_t initBoardsCount, uint16_t boardsCount);
	~Boards();
	__device__ void GetPossibleValuesCount(uint16_t tid, char x, char y, char* num_zeroes);
	__device__ void GetPossibleValues(uint16_t tid, char x, char y, char result[10]);
	__device__ void SetValueAndUpdateBitmasks(uint16_t tid, char x, char y, char value);
	__device__ void SetValueAndUpdateBitmasks(char* values, uint16_t tid, char x, char y, char value);
	__device__ void GoBackBitmask(char* values, uint16_t tid, char x, char y);
};
struct SmallStack {
	char array[RETURN_STACK_CAPACITY];
	int top = -1;
	uint16_t capacity = RETURN_STACK_CAPACITY;
	__device__ SmallStack();
	__device__ bool push(char cValue);
	__device__ bool pop(char* cValue);
};

void print(char* solution);
__device__ void DebugPrinting(char* values);
__device__ void PrintBitmasksFor(int x, int y, uint16_t Columns[], uint16_t Rows[], uint16_t Squeres[]);
void RunCpu(char** board, int boardCount, char* sols);
void PrintCardInfo();
int CalculateMaxBoardCount();
int GetMaxBlocks();
bool isValidSudoku(char board[81]);
void SaveToFile(char* line, int boardsCount);
cudaError_t SolveSudokuWithCuda(char* resultBoards, const char* board, const int initBoardCount, const int maxBoardsCount);
__global__ void SolveSudokuKernel(char* resultBoard, int* boardsCount, int* maxBoardsCount, Boards* globalBoards, int* runningThreads);
__global__ void BacktrackingKernel(char* resultBoard, int* boardsCount, Boards* globalBoards);
__global__ void warm_up_gpu();

