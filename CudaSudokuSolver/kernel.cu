#include "kernel.h"

int main()
{
	int maxGames = CalculateMaxBoardCount();
	int maxBlocks = GetMaxBlocks();

	// Reading data from file
	char filename[] = "dane.txt";
	printf(filename);
	printf("\n");
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		printf("Nie można otworzyć pliku %s.\n", filename);
		return 1;
	}

	char** lines = (char**)malloc(maxGames * sizeof(char*));
	int boardCount = 0;
	lines[0] = (char*)malloc((BOARD_SIZE + 2) * sizeof(char));
	while (fgets(lines[boardCount], BOARD_SIZE + 2, file)) {
		lines[boardCount][strcspn(lines[boardCount], "\r\n")] = '\0';
		boardCount++;
		if (boardCount >= maxGames) {
			printf("Przekroczono maksymalna liczbe tablic (%d).\n", maxGames);
			break;
		}
		lines[boardCount] = (char*)malloc((BOARD_SIZE + 2) * sizeof(char));
	}
	fclose(file);

	// Liczenie zer, żeby nie było za dużo nieużywanych wątków, próba przewidzenia ilości potrzebnych wątków
	int zeros = 0;
	for (int i = 0; i < boardCount; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {
			if (lines[i][j] == '0') ++zeros;
		}
	}

	// Maksymalna ilość plansz to minimum z dostępnej pamięci/dostępnych wątków
	int maxBoardsCount = min(zeros * 40, min(maxBlocks * BLOCK_SIZE, CalculateMaxBoardCount()));
	printf("maxBoardsCount: %d, Zeros: %d, Max: %d\n", maxBoardsCount, zeros, maxBlocks * BLOCK_SIZE);


	// Przygotowywanie danych dla GPU
	char* boards = new char[boardCount * BOARD_SIZE];
	for (int i = 0; i < boardCount; i++) {
		strcpy(boards + BOARD_SIZE * i, lines[i]);
	}
	char* resultBoards = new char[boardCount * BOARD_SIZE];

	// Solving on CPU
	RunCpu(lines, boardCount);

	// Solving on GPU
	cudaError_t cudaStatus = SolveSudokuWithCuda(resultBoards, boards, boardCount, maxBoardsCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SolveSudokuWithCuda failed!");
		return 1;
	}


	for (int i = 0; i < boardCount; i++) { // Wyświetlanie wyniku
		printf("board Id %d\n", i);
		print(resultBoards + i * BOARD_SIZE);
	}

	SaveToFile(resultBoards, boardCount); // Zapis wyniku z GPU

	for (int i = 0; i < boardCount; i++) { // Sprawdzania poprawności wyników
		if (!isValidSudoku(resultBoards + i * BOARD_SIZE))
			printf("Wykryto niepoprawne rozwiązanie w planszy: %d\n", i);
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// Freeing memory
	for (int i = 0; i < boardCount; i++) {
		free(lines[i]);
	}
	free(lines);
	delete[] resultBoards;
	delete[] boards;
	return 0;
}

cudaError_t SolveSudokuWithCuda(char* resultBoards, const char* board, int initBoardCount, int maxBoardsCount)
{
	PrintCardInfo();

	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;

	char* dev_resultBoards;
	int* dev_BoardsCount;
	int* dev_MaxBoardsCount;
	int* dev_runnigThreads;
	cudaError_t cudaStatus;
	int blockCount = maxBoardsCount / BLOCK_SIZE;
	if (maxBoardsCount % BLOCK_SIZE != 0) {
		blockCount++;
		maxBoardsCount = blockCount * BLOCK_SIZE;
	}
	Boards tempBoard(board, initBoardCount, maxBoardsCount);
	Boards* dev_globalBoards;

	cudaStatus = cudaSetDevice(0); // Choosing which GPU to run on
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_resultBoards, initBoardCount * BOARD_SIZE * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_BoardsCount, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_MaxBoardsCount, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_runnigThreads, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMalloc(&dev_globalBoards, sizeof(Boards));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_BoardsCount, &initBoardCount, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_MaxBoardsCount, &maxBoardsCount, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_runnigThreads, &initBoardCount, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaMemcpy(dev_globalBoards, &tempBoard, sizeof(Boards), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	warm_up_gpu << <blockCount, BLOCK_SIZE >> > ();
	{ // AfterKernelStuff
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "warm_up_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching warm_up_gpu!\n", cudaStatus);
			goto Error;
		}
	}
	ts = high_resolution_clock::now();
	SolveSudokuKernel << <blockCount, BLOCK_SIZE >> > (dev_resultBoards, dev_BoardsCount, dev_MaxBoardsCount, dev_globalBoards, dev_runnigThreads);
	{ // AfterKernelStuff
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "SolveSudokuKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SolveSudokuKernel!\n", cudaStatus);
			goto Error;
		}
	}
	te = high_resolution_clock::now();
	cout << "Time:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;
	BacktrackingKernel << <blockCount, BLOCK_SIZE >> > (dev_resultBoards, dev_BoardsCount, dev_globalBoards);
	{ // AfterKernelStuff
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "BacktrackingKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching BacktrackingKernel!\n", cudaStatus);
			goto Error;
		}
	}
	te = high_resolution_clock::now();
	cout << "Time:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(resultBoards, dev_resultBoards, initBoardCount * BOARD_SIZE * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:

	cudaFree(dev_resultBoards);
	cudaFree(dev_BoardsCount);
	cudaFree(dev_MaxBoardsCount);
	cudaFree(dev_globalBoards);
	return cudaStatus;
}

__global__ void SolveSudokuKernel(char* resultBoard, int* boardsCount, int* maxBoardsCount, Boards* globalBoards, int* runningThreads)
{
	char optionsCount;
	char possibleValues[10];
	unsigned short tid = threadIdx.x + blockIdx.x * blockDim.x;
	char values[BOARD_SIZE]; 
	bool firstTime = true;
	uint16_t ID;

	while (*runningThreads > 0) {
		//__syncthreads(); // TODO 
		char bestX, bestY; char minCount = 10;

		if (tid < *boardsCount && globalBoards->valid[tid]) {
			if (firstTime) { // Jeśli pierwszy raz, to skopiuj values do rejestru
				firstTime = false;
				memcpy(values, globalBoards->boardValues + BOARD_SIZE * tid, BOARD_SIZE * sizeof(char));
				ID = globalBoards->gameID[tid];
			}
			if (!globalBoards->done[ID]) {
#pragma unroll BOARD_DIM
				for (char x = 0; x < BOARD_DIM; x++) {
#pragma unroll BOARD_DIM
					for (char y = 0; y < BOARD_DIM; y++) {
						if (values[(y * BOARD_DIM) + x] == '0') {
							globalBoards->GetPossibleValuesCount(tid, x, y, &optionsCount);

							if (optionsCount < minCount) {
								bestX = x; bestY = y; minCount = optionsCount;
								if (minCount <= 0) break;
							}
						}
					}
					if (minCount <= 0) break;
				}
				if (minCount < 10 && minCount > 0) {
					globalBoards->GetPossibleValues(tid, bestX, bestY, possibleValues);
					unsigned short requiredBoards = minCount - 1;
					int currentBoards = atomicAdd(boardsCount, requiredBoards);
					if (currentBoards + requiredBoards > *maxBoardsCount) {
						atomicSub(boardsCount, requiredBoards);
						*runningThreads = -*runningThreads;
						break;
					}

					for (char i = 1; i < minCount; ++i) {

						unsigned short boardID = currentBoards + (i - 1);
#pragma unroll BOARD_DIM
						for (char x = 0; x < BOARD_DIM; ++x) {
#pragma unroll BOARD_DIM
							for (char y = 0; y < BOARD_DIM; ++y) {
								globalBoards->SetValueAndUpdateBitmasks(boardID, x, y, values[(y * BOARD_DIM) + x]);
							}
						}

						//memcpy(globalBoards->columnBitmask + boardID * BOARD_DIM, globalBoards->columnBitmask + tid * BOARD_DIM, 2 * BOARD_DIM);
						//memcpy(globalBoards->rowBitmask + boardID * BOARD_DIM, globalBoards->rowBitmask + tid * BOARD_DIM, 2 * BOARD_DIM);
						//memcpy(globalBoards->squareBitmask + boardID * BOARD_DIM, globalBoards->squareBitmask + tid * BOARD_DIM, 2 * BOARD_DIM);
						//memcpy(globalBoards->boardValues + boardID * BOARD_SIZE, values, BOARD_SIZE);

						globalBoards->gameID[boardID] = ID;
						globalBoards->SetValueAndUpdateBitmasks(boardID, bestX, bestY, possibleValues[i]);
						globalBoards->valid[boardID] = true;
						atomicAdd(runningThreads, 1);
					}

					// Reużywanie obecnego wątku
					globalBoards->SetValueAndUpdateBitmasks(values, tid, bestX, bestY, possibleValues[0]);

				}
				else {
#pragma unroll BOARD_SIZE
					for (char i = 0; i < BOARD_SIZE; i++) {
						if (values[i] == '0') {
							globalBoards->valid[tid] = false;
							break;
						}
					}
					if (globalBoards->valid[tid]) {

						globalBoards->done[ID] = true;
						memcpy(resultBoard + ID * BOARD_SIZE, values, BOARD_SIZE * sizeof(char));
						//for (char i = 0; i < BOARD_SIZE; i++) {
						//	resultBoard[ID * BOARD_SIZE + i] = values[i];
						//}
					}
					atomicSub(runningThreads, 1);
					break;
				}
			}
		}
	}
	if (!firstTime) {
		memcpy(globalBoards->boardValues + BOARD_SIZE * tid, values, BOARD_SIZE * sizeof(char));
	}
}
__global__ void BacktrackingKernel(char* resultBoard, int* boardsCount, Boards* globalBoards)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (globalBoards->valid[tid] && !globalBoards->done[globalBoards->gameID[tid]]) {
		char optionsCount;
		char possibleValues[10];
		Stack* stack = new Stack();
		char value = '0';
		char place;
		char lastplace;
		uint16_t ID = globalBoards->gameID[tid];
		char values[BOARD_SIZE];
		SmallStack* usedStack = new SmallStack();
		memcpy(values, globalBoards->boardValues + BOARD_SIZE * tid, BOARD_SIZE * sizeof(char));
		do {
			if (value != '0') { // Jeśli to nie pierwsza iteracja, pobierz dane z usedStack
				while (values[place] != '0') {
					usedStack->pop(&lastplace);
					globalBoards->GoBackBitmask(values, tid, lastplace % BOARD_DIM, lastplace / BOARD_DIM);
				}
				globalBoards->SetValueAndUpdateBitmasks(values, tid, place % BOARD_DIM, place / BOARD_DIM, value);
				usedStack->push(place);
			}
			char bestX, bestY; char minCount = 10;

			// Szukanie najlepszego pola do wypełnienia
#pragma unroll BOARD_DIM
			for (char x = 0; x < BOARD_DIM; x++) {
#pragma unroll BOARD_DIM
				for (char y = 0; y < BOARD_DIM; y++) {
					if (values[(y * BOARD_DIM) + x] == '0') {
						globalBoards->GetPossibleValuesCount(tid, x, y, &optionsCount);
						if (optionsCount < minCount) {
							bestX = x; bestY = y; minCount = optionsCount;
							if (minCount <= 0) break;
						}

					}
				}
				if (minCount <= 0) break;
			}

			if (minCount != 10 && minCount != 0) {
				globalBoards->GetPossibleValues(tid, bestX, bestY, possibleValues);
				bool overflow = false;
				for (char i = 0; i < minCount; ++i) {
					if (!stack->push(possibleValues[i], bestY * BOARD_DIM + bestX))
					{
						printf("Stack Limit Exceeded for board %d! Returning...\n", ID);
						overflow = true;
						break;
					}
				}
				if (overflow) break;

			}
			else { // Gdy nie ma opcji żadnych
#pragma unroll BOARD_SIZE
				for (char i = 0; i < BOARD_SIZE; i++) {
					if (values[i] == '0') {
						globalBoards->valid[tid] = false;
						break;
					}
				}
				if (globalBoards->valid[tid]) {
					globalBoards->done[ID] = true;
					//#pragma unroll BOARD_SIZE
					//					for (char i = 0; i < BOARD_SIZE; i++) {
					//						resultBoard[ID * BOARD_SIZE + i] = values[i];
					//					}
					memcpy(resultBoard + ID * BOARD_SIZE, values, BOARD_SIZE * sizeof(char));

					break;
				}
				globalBoards->valid[tid] = true;
			}
		} while (stack->pop(&value, &place) && !globalBoards->done[ID]);
		delete stack;
		delete usedStack;
	}
}
__global__ void warm_up_gpu() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

__device__ Stack::Stack() : top(-1), capacity(STACK_CAPACITY) {}
__device__ bool Stack::push(char cValue, char iValue) {
	if (top >= capacity - 1) {
		return false;
	}
	array[++top] = { cValue, iValue };
	return true;
}
__device__ bool Stack::pop(char* cValue, char* iValue) {
	if (top < 0) {
		return false;
	}
	*cValue = array[top].elem1;
	*iValue = array[top].elem2;
	top--;
	return true;
}

__device__ SmallStack::SmallStack() {};
__device__ bool SmallStack::push(char cValue) {
	if (top >= capacity - 1) {
		return false;
	}
	array[++top] = cValue;
	return true;
}
__device__ bool SmallStack::pop(char* cValue) {
	if (top < 0) {
		return false;
	}
	*cValue = array[top];
	top--;
	return true;
}

Boards::Boards() {}
Boards::Boards(const char* data, uint16_t initBoardsCount, uint16_t boardsCount) {
	size_t masksSize = BOARD_DIM * boardsCount * sizeof(uint16_t);
	size_t valuesSize = BOARD_SIZE * boardsCount * sizeof(char);
	size_t idSize = boardsCount * sizeof(uint16_t);
	size_t boolSize = boardsCount * sizeof(bool);
	uint16_t* h_Columns = new uint16_t[masksSize]();
	uint16_t* h_Rows = new uint16_t[masksSize]();
	uint16_t* h_Squeres = new uint16_t[masksSize]();
	char* h_values = new char[valuesSize]();
	uint16_t* h_ID = new uint16_t[idSize];
	bool* h_done = new bool[boolSize]();
	bool* h_valid = new bool[boolSize]();

	memset(h_values, '0', valuesSize);
	memset(h_Columns, 0, masksSize);
	memset(h_Squeres, 0, masksSize);
	memset(h_Rows, 0, masksSize);

	for (uint16_t i = 0; i < boardsCount; i++) {
		h_ID[i] = i;
		h_valid[i] = false;
		h_done[i] = false;
	}

	// Create Bitmasks and Values arrays
	for (uint16_t id = 0; id < initBoardsCount; id++) {
		h_valid[id] = true;
#pragma unroll BOARD_DIM
		for (char x = 0; x < BOARD_DIM; x++) {
#pragma unroll BOARD_DIM
			for (char y = 0; y < BOARD_DIM; y++) {
				h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] = data[x + BOARD_DIM * y + id * BOARD_SIZE];
				if (h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] >= '1' && h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] <= '9') {
					h_Columns[id * BOARD_DIM + x] |= (1 << (h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] - '0'));
					h_Rows[id * BOARD_DIM + y] |= (1 << (h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] - '0'));
					h_Squeres[id * BOARD_DIM + (y / 3) * 3 + (x / 3)] |= (1 << (h_values[(id * BOARD_SIZE) + (y * BOARD_DIM) + x] - '0'));
				}
			}
		}
	}

	// Move arrays to Device memory
	cudaMalloc(&columnBitmask, masksSize);
	cudaMalloc(&rowBitmask, masksSize);
	cudaMalloc(&squareBitmask, masksSize);
	cudaMalloc(&boardValues, valuesSize);
	cudaMalloc(&gameID, idSize);
	cudaMalloc(&done, boolSize);
	cudaMalloc(&valid, boolSize);

	cudaMemcpy(columnBitmask, h_Columns, masksSize, cudaMemcpyHostToDevice);
	cudaMemcpy(rowBitmask, h_Rows, masksSize, cudaMemcpyHostToDevice);
	cudaMemcpy(squareBitmask, h_Squeres, masksSize, cudaMemcpyHostToDevice);
	cudaMemcpy(boardValues, h_values, valuesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gameID, h_ID, idSize, cudaMemcpyHostToDevice);
	cudaMemcpy(done, h_done, boolSize, cudaMemcpyHostToDevice);
	cudaMemcpy(valid, h_valid, boolSize, cudaMemcpyHostToDevice);

	// Cleanup
	delete[] h_Columns;
	delete[] h_Rows;
	delete[] h_Squeres;
	delete[] h_values;
	delete[] h_ID;
	delete[] h_done;
	delete[] h_valid;

}
Boards::~Boards() {
	cudaFree(columnBitmask);
	cudaFree(rowBitmask);
	cudaFree(squareBitmask);
	cudaFree(boardValues);
	cudaFree(gameID);
	cudaFree(done);
	cudaFree(valid);
}
__device__ void Boards::GetPossibleValuesCount(uint16_t tid, char x, char y, char* num_zeroes) {
	*num_zeroes = 0;
	uint16_t value = columnBitmask[tid * BOARD_DIM + x] | rowBitmask[tid * BOARD_DIM + y] | squareBitmask[tid * BOARD_DIM + (y / 3) * 3 + (x / 3)];
#pragma unroll BOARD_DIM
	for (char i = 1; i <= BOARD_DIM; ++i) {
		if ((value & (1 << i)) == 0) {
			++*num_zeroes;
		}
	}
}
__device__ void Boards::GetPossibleValues(uint16_t tid, char x, char y, char result[10]) {
	uint16_t value = columnBitmask[tid * BOARD_DIM + x] | rowBitmask[tid * BOARD_DIM + y] | squareBitmask[tid * BOARD_DIM + (y / 3) * 3 + (x / 3)];
	int pos = 0;
#pragma unroll BOARD_DIM
	for (int i = 1; i <= BOARD_DIM; ++i) {
		if ((value & (1 << i)) == 0) {
			result[pos++] = '0' + i;
		}
	}
}
__device__ void Boards::SetValueAndUpdateBitmasks(uint16_t tid, char x, char y, char value) {
	boardValues[(tid * BOARD_SIZE) + (y * BOARD_DIM) + x] = value;
	columnBitmask[tid * BOARD_DIM + x] |= (1 << (value - '0'));
	rowBitmask[tid * BOARD_DIM + y] |= (1 << (value - '0'));
	squareBitmask[tid * BOARD_DIM + (y / 3) * 3 + (x / 3)] |= (1 << (value - '0'));
}
__device__ void Boards::SetValueAndUpdateBitmasks(char* values, uint16_t tid, char x, char y, char value) {
	values[(y * BOARD_DIM) + x] = value;
	columnBitmask[tid * BOARD_DIM + x] |= (1 << (value - '0'));
	rowBitmask[tid * BOARD_DIM + y] |= (1 << (value - '0'));
	squareBitmask[tid * BOARD_DIM + (y / 3) * 3 + (x / 3)] |= (1 << (value - '0'));
}
__device__ void Boards::GoBackBitmask(char* values, uint16_t tid, char x, char y) {
	char value = values[(y * BOARD_DIM) + x];
	columnBitmask[tid * BOARD_DIM + x] &= ~(1 << (value - '0'));
	rowBitmask[tid * BOARD_DIM + y] &= ~(1 << (value - '0'));
	squareBitmask[tid * BOARD_DIM + (y / 3) * 3 + (x / 3)] &= ~(1 << (value - '0'));
	values[(y * BOARD_DIM) + x] = '0';
}

int CalculateMaxBoardCount() {
	size_t free_t, total_t;
	cudaMemGetInfo(&free_t, &total_t);
	return (free_t / 140);
}
bool isValidSudoku(char board[BOARD_SIZE]) {
	for (int i = 0; i < BOARD_DIM; i++) {
		for (int j = 0; j < BOARD_DIM; j++) {
			for (int col = 0; col < BOARD_DIM; col++) {
				if (board[i * BOARD_DIM + j] == board[col * BOARD_DIM + j] && col != i) return false;
			}
			for (int row = 0; row < BOARD_DIM; row++) {
				if (board[i * BOARD_DIM + j] == board[i * BOARD_DIM + row] && row != j) return false;
			}
			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					if (board[i * BOARD_DIM + j] == board[(i / 3 * 3 + a) * BOARD_DIM + j / 3 * 3 + b] && ((i / 3 * 3 + a) * BOARD_DIM + j / 3 * 3 + b) != (i * BOARD_DIM + j)) {
						return false;
					}
				}
			}
		}
	}
	return true;
}

void SaveToFile(char* line, int boardsCount) {
	FILE* file = fopen("wyniki.txt", "w");

	if (file == NULL) {
		perror("Nie udało się otworzyć pliku");
		return;
	}
	for (int i = 0; i < boardsCount; i++) {
		char shortLine[82];
		strncpy(shortLine, line + i * BOARD_SIZE, BOARD_SIZE);
		shortLine[BOARD_SIZE] = '\0';
		fprintf(file, "%s\n", shortLine);
	}
	fclose(file);
}

void RunCpu(char** board, int boardCount) {
	char solution[BOARD_SIZE];
	auto ts = high_resolution_clock::now();
	for (int i = 0; i < boardCount; i++) {
		int res = sudokuCPU(board[i], solution);
	}
	auto te = high_resolution_clock::now();
	cout << "Time:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

}

void print(char* solution)
{
	for (int i = 0, ij = 0; i < N2; ++i)
	{
		if (i % N == 0) printf("\n");
		for (int j = 0; j < N2; ++j, ++ij)
		{
			if (j % N == 0) printf(" |");
			printf("  %c", solution[ij]);
		}
		printf(" |\n");
	}
	printf("\n");
}

void PrintCardInfo() {
	cudaDeviceProp deviceProp;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	int maxThreadsPerSM = 0;


	if (cudaStatus != cudaSuccess) {
		printf("Failed to get properties for device %d: %s\n", 0, cudaGetErrorString(cudaStatus));
		return;
	}
	cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to get properties for device %d: %s\n", 0, cudaGetErrorString(cudaStatus));
		return;
	}
	int maxBlocksPerSM = maxThreadsPerSM / deviceProp.maxThreadsPerBlock;


	printf("\nInformation for device %d:\n", 0);
	printf("GPU Name: %s\n", deviceProp.name);
	printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("CUDA Cores (Multiprocessors): %d\n", deviceProp.multiProcessorCount);
	printf("Clock Rate (MHz): %.2f\n", deviceProp.clockRate / 1000.0f);
	printf("Memory Clock Rate (MHz): %.2f\n", deviceProp.memoryClockRate / 1000.0f);
	printf("Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
	printf("Total Global Memory (MB): %zu\n", deviceProp.totalGlobalMem / (1024 * 1024));
	printf("L2 Cache Size (bytes): %d\n", deviceProp.l2CacheSize);
	printf("Shared Memory Per Block (bytes): %zu\n", deviceProp.sharedMemPerBlock);
	printf("Registers Per Block: %d\n", deviceProp.regsPerBlock);
	printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Max threads per SM: %d\n", maxThreadsPerSM);
	printf("Max blocks per SM: %d\n", maxBlocksPerSM);
	printf("Max blocks for the device: %d\n", maxBlocksPerSM * deviceProp.multiProcessorCount);
	printf("Max Threads Dim: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max Grid Size: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Warp Size: %d\n", deviceProp.warpSize);
	printf("Max Pitch Memory (bytes): %zu\n", deviceProp.memPitch);
	printf("Number of Async Engines: %d\n", deviceProp.asyncEngineCount);
	printf("Texture Alignment (bytes): %zu\n", deviceProp.textureAlignment);
	printf("Device Overlap: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
	printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
	printf("ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
	printf("PCI Bus ID: %d\n", deviceProp.pciBusID);
	printf("PCI Device ID: %d\n", deviceProp.pciDeviceID);
	printf("Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
	printf("Memory Clock Rate (MHz): %d\n", deviceProp.memoryClockRate / 1000);
	printf("\n");
}

int GetMaxBlocks() {
	cudaDeviceProp deviceProp;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	int maxThreadsPerSM = 0;

	if (cudaStatus != cudaSuccess) {
		printf("Failed to get properties for device %d: %s\n", 0, cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to get properties for device %d: %s\n", 0, cudaGetErrorString(cudaStatus));
		return -1;
	}
	return maxThreadsPerSM * deviceProp.multiProcessorCount / deviceProp.maxThreadsPerBlock;
}

// Debugging code
__device__ void DebugPrinting(char* values) {
	for (int i = 0, ij = 0; i < N2; ++i)
	{
		if (i % N == 0) printf("\n");
		for (int j = 0; j < N2; ++j, ++ij)
		{
			if (j % N == 0) printf(" |");
			if (values[ij] != '0')
				printf("  %c", values[ij]);
			else
				printf("  -");

		}
		printf(" |\n");
	}
}

__device__ void PrintBitmasksFor(int x, int y, uint16_t Columns[], uint16_t Rows[], uint16_t Squeres[]) {
	for (int i = 1; i <= BOARD_DIM; ++i) {
		unsigned int bit = (Columns[x] >> i) & 1;
		printf("%u", bit);
	}
	printf("\n");
	for (int i = 1; i <= BOARD_DIM; ++i) {
		unsigned int bit = (Rows[y] >> i) & 1;
		printf("%u", bit);
	}
	printf("\n");
	for (int i = 1; i <= BOARD_DIM; ++i) {
		unsigned int bit = (Squeres[y / 3 * 3 + x / 3] >> i) & 1;
		printf("%u", bit);
	}
	printf("\n");
}

