# CudaSudokuSolver

Maksymalna liczba plansz w programie: <Ilość multiprocesorów>*<maksymalna ilość wątków na multiprocesor>
Program domyślnie pobiera dane z pliku "dane.txt" - można to zmienić w kodzie

Działanie programu:
1. SolveSudokuKernel szuka komórki z najmniejszą ilością możliwych elemntów do wpisania
2. Robi tyle kopii plansz ile jest możliwości-1, przekazując je do pamięci globalnej przyporządkowanej wolnym wątkom
Jesli nie ma wystarczająco wolnych wątków, to kernel się kończy i przechodzimy do backtrackingu
