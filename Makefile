all:
	g++ -g main.cpp -o main.exe

debug:
	valgrind --leak-check=full --error-limit=no ./main.exe
