all:
	g++ -g gpu.cpp -o gpu.exe

debug:
	valgrind --leak-check=full --error-limit=no ./main.exe

clean:
	rm *.tar.gz
	rm gpu.exe
