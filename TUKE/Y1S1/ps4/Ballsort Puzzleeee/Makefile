# variables
CC=gcc
CFLAGS=-std=c11 -Wall -Werror
LDLIBS=-lm
OUTPUT=ballsortpuzzle

# targets
all: $(OUTPUT)

$(OUTPUT): ballsortpuzzle.o main.o
	cppcheck --enable=performance,unusedFunction --error-exitcode=1 *.c
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $^ $(LDLIBS) -o $@

ballsortpuzzle.o: ballsortpuzzle.c ballsortpuzzle.h
	$(CC) $(CFLAGS) -c ballsortpuzzle.c $(LDLIBS) -o $@

# remove compiled files
clean: 
	rm -rf $(OUTPUT) *.o
