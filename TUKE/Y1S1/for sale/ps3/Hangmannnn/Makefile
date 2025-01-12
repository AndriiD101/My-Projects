# variables
CC=gcc
CFLAGS=-std=c11 -Wall -Werror
LDLIBS=-lm
OUTPUT=hangman

# targets
all: $(OUTPUT)

$(OUTPUT): hangman.o main.o
	cppcheck --enable=performance,unusedFunction --error-exitcode=1 *.c
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $^ $(LDLIBS) -o $@

hangman.o: hangman.c hangman.h
	$(CC) $(CFLAGS) -c hangman.c $(LDLIBS) -o $@

# remove compiled files
clean: 
	rm -rf $(OUTPUT) *.o


