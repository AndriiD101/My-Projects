# variables
CC=gcc
CFLAGS=-std=c11 -Wall -Werror
LDLIBS=-lm -lncurses

# target
autodrift: autodrift.c
	cppcheck --enable=unusedFunction --error-exitcode=1 $^
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

# remove compiled files
clean: 
	rm -rf $(OUTPUT) *.o