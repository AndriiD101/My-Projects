CC = gcc
CFLAGS = -std=c11 -Werror -Wall -lm
OBJECTS = main.o bmp.o playfair.o
TARGET = ps1
LDLIBS = -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

bmp.o: bmp.c bmp.h
	$(CC) $(CFLAGS) -c bmp.c -o bmp.o

playfair.o: playfair.c playfair.h
	$(CC) $(CFLAGS) -c playfair.c -o playfair.o

tests: bmp.o test_bmp.c
	$(CC) $(CFLAGS) bmp.c test_bmp.c $(LDLIBS) -o $@
	./tests

check:
	cppcheck *.c

clean:
	rm -f $(OBJECTS) $(TARGET)

