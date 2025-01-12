CC = gcc
CFLAGS = -std=c11 -Werror -Wall -lm
OBJECTS = main.o bmp.o transformations.o
TARGET = ps4
LDLIBS = -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

transformations.o: transformations.c
	$(CC) $(CFLAGS) -c transformations.c -o transformations.o

bmp.o: bmp.c
	$(CC) $(CFLAGS) -c bmp.c -o bmp.o
check:
	cppcheck *.c

clean:
	rm -f $(OBJECTS) $(TARGET)
