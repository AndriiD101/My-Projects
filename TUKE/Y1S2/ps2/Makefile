CC = gcc
CFLAGS = -std=c11 -Werror -Wall -lm 
OBJECTS = main.o k.o hof.o
TARGET = ps2
LDLIBS = -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

hof.o: hof.c hof.h
	$(CC) $(CFLAGS) -c hof.c -o hof.o

k.o: k.c k.h
	$(CC) $(CFLAGS) -c k.c -o k.o

ui.o: ui.c ui.h
	$(CC) $(CFLAGS) -c ui.c -o ui.o

tests: bmp.o test_bmp.c
	$(CC) $(CFLAGS) bmp.c test_bmp.c $(LDLIBS) -o $@
	./tests


check:
	cppcheck *.c

clean:
	rm -f $(OBJECTS) $(TARGET)
