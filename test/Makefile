LDFLAGS=-lopencv_core
CFLAGS=-std=c++11 -I../src
CC=g++
SOURCES=main.cpp findxyh.cpp arraytoip.cpp ../src/xyh.cpp ../src/util.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=./test

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $(EXECUTABLE) $(OBJECTS) $(LDFLAGS) $(OTHER_LDFLAGS)
.cpp.o:
	$(CC) -c $< -o $@ $(CFLAGS) $(OTHER_CFLAGS)
check: $(EXECUTABLE)
	$(EXECUTABLE)
clean:
	-rm -f $(OBJECTS) $(EXECUTABLE)

