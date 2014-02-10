# To build a gibbs sampler for the phone-learning project

CC= g++
#CFLAGS = -c -xhost -parallel -O3 
CFLAGS = -c -Wall -O3
SOURCES = main.cc decoder.cc loader.cc model.cc cluster.cc gmm.cc mixture.cc util.cc data.cc calculator.cc 
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE = decode_to_pg 

all: $(SOURCES) $(EXECUTABLE) 

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ 

.cc.o:
	$(CC) $(CFLAGS)  $< -o $@ 

clean:
	rm -rf *.o $(EXECUTABLE) 
