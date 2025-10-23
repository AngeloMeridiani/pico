CXXFLAGS=-g -O3 -Wno-deprecated -Wall
CCFLAGS=-g -O3 -g
LDFLAGS=-g -O3 -g #-lboost_iostreams-mt 

COLL_OBJS=coll/bcast.o \
          coll/reduce.o \
          coll/barrier.o \
          coll/allreduce.o \
          coll/alltoall.o \
          coll/gather.o \
          coll/scatter.o \
          coll/pairwise.o \
          coll/collective_registry.o \
          coll/collective_selector.o

OBJS=buffer_element.o cmdline.o process_trace.o schedgen.o $(COLL_OBJS)

force: all

cmdline.c: schedgen.ggo
	gengetopt < $<

%.o: %.cpp *hpp *h
	g++ $(CXXFLAGS) -c $<

%.o: %.c *h
	gcc $(CCFLAGS) -c $<

coll/%.o: coll/%.cpp
	g++ $(CXXFLAGS) -c $< -o $@

all: $(OBJS) schedgen.ggo
	g++ $(CXXFLAGS) $(OBJS) -o schedgen $(LDFLAGS)

clean:
	rm -f *.o
	rm -f coll/*.o
	rm -f cmdline.c cmdline.h
	rm -f schedgen
