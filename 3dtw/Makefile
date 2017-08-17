
CC = g++ 
CXXFLAGS= -std=c++11 -O3
CPPFLAGS=
SRC= src
BIN=bin
TIMESERIES=TimeSeries
TIMESERIES_SRC=$(SRC)/$(TIMESERIES).cpp
TIMESERIES_H = $(SRC)/$(TIMESERIES).h
TIMESERIES_O=$(BIN)/$(TIMESERIES).o
AA=aa
AA_BIN=$(BIN)/$(AA)
AA_SRC=$(SRC)/$(AA).cpp
AA_O=$(BIN)/$(AA).o


debug: CXXFLAGS += -g
debug:	clean all

#kimh ?= 1
ifeq ($(kimh), 1)
    CXXFLAGS += -DKIMH
endif

ifeq ($(kimm), 1)
    CXXFLAGS += -DKIMMINMAX
endif


#testall ?= 1
ifeq ($(testall), 1)
    CXXFLAGS += -DTEST_ALL
endif

ifeq ($(cv), 1)
    CXXFLAGS += -DCV
endif

ifeq ($(rec), 1)
    CXXFLAGS += -DS_RECURSIVE
endif




#kimh:  CPPFLAGS += -DKIMH
#kimh:	clean all

#testall: CPPFLAGS += -DTEST_ALL
#testall: clean all

#kimh testall: CPPFLAGS += -DKIMH -DTEST_ALL
#kimh testall: clean all

all: $(BIN) $(AA_BIN)

clean: 
	rm -fr $(BIN)

$(BIN):
	@mkdir $(BIN)

$(AA_O): $(AA_SRC) $(TIMESERIES_O)
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TIMESERIES_O) : $(TIMESERIES_SRC) $(TIMESERIES_H)
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(AA_BIN): $(AA_O) $(TIMESERIES_O)
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@


