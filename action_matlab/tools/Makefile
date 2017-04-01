# Makefile for make inference
WGET := wget
ECHO := echo
MATLABDIR ?= /sw/arcts/centos7/matlab/R2016a
MATLAB := $(MATLABDIR)/bin/matlab
MEX := $(MATLABDIR)/bin/mex
MEXEXT := $(shell $(MATLABDIR)/bin/mexext)
MEXFLAGS := -Iinclude CXXFLAGS="\$$CXXFLAGS -std=c++11"
TARGET1 := inference.$(MEXEXT)

.PHONY: all clean

all: $(TARGET1) $(TARGET2)

$(TARGET1): inference.cpp
	$(MEX) -output $@ $< $(MEXFLAGS)


clean:
	$(RM) $(TARGET1)
