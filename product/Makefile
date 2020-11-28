#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-08-04 11:12:04
#***
#************************************************************************************/
#
INSTALL_DIR := /usr/local

TARGET := imageclean

SOURCE :=  \
	prediction.cpp

TORCH_INSTALL_DIR :=/opt/libtorch

# INCS	:= $(shell pkg-config --cflags libtorch opencv)
INCS := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14 -I${TORCH_INSTALL_DIR}/include

# LIBS	:= $(shell pkg-config --libs libtorch opencv)

LIBS := \
 	-L${TORCH_INSTALL_DIR}/lib \
    -Wl,--no-as-needed \
	-lc10 -lc10_cuda -ltorch -ltorch_cuda -ltorch_cpu \
	-lcudart \
 	-lnimage -ljpeg -lpng -lm


DEFINES :=

CXXFLAGS := -fPIC
LDFLAGS := -fPIC -O3

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
# 	${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -rf core ${OBJECT} ${TARGET}
