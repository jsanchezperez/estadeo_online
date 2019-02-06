CFLAGS=-Wall -Wextra -O3 #-Werror   
LFLAGS=-lstdc++ -lm -lfftw3 -lfftw3f -fopenmp
INCLUDE=-I./src/ica -I./src

#object files
OBJ_ICA= bicubic_interpolation.o file.o inverse_compositional_algorithm.o mask.o matrix.o transformation.o zoom.o

OBJ_ESTADEO= color_bicubic_interpolation.o estadeo.o main.o utils.o

OBJ= $(OBJ_ICA) $(OBJ_ESTADEO)

#executable files
all: bin obj bin/estadeo

bin:
	mkdir -p bin

obj:
	mkdir -p obj

#generate executables
bin/estadeo: $(addprefix obj/,$(OBJ)) 
	g++ $^ -o $@ $(CFLAGS) $(LFLAGS)

#compile ica
obj/%.o: src/ica/%.cpp
	g++ -c $< -o $@ $(INCLUDE) $(CFLAGS) $(LFLAGS)

#compile estadeo 
obj/%.o: src/%.cpp
	g++ -c $< -o $@ $(INCLUDE) $(CFLAGS) $(LFLAGS)

clean: 
	rm -f bin/estadeo bin/generate_graphics
	rm -R obj

