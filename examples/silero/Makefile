CXX = g++
CXXFLAGS = -std=c++11 -I ./onnxruntime-linux-x64-1.12.1/include/
LDFLAGS = -L ./onnxruntime-linux-x64-1.12.1/lib/ -lonnxruntime -Wl,-rpath,./onnxruntime-linux-x64-1.12.1/lib/
SRC = silero-vad.cpp test.cpp
EXECUTABLE = test

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXECUTABLE) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)
