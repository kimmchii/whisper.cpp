CXX = g++
CXXFLAGS = -std=c++11 -I ./onnxruntime-linux-x64-1.12.1/include/
LDFLAGS = -L ./onnxruntime-linux-x64-1.12.1/lib/ -lonnxruntime -Wl,-rpath,./onnxruntime-linux-x64-1.12.1/lib/

test: silero-vad.cpp test.cpp
	$(CXX) $(CXXFLAGS) silero-vad.cpp test.cpp -o test $(LDFLAGS)

clean:
	rm -f test
