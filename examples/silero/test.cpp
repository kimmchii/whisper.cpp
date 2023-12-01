#include "silero-vad.h"

int main(int argc, char *argv[]){
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <audio_file_path>" << std::endl;
        return 1;
    }
    std::string audio_file_path = argv[1];
    std::vector<Timestamps> hello;
    hello = PerformVad(audio_file_path);
}