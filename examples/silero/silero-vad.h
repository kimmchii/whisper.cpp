// silero-vad.h 

#ifndef SILERO_VAD_H
#define SILERO_VAD_H

#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "wav.h"

struct Timestamps{
    float start;
    float end;
};
std::vector<Timestamps> PerformVad(std::string &audio_file_path);

#endif //SILERO_VAD_H