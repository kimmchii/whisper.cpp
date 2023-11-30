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

class VadIerator {
public:
    VadIerator(const std::string ModelPath, 
                int Sample_rate,
                int frame_size,
                float Threshold,
                int min_silence_duration_ms,
                int speech_pad_ms
            );
    void predict(const std::vector<float> &data);

};

void perform_vad(std::string &audio_file_path);

#endif //SILERO_VAD_H