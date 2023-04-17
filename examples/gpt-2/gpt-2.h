#ifndef GPT2_H
#define GPT2_H

#include <string>
#include <vector>
#include <map>

#include "ggml/ggml.h"
#include "utils.h"

struct gpt2_layer;
struct gpt2_model;

// default hparams (GPT-J 6B)
struct gpt2_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

bool gptj_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab);

bool gptj_eval(
        const gpt2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token);

#endif // GPT2_H