// loader.cpp  (표준 헤더만 사용)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

static constexpr int IMG_H = 28;
static constexpr int IMG_W = 28;
static constexpr int IMG_SIZE = IMG_H * IMG_W; // 784
static constexpr int REC_LEN  = 1 + IMG_SIZE;  // 785

struct DataSet {
    vector<uint8_t> labels;         // [N]
    vector<uint8_t> images;         // [N * 784]
    size_t N = 0;

    static DataSet load_rec(const string& path) {
        DataSet ds;
        ifstream f(path, ios::binary);
        if (!f) throw runtime_error("cannot open: " + path);

        f.seekg(0, ios::end);
        auto fsize = f.tellg();
        f.seekg(0, ios::beg);
        if (fsize < 0 || (static_cast<size_t>(fsize) % REC_LEN) != 0) {
            throw runtime_error("file size not multiple of record length (785)");
        }
        ds.N = static_cast<size_t>(fsize) / REC_LEN;

        ds.labels.resize(ds.N);
        ds.images.resize(ds.N * IMG_SIZE);

        vector<uint8_t> buf(static_cast<size_t>(fsize));
        if (!f.read(reinterpret_cast<char*>(buf.data()), buf.size())) {
            throw runtime_error("failed to read entire file");
        }

        for (size_t i = 0; i < ds.N; ++i) {
            ds.labels[i] = buf[i * REC_LEN + 0];
            memcpy(&ds.images[i * IMG_SIZE], &buf[i * REC_LEN + 1], IMG_SIZE);
        }
        return ds;
    }
};

struct DataLoader {
    const DataSet& ds;
    int batch;
    bool shuffle_flag = true;
    bool drop_last = false;
    uint64_t seed = 42;

    vector<int> index;
    size_t cursor = 0;
    mt19937 rng;

    DataLoader(const DataSet& ds, int batch, bool shuffle_flag=true,
               bool drop_last=false, uint64_t seed=42)
        : ds(ds), batch(batch), shuffle_flag(shuffle_flag),
          drop_last(drop_last), seed(seed), rng(static_cast<uint32_t>(seed))
    {
        index.resize(ds.N);
        iota(index.begin(), index.end(), 0);
        reset_epoch();
    }

    void reset_epoch() {
        cursor = 0;
        if (shuffle_flag) {
            rng.seed(static_cast<uint32_t>(seed));
            std::shuffle(index.begin(), index.end(), rng);
        }
    }

    bool next_batch(vector<float>& X, vector<int>& y) {
        if (cursor >= ds.N) return false;

        size_t remain = ds.N - cursor;
        size_t B = static_cast<size_t>(batch);
        if (remain < B) {
            if (drop_last) return false;
            B = remain;
        }

        X.assign(B * IMG_SIZE, 0.0f);
        y.assign(B, 0);

        const float inv255 = 1.0f / 255.0f;
        for (size_t bi = 0; bi < B; ++bi) {
            size_t idx = index[cursor + bi];
            y[bi] = static_cast<int>(ds.labels[idx]);
            const uint8_t* src = &ds.images[idx * IMG_SIZE];
            float* dst = &X[bi * IMG_SIZE];
            for (int k = 0; k < IMG_SIZE; ++k) dst[k] = static_cast<float>(src[k]) * inv255;
        }
        cursor += B;
        return true;
    }
};

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);

    if (argc < 3) {
        cerr << "사용법: " << argv[0]
             << " <train.rec|test.rec> <batch_size> [--no-shuffle] [--drop-last] [--seed=42]\n";
        return 1;
    }
    string rec_path = argv[1];
    int batch_size = stoi(argv[2]);
    bool do_shuffle = true;
    bool drop_last = false;
    uint64_t seed = 42;

    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-shuffle") do_shuffle = false;
        else if (a == "--drop-last") drop_last = true;
        else if (a.rfind("--seed=", 0) == 0) seed = stoull(a.substr(7));
        else { cerr << "알 수 없는 옵션: " << a << "\n"; return 1; }
    }

    try {
        cout << "[load] " << rec_path << "\n";
        DataSet ds = DataSet::load_rec(rec_path);
        cout << "  N=" << ds.N << " samples\n";

        DataLoader loader(ds, batch_size, do_shuffle, drop_last, seed);
        vector<float> X; vector<int> y;

        size_t total = 0, batches = 0;
        for (int step = 0; step < 3; ++step) {
            if (!loader.next_batch(X, y)) break;
            ++batches;
            total += y.size();
            cout << "  batch " << batches << " size=" << y.size()
                 << "  y0=" << y[0]
                 << "  X0[0..3]=[" << X[0] << "," << X[1] << "," << X[2] << "]\n";
        }
        cout << "[done] preview batches=" << batches
             << ", samples_shown=" << total << "\n";
    } catch (const exception& e) {
        cerr << "오류: " << e.what() << "\n";
        return 1;
    }
    return 0;
}