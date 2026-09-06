#include <cstdio>
#include <cstdlib>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法: %s <输入图片> <输出图片> [块大小]\n", argv[0]);
        return 1;
    }
    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    const int block = (argc >= 4) ? std::atoi(argv[3]) : 8;
    if (block < 1) {
        std::fprintf(stderr, "块大小必须 >= 1\n");
        return 1;
    }

    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        std::fprintf(stderr, "无法读取图片: %s\n", input_path);
        return 1;
    }
    std::printf("输入: %s (%dx%d, block=%d)\n", input_path, width, height, block);

    auto t0 = std::chrono::steady_clock::now();

    for (int by = 0; by < height; by += block) {
        for (int bx = 0; bx < width; bx += block) {
            long sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
            const int y_end = (by + block < height) ? by + block : height;
            const int x_end = (bx + block < width)  ? bx + block : width;
            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    unsigned char* p = img + (y * width + x) * 3;
                    sum_r += p[0];
                    sum_g += p[1];
                    sum_b += p[2];
                    ++count;
                }
            }
            const unsigned char avg_r = (unsigned char)(sum_r / count);
            const unsigned char avg_g = (unsigned char)(sum_g / count);
            const unsigned char avg_b = (unsigned char)(sum_b / count);

            // 2. 把整块填成平均颜色
            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    unsigned char* p = img + (y * width + x) * 3;
                    p[0] = avg_r;
                    p[1] = avg_g;
                    p[2] = avg_b;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("像素化耗时: %.3f ms (串行, 1 线程)\n", ms);

    if (!stbi_write_png(output_path, width, height, 3, img, width * 3)) {
        std::fprintf(stderr, "无法写出图片: %s\n", output_path);
        stbi_image_free(img);
        return 1;
    }
    std::printf("输出: %s\n", output_path);

    stbi_image_free(img);
    return 0;
}
