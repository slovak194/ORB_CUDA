#pragma once

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

class multi_video_loader {
public:
    multi_video_loader(const std::vector<std::string>& video_file_paths, unsigned int start_time, unsigned int fps) {
        for (auto& video_file_path : video_file_paths) {
            m_caps.emplace_back(cv::VideoCapture(video_file_path, cv::CAP_FFMPEG));
            m_caps.back().set(cv::CAP_PROP_POS_FRAMES, start_time * fps);
        }
    }

    multi_video_loader(const std::string& video_file_path, unsigned int start_time, unsigned int fps) {
        m_caps.emplace_back(cv::VideoCapture(video_file_path, cv::CAP_FFMPEG));
        m_caps.back().set(cv::CAP_PROP_POS_FRAMES, start_time * fps);
    }

    std::vector<cv::Mat> next() {
        std::vector<cv::Mat> frames(m_caps.size());
        for (std::size_t i = 0; i < m_caps.size(); i++) {
            auto is_not_end = m_caps[i].read(frames[i]);

            if (!is_not_end) {
                frames.resize(0);
                return frames;
            }

            cv::cvtColor(frames[i], frames[i], cv::COLOR_BGR2GRAY);
        }
        return frames;
    }

private:
    std::vector<cv::VideoCapture> m_caps;
};

