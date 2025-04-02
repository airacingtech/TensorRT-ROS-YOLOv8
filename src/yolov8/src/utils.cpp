#include "utils.h"

std::string getFrameIdFromTopic(const std::string &camera_topic) {
    // Extract the relevant part from the topic name
    // Example: "/vimba_rear/image/ptr" -> "camera_rear"
    
    // Remove any leading '/' character
    std::string clean_topic = camera_topic;
    if (clean_topic.at(0) == '/') {
        clean_topic = clean_topic.substr(1);
    }
    
    // Find the first part of the topic (before the first '/')
    size_t pos = clean_topic.find('/');
    std::string topic_part = clean_topic;
    if (pos != std::string::npos) {
        topic_part = clean_topic.substr(0, pos);
    }
    
    // Extract the facing by removing "vimba_" prefix if it exists
    std::string facing = topic_part;
    if (topic_part.find("vimba_") == 0) {
        facing = topic_part.substr(6); // Length of "vimba_"
    }
    
    // Return the frame ID
    return "camera_" + facing;
}

bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat& img, int w, int h)
{
	cv::Mat result;
	int nw, nh;
	// int ih = img.rows;
	// int iw = img.cols;
	float aspectRatio = (float)img.cols / (float)img.rows;

	if (aspectRatio >= 1)
	{
		nw = w;
		nh = int(h / aspectRatio);
	}
	else
	{
		nw = int(w * aspectRatio);
		nh = h;
	}
	cv::resize(img, img, cv::Size(nw, nh));
	result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
	cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(h, w, CV_8UC3, 0.0);
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

	std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);

	return res_tuple;
}