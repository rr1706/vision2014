#ifndef CAMERAFRAME_HPP_
#define CAMERAFRAME_HPP_

#include <cstring>
#include <stdint.h>
#include <sys/time.h>
#ifdef WEBCAM_OCV
#include <opencv2/core/core.hpp>
#endif

class CameraFrame {
public:
	CameraFrame(void);
	CameraFrame(CameraFrame const &src);
	~CameraFrame(void);

	CameraFrame &operator=(CameraFrame const &src);

	bool IsValid(void) const;
	uint32_t GetWidth(void) const;
	uint32_t GetHeight(void) const;
	timeval GetTimestamp(void) const;

	uint8_t       *GetDataBGR(void);
	uint8_t const *GetDataBGR(void) const;

	void Update(uint32_t width, uint32_t height, timeval time);

#ifdef WEBCAM_OCV
    cv::Mat getMat();
#endif

private:
	uint32_t m_width;
	uint32_t m_height;
	size_t   m_length;
	uint8_t *m_data;
	timeval  m_time;

	void Resize(uint32_t width, uint32_t height);
};

#endif
