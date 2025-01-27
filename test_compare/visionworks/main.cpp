#include "graph_mode_stabilizer.hpp"
#include "NVX/nvx.h"
#include "NVX/nvx_opencv_interop.hpp"
#include "NVX/nvx_timer.hpp"
#include "OVX/UtilityOVX.hpp"
#include "NVX/Application.hpp"
#include "OVX/FrameSourceOVX.hpp"
#include "OVX/RenderOVX.hpp"
struct EventData
{
  EventData(): stop(false) {}
  bool stop;
};
void keyboardEventCallback(void* eventData, vx_char key, vx_uint32 /*x*/, vx_uint32 /*y*/)
{
  EventData* data = static_cast<EventData*>(eventData);
  if (key == 27) // escape
  {
    data->stop = true;
  }
}
int main(int argc, char **argv)
{
  if (argc == 1)
  {
    std::cout << "Please specify a video file name..." << std::endl;
    return -1;
  }
  nvxio::Application &app = nvxio::Application::get();
  std::vector<std::string> result;
  app.setDescription("This sample demonstrates stabilization of video");
  app.allowPositionalParameters("Input video file name", &result);
  app.init(argc, argv);
  ovxio::ContextGuard context;
  {
    std::string sourceUri = result[0];
    std::unique_ptr<ovxio::FrameSource> frameSource(
        ovxio::createDefaultFrameSource(context, sourceUri));
    if (!frameSource || !frameSource->open())
    {
      std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
      return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
    }
    ovxio::FrameSource::Parameters frameConfig = frameSource->getConfiguration();
    vx_image frame = vxCreateImage(context,
                                   frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGBX);
    std::unique_ptr<ovxio::Render> render(
        ovxio::createDefaultRender(context, "Video stabilizer", frameConfig.frameWidth, frameConfig.frameHeight));
    if (!render)
    {
      std::cerr << "Error: Cannot create render!" << std::endl;
      return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
    }
    EventData eventData;
    render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);
    GraphModeStabilizer stabilizer(context);
    vx_image stabilized_frame;
    while(!eventData.stop)
    {
      ovxio::FrameSource::FrameStatus frameStatus = frameSource->fetch(frame);
      if (frameStatus == ovxio::FrameSource::FrameStatus::CLOSED)
      {
        std::cout << "End of the video file!" << std::endl;
        break;
      }
      NVX_TIMER(total, "TOTAL");
      stabilized_frame = stabilizer.process(frame);
      NVX_TIMEROFF(total);
      render->putImage(stabilized_frame);
      if (!render->flush())
      {
        eventData.stop = true;
      }
    }
  }
  return 0;
}

