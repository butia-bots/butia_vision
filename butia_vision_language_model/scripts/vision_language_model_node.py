#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from butia_vision_msgs.srv import VisualQuestionAnswering, VisualQuestionAnsweringRequest, VisualQuestionAnsweringResponse
import PIL
from ros_numpy import numpify
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage

try:
    from langchain_community.chat_models.ollama import ChatOllama
except:
    pass
try:
    from langchain_openai.chat_models import ChatOpenAI
except:
    pass
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
except:
    pass


class VisionLanguageModelNode:
    def __init__(self):
        self.read_parameters()
        if self.vlm_api_type == 'ollama':
            self.vlm = ChatOllama(model=self.vlm_api_model, base_url=self.vlm_api_host)
        elif self.vlm_api_type == 'openai':
            self.vlm = ChatOpenAI(model_name=self.vlm_api_model, openai_api_base=self.vlm_api_host)
        if self.vlm_api_type == 'google-genai':
            self.vlm = ChatGoogleGenerativeAI(model=self.vlm_api_model, convert_system_message_to_human=True)
        else:
            raise ValueError(f"VLM API type must be one of: {['ollama', 'openai', 'google-genai']}!")
        self.image_rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, callback=self._update_rgb_image)
        self.visual_question_answering_server = rospy.Service(self.visual_question_answering_service, VisualQuestionAnswering, handler=self._handle_visual_question_answering)

    def _update_rgb_image(self, msg: Image):
        self.rgb_image_msg = msg

    def _handle_visual_question_answering(self, req: VisualQuestionAnsweringRequest):
        message = HumanMessage(
            content=[
                self.get_image_content(),
                {
                    'type': 'text',
                    'text': f'{req.question}'
                }
            ]
        )
        res = VisualQuestionAnsweringResponse()
        res.answer = self.vlm.invoke([message,]).content
        res.confidence = 1.0
        return res

    def get_image_content(self):
        rospy.wait_for_message(self.rgb_image_topic, Image)
        buffered = BytesIO()
        img = PIL.Image.fromarray(numpify(self.rgb_image_msg)[:,:,::-1])
        img.save(buffered, format='JPEG')
        b64_image_str = base64.b64encode(buffered.getvalue()).decode()
        if self.vlm_api_type in ('ollama',):
            return {
                'type': 'image_url',
                'image_url': f"data:image/jpeg;base64,{b64_image_str}"
            }
        else:
            return {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{b64_image_str}"
                }
            }
        

    def read_parameters(self):
        self.vlm_api_type = rospy.get_param('~vlm_api_type')
        self.vlm_api_host = rospy.get_param('~vlm_api_host')
        self.vlm_api_model = rospy.get_param('~vlm_api_model')
        self.rgb_image_topic = rospy.get_param('~subscribers/image_rgb/topic')
        self.visual_question_answering_service = rospy.get_param('~servers/visual_question_answering/service')

if __name__ == '__main__':
    rospy.init_node('vision_language_model_node', anonymous=True)
    node = VisionLanguageModelNode()
    rospy.spin()