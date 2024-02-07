import rospy

from PIL import Image
from PIL import ImageDraw
from transformers import pipeline
from butia_vision_msgs.msg import Recognitions2D

class OwlVitRecognition:
    def __init__(self):
        self.detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
        self.message_pub = rospy.Publisher("butia_vision/br/recognition2D", Recognitions2D, queue_size=1)

    def read_params(self):
        self.prompt = rospy.get_param("~prompt", [])

    def callback(self, data):
        image = Image.open(data.image).convert("RGB")

        predictions = self.detector(
            image,
            candidate_labels=[self.prompt],
        )

        draw = ImageDraw.Draw(image)
        for prediction in predictions:
            label = prediction['label']
            score = prediction['score']
            box = prediction['box']
            
            xmin, ymin, xmax, ymax = box.values()
            
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
            
            draw.text((xmin, ymin), f"{label} ({score:.2f})", fill="red")

            self.message_pub.publish(Recognitions2D(
                header=data.header,
                objects=[label],
                x=[xmin],
                y=[ymin],
                width=[xmax-xmin],
                height=[ymax-ymin],
                confidence=[score]
            ))

        image.show()

if __name__ == '__main__':
    rospy.init_node('owl_vit_node', anonymous = True)

    owl_vit = OwlVitRecognition()

    rospy.spin()
