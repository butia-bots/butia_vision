import smach
import rospy
import yaml
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from butia_vision_msgs.srv import ShelfClassificationResponse, ShelfClassificationRequest, ShelfClassification
from butia_vision_msgs.msg import Recognitions3D

class ShelfClassificationNode:
    def __init__(self, numberOfShelfs):
        self.numberOfShelfs = numberOfShelfs
        self.cv_bridge = CvBridge()
        self.initRosComm()

    def execute(self, userdata):
        if self.preempt_requested():
            return 'preempted'
        # userdata.shelfs = self.shelfs

        self.main()

        return 'succeeded'

    def request_preempt(self):
        super().request_preempt()

    def verifyIfPointInsideVolume(self, point: [float], volume: Marker):
        # volume is a Marker type
        # point is a list of 3 points
        # returns true if point is inside volume
        # returns false if point is outside volume
        if point[0] > volume.pose.position.x + volume.scale.x / 2 or point[0] < volume.pose.position.x - volume.scale.x / 2:
            return False
        if point[1] > volume.pose.position.y + volume.scale.y / 2 or point[1] < volume.pose.position.y - volume.scale.y / 2:
            return False
        if point[2] > volume.pose.position.z + volume.scale.z / 2 or point[2] < volume.pose.position.z - volume.scale.z / 2:
            return False
        return True

    def read_yaml_config(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_shelf_props(self):
        config_path = rospy.get_param('~config_path')
        config = self.read_yaml_config(config_path)

        # Calculating the center of the shelf area
        shelf_area = config['shelf_area']

        p1 = shelf_area[0]
        p2 = shelf_area[1]
        p3 = shelf_area[2]
        p4 = shelf_area[3]

        SHELF_HEIGHT = config['height']
        Z_VARIATION = SHELF_HEIGHT + 0.01 # Define how much z will vary for each marker

        x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
        y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        z = (p1[2] + p2[2] + p3[2] + p4[2]) / 4 + SHELF_HEIGHT * 2

        # Calculating the scale of the shelf area (in meters) of x, y and z is 10 meters + mean of z points

        x_scale = (p1[0] + p2[0]) / 2
        y_scale = (p1[1] + p4[1]) / 2
        z_scale = SHELF_HEIGHT # height of the shelf
        return [x, y, z], [x_scale, y_scale, z_scale], config['num_shelves']

    def get_most_common_object(self, shelfs, shelf):
        most_common_object = None
        count = 0
        for obj in shelfs[shelf]:
            if most_common_object is None:
                most_common_object = obj
                count = shelfs[shelf][obj]
            elif shelfs[shelf][obj] > shelfs[shelf][most_common_object]:
                most_common_object = obj
                count = shelfs[shelf][obj]
        return most_common_object, count

    def get_shelf_count(self, marker_array, num_shelves, objects):
        """
        Count how many objects are in each shelf.

        Args:
            marker_array (MarkerArray): Array of markers representing shelves.
            num_shelves (int): Number of shelves to consider.
            objects (list): List of objects with their shelf and label information.

        Returns:
            dict: A dictionary containing the count of objects in each shelf.
            The keys are the shelf IDs and the values are dictionaries
            where the keys are the object labels and the values are the counts.
            
        Example:
        {
            'shelf1': {
            'object1': 3,
            'object2': 2
            },
            'shelf2': {
            'object3': 1,
            'object4': 4
            },
            'shelf3': {
            'object5': 2,
            'object6': 1
            }
        }
        """
        shelfs = {}
                    
        for index, shelfs in enumerate(marker_array.markers[0:num_shelves]):
            shelfs[index] = {}

        for obj in objects:
            if obj['label'] in shelfs[obj['shelf']]:
                shelfs[obj['shelf']][obj['label']] += 1
            else:
                shelfs[obj['shelf']][obj['label']] = 1

        return shelfs
    
    def get_choiced_shelf(self, shelfs, labelObjectToPut):
        choiced_shelf = None
        shelf_with_less_objects = float('inf')
        for shelf in shelfs:
            if labelObjectToPut in shelfs[shelf]:
                return shelf
            if len(shelfs[shelf]) < shelf_with_less_objects:
                choiced_shelf = shelf
                shelf_with_less_objects = len(shelfs[shelf])
        return choiced_shelf

    def create_marker(self, z, center, scale):
        """
        Create a marker object with the specified parameters.

        Args:
            z (int or str): The z-coordinate or namespace of the marker.
            center (list): The x, y, and z coordinates of the marker's center.
            scale (list): The x, y, and z scales of the marker.

        Returns:
            Marker: The created marker object.
        """
        # cube marker

        marker = Marker()
        marker.header.frame_id = "map"
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2] - z * Z_VARIATION if isinstance(z, int) else center[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        if isinstance(z, int):
            marker.ns = "shelf_area" + str(z)
        else:
            marker.ns = z

        marker.id = 0
        marker.type = Marker.CUBE
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.lifetime = rospy.Duration()
        return marker

    def create_shelf_marker(self):
        marker_array = MarkerArray()

        shelf_area_center, shelf_area_scale, num_shelves = self.get_shelf_props()

        for i in range(num_shelves):
            marker = self.create_marker(i, shelf_area_center, shelf_area_scale)
            marker_array.markers.append(marker)

        return marker_array, num_shelves

    def create_object_marker(self, marker_array, objects):
        for obj in objects:
            marker = self.create_marker(obj['id'], obj['center'], [0.1, 0.1, 0.1])
            marker_array.markers.append(marker)
        return marker_array

    def objects_inside_shelf(self, marker_array, num_shelves, objects):
            for shelf in marker_array.markers[0:num_shelves]:
                for obj in objects:
                    if self.verifyIfPointInsideVolume(obj['center'], shelf):
                        obj['shelf'] = shelf.ns
                        rospy.loginfo(f'Object {obj["label"]} inside shelf: {shelf.ns}')
            return objects

    def map_objects_to_shelfs(self, shelfs, objects):
        # Attribute shelf to objects, where most of the object is inside the shelf
        shelf_per_object = {}
        objects_not_mapped = {}
        while True:
            shelf_with_most_objects = None
            obj_label = None
            count = 0
            for i, shelf in enumerate(shelfs):
                if shelf_with_most_objects is None:
                    shelf_with_most_objects = shelf
                    obj_label, count = self.get_most_common_object(shelfs, shelf)
                elif self.get_most_common_object(shelfs, shelf)[1] > count:
                    shelf_with_most_objects = shelf
                    obj_label, count = self.get_most_common_object(shelfs, shelf)
            if len(shelfs) == 0 or shelfs[shelf_with_most_objects] == {}:
                break
            rospy.loginfo(f'Shelf with most objects: {shelf_with_most_objects}')
            rospy.loginfo(f'Objects in shelf: {shelfs[shelf_with_most_objects]}')
            shelf_per_object[shelf_with_most_objects] = obj_label
            # Remove shelf with most objects from shelfs
            shelfs.pop(shelf_with_most_objects)

            # Remove the choiced object from all shelfs
            for shelf in shelfs:
                if obj_label in shelfs[shelf]:
                    shelfs[shelf].pop(obj_label)

            # Objects and the quantity of the removed shelf where the object is'n the choiced object
            for shelf in shelfs:
                for obj in shelfs[shelf]:
                    if obj in objects_not_mapped:
                        objects_not_mapped[obj] += shelfs[shelf][obj]
                    else:
                        objects_not_mapped[obj] = shelfs[shelf][obj]
            
             # Remove from not mapped objects
            if obj_label in objects_not_mapped:
                del objects_not_mapped[obj_label]

        if len(objects_not_mapped) > 0:
            # Redistribute objects not mapped in the rest of the shelfs
            obj_to_remove = []
            for obj in objects_not_mapped:
                if len(shelfs) == 0:
                    break
                shelf_choiced= list(shelfs.keys())[0]
                shelf_per_object[shelf_choiced] = obj
                del shelfs[shelf_choiced] 
                obj_to_remove.append(obj)
            for obj in obj_to_remove:
                del objects_not_mapped[obj]

        
        return shelf_per_object, objects_not_mapped

    def initRosComm(self):
      self.grasp_generator_srv = rospy.Service('butia_vision_msgs/shelf_classification', ShelfClassification, self.callback)
    
    def simplify_recognitions3d(self, recognitions3d):
        simplified_objects = []
        for description in recognitions3d.descriptions:
            simplified_object = {
                'id': description.id,
                'center': [
                    description.bbox.center.position.x,
                    description.bbox.center.position.y,
                    description.bbox.center.position.z
                ],
                'label': description.label
            }
            simplified_objects.append(simplified_object)
        return simplified_objects

    def callback(self, data: ShelfClassificationRequest):
        objects = self.simplify_recognitions3d(data.objects)
        labelObjectToPut: str = data.labelObjectToPut

        marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size = 2)

        marker_array, num_shelves = self.create_shelf_marker()

        marker_array = self.create_object_marker(marker_array, objects)  

        rospy.loginfo(f'----- Object inside each shelf -----')
        objects = self.objects_inside_shelf(marker_array, num_shelves, objects)    
        
        shelf = self.get_shelf_count(marker_array, num_shelves, objects)

        choiced_shelf = self.get_choiced_shelf(shelf, labelObjectToPut)

        response = ShelfClassificationResponse()
       
        response.response = choiced_shelf

        return response

if __name__ == '__main__':
  rospy.init_node('shelf_classification_node', anonymous = False)
  shelf_classification = ShelfClassificationNode()
  rospy.spin()
  rospy.on_shutdown(shelf_classification.shutdown)